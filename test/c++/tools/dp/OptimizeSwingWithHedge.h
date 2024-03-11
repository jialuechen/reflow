
#ifndef OPTIMIZESWINGWITHHEDGE_H
#define OPTIMIZESWINGWITHHEDGE_H
#include <Eigen/Dense>
#include "libflow/core/grids/Interpolator.h"
#include "libflow/core/grids/LinearInterpolator.h"
#include "libflow/core/utils/StateWithStocks.h"
#include "libflow/regression/ContinuationValue.h"
#include "libflow/regression/GridAndRegressedValue.h"
#include "libflow/core/grids/SpaceGrid.h"
#include "libflow/regression/BaseRegression.h"
#include "libflow/dp/OptimizerDPBase.h"

template< class PayOff, class Simulator >
class OptimizeSwingWithHedge : public libflow::OptimizerDPBase
{
private :

    PayOff m_payoff;///< pay off function
    int m_nPointStock ; ///< number of point stocks
    /// store the simulator
    std::shared_ptr<Simulator> m_simulator;

public :

    /// \brief Constructor
    /// \param p_payoff pay off used
    /// \param  p_nPointStock number of stock points
    OptimizeSwingWithHedge(const PayOff &p_payoff, const int &p_nPointStock): m_payoff(p_payoff), m_nPointStock(p_nPointStock) {}

    /// \brief define the diffusion cone for parallelism
    /// \param p_regionByProcessor             region (min max) treated by the processor for the different regimes treated
    /// \return returns in each dimension the min max values in the stock that can be reached from the grid p_gridByProcessor for each regime
    std::vector< std::array< double, 2> > getCone(const  std::vector<  std::array< double, 2>  > &p_regionByProcessor) const
    {
        std::vector< std::array< double, 2> > extrGrid(p_regionByProcessor);
        // only a single  exercise
        extrGrid[0][1] += 1.;
        return extrGrid;
    }

    /// \brief defines the dimension to split for MPI parallelism
    ///        For each dimension return true is the direction can be split
    Eigen::Array< bool, Eigen::Dynamic, 1> getDimensionToSplit() const
    {
        Eigen::Array< bool, Eigen::Dynamic, 1> bDim = Eigen::Array< bool, Eigen::Dynamic, 1>::Constant(1, true);
        return  bDim ;
    }

    /// \brief defines a step in optimization
    /// Notice that this implementation is not optimal. In fact no interpolation is necessary for this asset.
    /// This implementation is for test and example purpose
    /// \param p_grid      grid at arrival step after command
    /// \param p_stock     coordinate of the stock point to treat
    /// \param p_condEsp   continuation values for each regime
    /// \param p_phiIn     for each regime  gives the solution calculated at the previous step ( next time step by Dynamic Programming resolution)
    /// \return   a pair  :
    ///              - for each regimes (column) gives the solution for each particle (row)
    ///              - for each control (column) gives the optimal control for each particle (rows)
    ///              .
    virtual std::pair< Eigen::ArrayXXd, Eigen::ArrayXXd>  stepOptimize(const std::shared_ptr< libflow::SpaceGrid>  &p_grid,
            const Eigen::ArrayXd   &p_stock, const std::vector<libflow::ContinuationValue> &p_condEsp,
            const std::vector < std::shared_ptr< Eigen::ArrayXXd > > &p_phiIn) const
    {
        // actualization
        double actuStep = m_simulator->getActuStep(); // for one step
        int nbSimul = p_condEsp[0].getNbSimul();
        std::pair< Eigen::ArrayXXd, Eigen::ArrayXXd> solutionAndControl;
        solutionAndControl.first.resize(nbSimul, 2);
        solutionAndControl.second.resize(nbSimul, 1);
        Eigen::ArrayXd payOffVal = m_payoff.applyVec(m_simulator->getParticles()).array();
        // create interpolator at current stock point
        std::shared_ptr<libflow::Interpolator>  interpolatorCurrentStock = p_grid->createInterpolator(p_stock);
        // cash flow at current stock and previous step
        Eigen::ArrayXd cashSameStock = interpolatorCurrentStock->applyVec(*p_phiIn[0]);
        // tangent process at current stock and previous step
        Eigen::ArrayXd  tangentSameStock =  interpolatorCurrentStock->applyVec(*p_phiIn[1]);
        // conditional expectation at current stock point
        Eigen::ArrayXd condExpSameStock = actuStep * p_condEsp[0].getAllSimulations(*interpolatorCurrentStock);
        // get tangent process
        Eigen::ArrayXXd tangentProcess =  m_simulator->getTangent();
        // derivative of the pay off
        Eigen::ArrayXXd derivative = m_payoff.getDerivativeVec(p_condEsp[0].getParticles().matrix()).array();
        if (p_stock(0) < m_nPointStock)
        {
            // calculation detailed for clarity
            // create interpolator at next stock point accessible
            Eigen::ArrayXd nextStock(p_stock);
            nextStock(0) += 1;
            std::shared_ptr<libflow::Interpolator>  interpolatorNextStock = p_grid->createInterpolator(nextStock);
            // cash flow at next stock previous step
            Eigen::ArrayXd cashNextStock = interpolatorNextStock->applyVec(*p_phiIn[0]);
            // tangent process at next stock previous step
            Eigen::ArrayXd tangentNextStock =  interpolatorNextStock->applyVec(*p_phiIn[1]);
            // conditional expectation at next stock
            Eigen::ArrayXd condExpNextStock = actuStep * p_condEsp[0].getAllSimulations(*interpolatorNextStock);
            // arbitrage
            for (int is = 0; is < nbSimul; ++is)
                if (payOffVal(is) + condExpNextStock(is)  > condExpSameStock(is))
                {
                    solutionAndControl.first(is, 0) = payOffVal(is) + actuStep * cashNextStock(is);
                    solutionAndControl.second(is, 0) = 1;
                    solutionAndControl.first(is, 1) = derivative(0, is) * tangentProcess(0, is) + actuStep * tangentNextStock(is);
                }
                else
                {
                    solutionAndControl.first(is, 0) =  actuStep * cashSameStock(is);
                    solutionAndControl.second(is, 0) = 0.;
                    solutionAndControl.first(is, 1) = actuStep * tangentSameStock(is);

                }
        }
        else
        {
            solutionAndControl.first.col(0).setConstant(0.);
            solutionAndControl.second.col(0).setConstant(0.);
            solutionAndControl.first.col(1).setConstant(0.);
        }
        return solutionAndControl;
    }

/// \brief defines a step in simulation
/// Notice that this implementation is not optimal. In fact no interpolation is necessary for this asset.
/// This implementation is for test and example purpose
/// \param p_continuation  defines the continuation operator for each regime
/// \param p_state         defines the state value (modified)
/// \param p_phiInOut      defines the value function (modified): size number of functions to follow
    void stepSimulate(const std::shared_ptr< libflow::SpaceGrid> &, const std::vector< libflow::GridAndRegressedValue  > &p_continuation,
                      libflow::StateWithStocks &p_state,
                      Eigen::Ref<Eigen::ArrayXd> p_phiInOut) const
    {
        // actualization
        double actu = m_simulator->getActu(); // at time zero
        double actuStep = m_simulator->getActuStep(); // for one step
        //  no  delta hedging
        p_phiInOut(1) = 0. ;
        // only if stock not maximal
        if (p_state.getPtStock()(0) < m_nPointStock)
        {
            double payOffVal = m_payoff.apply(p_state.getStochasticRealization());
            double continuationValue = actuStep * p_continuation[0].getValue(p_state.getPtStock(), p_state.getStochasticRealization());
            if (p_state.getPtStock()(0) < m_nPointStock)
            {
                Eigen::ArrayXd nextStock(p_state.getPtStock());
                nextStock(0) += 1;
                double continuationValueNext = actuStep * p_continuation[0].getValue(nextStock, p_state.getStochasticRealization());
                if (payOffVal + continuationValueNext > continuationValue)
                {
                    p_state.setPtStock(nextStock);
                    p_phiInOut(0) += payOffVal * actu;
                    // delta for hedging (with next stock)
                    double delta = p_continuation[1].getValue(nextStock, p_state.getStochasticRealization());
                    p_phiInOut(1) = actu * delta;
                }
                else
                {
                    // delta for hedging (with current stock)
                    double delta = p_continuation[1].getValue(p_state.getPtStock(), p_state.getStochasticRealization());
                    // hedge position
                    p_phiInOut(1) = actu * delta;
                }
            }
        }
    }

    /// \brief Defines a step in simulation using interpolation in controls
    virtual void stepSimulateControl(const std::shared_ptr< libflow::SpaceGrid> &, const std::vector< libflow::GridAndRegressedValue  > &,
                                     libflow::StateWithStocks &,
                                     Eigen::Ref<Eigen::ArrayXd>) const {}

    /// \brief get number of regimes
    inline int getNbRegime() const
    {
        return 2;
    }

    /// \brief number of controls
    inline int getNbControl() const
    {
        return 1;
    }

    ///\brief store the simulator
    inline void setSimulator(const std::shared_ptr<Simulator> &p_simulator)
    {
        m_simulator = p_simulator ;
    }

    /// \brief get the simulator back
    inline std::shared_ptr< libflow::SimulatorDPBase > getSimulator() const
    {
        return m_simulator ;
    }

    /// \brief get size of the  function to follow in simulation
    ///        here two : follow hedge so give hedging position
    inline int getSimuFuncSize() const
    {
        return 2;
    }
}
;
#endif /* OPTIMIZESWINGWITHHEDGE_H */
