
#ifndef OPTIMIZEFICTITIOUSSWING_H
#define OPTIMIZEFICTITIOUSSWING_H
#include <memory>
#include <Eigen/Dense>
#include "libflow/core/utils/StateWithStocks.h"
#include "libflow/core/grids/SpaceGrid.h"
#include "libflow/regression/BaseRegression.h"
#include "libflow/core/grids/Interpolator.h"
#include "libflow/core/grids/LinearInterpolator.h"
#include "libflow/regression/ContinuationValue.h"
#include "libflow/regression/GridAndRegressedValue.h"
#include "libflow/dp/OptimizerDPBase.h"

/** \file OptimizeFictitiousSwing.h
 *  \brief Give an example of a fictitious n dimensional swing
 *  \author Xavier Warin
 */

/// \class OptimizeFictitiousSwing OptimizeFictitiousSwing.h
///  Defines a fictitious swing on Ndim stocks : the valorisation will be equal to the valorisation of ndim swing
///  where ndim is the number of stocks
template< class PayOff, class Simulator>
class OptimizeFictitiousSwing: public libflow::OptimizerDPBase
{
private :

    PayOff m_payoff;///< pay off function
    int m_nPointStock ; ///< number of point stocks per dimension
    int m_ndim ; ///< number of stocks to deal with
    int m_numberOfRemainingDates ; // number of remaining dates
    /// store the simulator
    std::shared_ptr<Simulator> m_simulator;

public :

    /// \brief Constructor
    /// \param  p_payoff pay off used
    /// \param  p_nPointStock number of stock points
    /// \param  p_ndim  dimension of the problem (stock number)
    OptimizeFictitiousSwing(const PayOff &p_payoff, const int &p_nPointStock, const int &p_ndim): m_payoff(p_payoff), m_nPointStock(p_nPointStock),
        m_ndim(p_ndim) {}


    /// \brief define the diffusion cone for parallelism
    /// \param  p_regionByProcessor         region (min max) treated by the processor for the different regimes treated
    /// \return returns in each dimension the min max values in the stock that can be reached from the grid p_gridByProcessor for each regime
    std::vector< std::array< double, 2> > getCone(const  std::vector<  std::array< double, 2>  > &p_regionByProcessor) const
    {
        std::vector< std::array< double, 2> > extrGrid(p_regionByProcessor);
        // only a single  exercise
        for (int id = 0; id < m_ndim; ++id)
            extrGrid[id][1] += 1.;
        return extrGrid;
    }

    /// \brief defines the dimension to split for MPI parallelism
    ///        For each dimension return true is the direction can be split
    Eigen::Array< bool, Eigen::Dynamic, 1> getDimensionToSplit() const
    {
        Eigen::Array< bool, Eigen::Dynamic, 1> bDim = Eigen::Array< bool, Eigen::Dynamic, 1>::Constant(m_ndim, true);
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
    /// \return for each regimes (column) gives the solution for each particle (row)
    std::pair< Eigen::ArrayXXd, Eigen::ArrayXXd>   stepOptimize(const std::shared_ptr< libflow::SpaceGrid> &p_grid, const Eigen::ArrayXd   &p_stock,
            const std::vector<libflow::ContinuationValue> &p_condEsp,
            const std::vector < std::shared_ptr< Eigen::ArrayXXd > > &p_phiIn) const
    {
        int nbSimul = p_condEsp[0].getNbSimul();
        std::pair< Eigen::ArrayXXd, Eigen::ArrayXXd> solutionAndControl;
        solutionAndControl.first.resize(nbSimul, 1);
        solutionAndControl.second.resize(nbSimul, 1);
        Eigen::ArrayXd payOffVal = m_payoff.applyVec(m_simulator->getParticles()).array();
        // create interpolator at current stock point
        std::shared_ptr<libflow::Interpolator>  interpolatorCurrentStock = p_grid->createInterpolator(p_stock);
        // cash flow at current stock and previous step
        Eigen::ArrayXd cashSameStock = interpolatorCurrentStock->applyVec(*p_phiIn[0]);
        // actualization
        double actuStep = m_simulator->getActuStep(); // for one step
        // conditional expectation at current stock point
        Eigen::ArrayXd condExpSameStock = actuStep * p_condEsp[0].getAllSimulations(*interpolatorCurrentStock);
        // number of possibilities for arbitrage with m_ndim stocks
        int nbArb = (0x01 << m_ndim);
        // stock to add
        Eigen::ArrayXd stockToAdd(m_ndim);
        // store optimal conditional  expectation
        Eigen::ArrayXd condExpOpt = 	Eigen::ArrayXd::Constant(nbSimul, -1e6);
        for (int j = 0 ; j < nbArb; ++j)
        {
            unsigned int ires = j ;
            for (int id = m_ndim - 1 ; id >= 0  ; --id)
            {
                unsigned int idec = (ires >> id) ;
                stockToAdd(id) = idec;
                ires -= (idec << id);
            }
            // calculation detailed for clarity
            // create interpolator at next stock point accessible
            Eigen::ArrayXd nextStock = p_stock + stockToAdd;
            // test that stock is possible
            if (nextStock.maxCoeff() <= m_nPointStock)
            {
                std::shared_ptr<libflow::Interpolator>  interpolatorNextStock = p_grid->createInterpolator(nextStock);
                // cash flow at next stock previous step
                Eigen::ArrayXd cashNextStock = interpolatorNextStock->applyVec(*p_phiIn[0]);
                // conditional expectation at next stock
                Eigen::ArrayXd condExpNextStock = actuStep * p_condEsp[0].getAllSimulations(*interpolatorNextStock);
                // quantity exercised
                double qExercized = stockToAdd.sum();
                // arbitrage
                for (int is = 0; is < condExpOpt.size(); ++is)
                {
                    double condExp = payOffVal(is) * qExercized + condExpNextStock(is);
                    if (condExp  > condExpOpt(is))
                    {
                        solutionAndControl.first(is, 0) = payOffVal(is) * qExercized + actuStep * cashNextStock(is);
                        condExpOpt(is) = condExp;
                        solutionAndControl.second(is, 0) = qExercized;
                    }
                }
            }
        }
        return solutionAndControl;
    }

    /// \brief get number of regimes
    inline int getNbRegime() const
    {
        return 1;
    }

    /// \brief number of controls
    inline int getNbControl() const
    {
        return 1;
    }

    /// \brief defines a step in simulation
    /// Notice that this implementation is not optimal. In fact no interpolation is necessary for this asset.
    /// This implementation is for test and example purpose
    /// \param p_continuation  defines the continuation operator for each regime
    /// \param p_state         defines the state value (modified)
    /// \param p_phiInOut      defines the value function (modified): size number of functions to follow
    void stepSimulate(const std::shared_ptr< libflow::SpaceGrid> &, const std::vector< libflow::GridAndRegressedValue  > &p_continuation,
                      libflow::StateWithStocks &p_state,  Eigen::Ref<Eigen::ArrayXd> p_phiInOut) const
    {
        // number of possibilities for arbitrage with m_ndim stocks
        int nbArb = (0x01 << m_ndim);

        // stock to add
        Eigen::ArrayXd stockToAdd(m_ndim);
        // actualization
        double actu = m_simulator->getActu(); // at time zero
        double actuStep = m_simulator->getActuStep(); // for one step
        // initialize
        double phiAdd = -1e6;
        // max of conditional expectation
        double expectationMax = -1e6;
        Eigen::ArrayXd ptStockMax(p_state.getPtStock().size());
        double payOffVal = m_payoff.apply(p_state.getStochasticRealization());
        for (int j = 0 ; j < nbArb; ++j)
        {
            unsigned int ires = j ;
            for (int id = m_ndim - 1 ; id >= 0  ; --id)
            {
                unsigned int idec = (ires >> id) ;
                stockToAdd(id) = idec;
                ires -= (idec << id);
            }
            Eigen::ArrayXd nextStock = p_state.getPtStock() + stockToAdd;
            // test that stock is possible
            if (nextStock.maxCoeff() <= m_nPointStock)
            {
                // quantity exercised
                double qExercized = stockToAdd.sum();
                double continuationValueNext = actuStep * p_continuation[0].getValue(nextStock, p_state.getStochasticRealization());
                double expectation = payOffVal * qExercized + continuationValueNext;
                if (expectation > expectationMax)
                {
                    ptStockMax = nextStock;
                    phiAdd = payOffVal * qExercized * actu;
                    expectationMax = expectation;
                }
            }
        }
        p_state.setPtStock(ptStockMax);
        p_phiInOut(0) += phiAdd;
    }

    /// \brief Defines a step in simulation using interpolation in controls
    virtual void stepSimulateControl(const std::shared_ptr< libflow::SpaceGrid> &, const std::vector< libflow::GridAndRegressedValue  > &,
                                     libflow::StateWithStocks &,
                                     Eigen::Ref<Eigen::ArrayXd>) const {}


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
    inline int getSimuFuncSize() const
    {
        return 1;
    }
}
;
#endif /* OPTIMIZEFICTITIOUSSWING_H */
