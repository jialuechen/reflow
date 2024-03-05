
#ifndef OPTIMIZELAKE_H
#define OPTIMIZELAKE_H
#include <memory>
#include <Eigen/Dense>
#include "libflow/core/utils/constant.h"
#include "libflow/core/utils/StateWithStocks.h"
#include "libflow/core/grids/SpaceGrid.h"
#include "libflow/regression/BaseRegression.h"
#include "libflow/core/grids/Interpolator.h"
#include "libflow/regression/ContinuationValue.h"
#include "libflow/regression/GridAndRegressedValue.h"
#include "libflow/dp/OptimizerDPBase.h"

/** \file OptimizeLake.h
 *  \brief  Simple example of a  lake optimizer. Stochasticity comes from the flows
 *          In this very simple model, gains are constant equal to one.
 *          Designed to work in parallel/multi threaded framework
 *  \author Xavier Warin
 */

/// \class OptimizeLake OptimizeLake.h
/// Defines a simple lake for optimization and simulation
template< class ARModel>
class OptimizeLake : public libflow::OptimizerDPBase
{
private :
    /// \brief Physical constraints
    //@{
    double m_withdrawalRate ; ///< withdrawal rate (volume) per time step
    //@}

    /// store the simulator
    std::shared_ptr<ARModel> m_simulator;

public :

    /// \brief Constructor
    /// \param  p_withdrawalRate    withdrawal rate between two time steps
    OptimizeLake(const double &p_withdrawalRate):  m_withdrawalRate(p_withdrawalRate) {}

    /// \brief define the diffusion cone for parallelism
    /// \param  p_regionByProcessor         region (min max) treated by the processor for the different regimes treated
    /// \return returns in each dimension the min max values in the stock that can be reached from the grid p_gridByProcessor for each regime
    std::vector< std::array< double, 2> > getCone(const  std::vector<  std::array< double, 2>  > &p_regionByProcessor) const
    {
        std::vector< std::array< double, 2> > extrGrid(1);
        extrGrid[0][0] = p_regionByProcessor[0][0] - m_withdrawalRate;
        extrGrid[0][1] = libflow::infty ;
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
    /// \param p_grid      grid at arrival step after command
    /// \param p_stock     coordinate of the stock point to treat
    /// \param p_condEsp   continuation values for each regime
    /// \param p_phiIn     for each regime  gives the solution calculated at the previous step ( next time step by Dynamic Programming resolution)
    /// \return   a pair  :
    ///              - for each regimes (column) gives the solution for each particle (row)
    ///              - for each control (column) gives the optimal control for each particle (rows)
    ///              .
    virtual std::pair< Eigen::ArrayXXd, Eigen::ArrayXXd>   stepOptimize(const   std::shared_ptr< libflow::SpaceGrid> &p_grid,
            const Eigen::ArrayXd   &p_stock,
            const std::vector<libflow::ContinuationValue> &p_condEsp,
            const std::vector < std::shared_ptr< Eigen::ArrayXXd > > &p_phiIn) const
    {
        int nbSimul = p_condEsp[0].getNbSimul();
        std::pair< Eigen::ArrayXXd, Eigen::ArrayXXd> solutionAndControl;
        // to store final solution (here one regime) and the optimal control
        solutionAndControl.first.resize(nbSimul, 1);
        solutionAndControl.second.resize(nbSimul, 1);
        // Inflows (one lake)
        Eigen::ArrayXd inflows =  m_simulator->getParticles().array().row(0).transpose();
        // level min, max of the stock
        double minStorage = p_grid->getExtremeValues()[0][0];
        double maxStorage = p_grid->getExtremeValues()[0][1];
        // Suppose that no withdrawal
        // create interpolator at current stock point
        Eigen::ArrayXd condExpSameStock(inflows.size()), cashSameStock(inflows.size());
        // due to inflows all simulations have to be done separately
        for (int is = 0; is < inflows.size(); ++is)
        {
            // stock:  if too much, "deverse"
            Eigen::ArrayXd  stockWithInflows(1);
            stockWithInflows(0) = std::min(p_stock(0) + inflows(is), maxStorage);
            if (p_grid->isInside(stockWithInflows))
            {
                std::shared_ptr<libflow::Interpolator>  interpolatorCurrentStock = p_grid->createInterpolator(stockWithInflows);
                // cash flow at current stock and previous step
                cashSameStock(is) = interpolatorCurrentStock->apply(p_phiIn[0]->row(is).transpose());
                // conditional expectation at current stock point
                condExpSameStock(is) =  p_condEsp[0].getASimulation(is, *interpolatorCurrentStock) ;
            }
            else
            {
                cashSameStock(is) = -libflow::infty;
                condExpSameStock(is) = -libflow::infty;
            }
        }
        // withdrawal
        Eigen::ArrayXd  gainWithdrawal(inflows.size()), cashWithdrawalStock(inflows.size()), condExpWithdrawalStock(inflows.size());
        // withdrawal per simulation
        Eigen::ArrayXd withdrawalMax(nbSimul);
        // due to inflows all simulations have to be done separately
        for (int is = 0; is < inflows.size(); ++is)
        {
            double stockWithIn = p_stock(0) + inflows(is);
            withdrawalMax(is) = std::min(stockWithIn - minStorage, m_withdrawalRate);
            if (libflow::isStrictlyLesser(withdrawalMax(is), 0.))
            {
                cashWithdrawalStock(is) = -libflow::infty;
                condExpWithdrawalStock(is)  = -libflow::infty;
                gainWithdrawal(is) = -libflow::infty;
            }
            else
            {
                Eigen::ArrayXd stockWithInflows(1);
                // outflow  if too much
                stockWithInflows(0) = std::min(stockWithIn - withdrawalMax(is), maxStorage);
                // Interpolator
                std::shared_ptr<libflow::Interpolator> interpolatorWithdrawalStock = p_grid->createInterpolator(stockWithInflows);
                // cash flow  at previous step at withdrawal
                cashWithdrawalStock(is) = interpolatorWithdrawalStock->apply(p_phiIn[0]->row(is).transpose());
                // conditional expectation at withdrawal stock level
                condExpWithdrawalStock(is) = p_condEsp[0].getASimulation(is, *interpolatorWithdrawalStock);
                // instantaneous gain if withdrawal (one par unit)
                gainWithdrawal(is) =  withdrawalMax(is) ;
            }
        }

        // arbitrage
        for (int is = 0; is < inflows.size(); ++is)
        {
            solutionAndControl.first(is, 0) = cashSameStock(is);
            solutionAndControl.second(is, 0) = 0.;
            double espCondMax = condExpSameStock(is);
            double espCondWithdrawal = gainWithdrawal(is) + condExpWithdrawalStock(is);
            if (espCondWithdrawal > espCondMax)
            {
                solutionAndControl.first(is, 0) = gainWithdrawal(is) + cashWithdrawalStock(is);
                solutionAndControl.second(is, 0) = -withdrawalMax(is);
            }
        }
        return solutionAndControl;
    }


    /// \brief Defines a step in simulation using interpolation in controls
    /// \param p_grid          grid at arrival step after command
    /// \param p_control       defines the controls
    /// \param p_state         defines the state value (modified)
    /// \param p_phiInOut      defines the value function (modified): size number of functions to follow
    virtual void stepSimulateControl(const std::shared_ptr< libflow::SpaceGrid>   &p_grid, const std::vector< libflow::GridAndRegressedValue > &p_control,
                                     libflow::StateWithStocks &p_state,
                                     Eigen::Ref<Eigen::ArrayXd>  p_phiInOut) const
    {

        Eigen::ArrayXd ptStock = p_state.getPtStock();
        // size of the stock
        double maxStorage = p_grid->getExtremeValues()[0][1];
        double minStorage =  p_grid->getExtremeValues()[0][0];
        //  inflow
        double inflow = p_state.getStochasticRealization()(0);
        //  get back control
        double control = p_control[0].getValue(ptStock, p_state.getStochasticRealization());
        ptStock(0) += inflow + control;
        ptStock(0) = std::max(std::min(maxStorage, ptStock(0)), minStorage);
        // for return
        p_state.setPtStock(ptStock);
        p_phiInOut(0) -= control  ;
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
    /// \param p_grid          grid at arrival step after command
    /// \param p_continuation  defines the continuation operator for each regime
    /// \param p_state         defines the state value (modified)
    /// \param p_phiInOut      defines the value function (modified): size number of functions to follow
    void stepSimulate(const std::shared_ptr< libflow::SpaceGrid>   &p_grid, const std::vector< libflow::GridAndRegressedValue  > &p_continuation,
                      libflow::StateWithStocks &p_state, Eigen::Ref<Eigen::ArrayXd> p_phiInOut) const
    {
        Eigen::ArrayXd ptStock = p_state.getPtStock();
        Eigen::ArrayXd ptStockMax = ptStock;
        // size of the stock
        double maxStorage = p_grid->getExtremeValues()[0][1];
        //  inflow
        double inflow = p_state.getStochasticRealization()(0);
        // update storage
        Eigen::ArrayXd ptStockCur(ptStock);
        ptStockCur(0) += inflow;
        ptStockCur(0) = std::min(ptStockCur(0), maxStorage); // outflow if necessary
        // if do nothing
        double espCondMax = - libflow::infty;
        if (p_grid->isInside(ptStockCur))
        {
            espCondMax = p_continuation[0].getValue(ptStockCur, p_state.getStochasticRealization());
            ptStockMax(0) += inflow;
        }
        // gain to add at current point
        double phiAdd = 0;
        // if withdrawal
        // level min of the stock
        double minStorage = p_grid->getExtremeValues()[0][0];
        double withdrawalMax = std::min(p_state.getPtStock()(0) + inflow - minStorage, m_withdrawalRate);
        if (libflow::isStrictlyLesser(0., withdrawalMax))
        {
            double gainWithdrawal =  withdrawalMax ;
            // level reached
            ptStockCur = ptStock + inflow - withdrawalMax;
            ptStockCur(0) = std::min(ptStockCur(0), maxStorage);
            double continuationWithdrawal = p_continuation[0].getValue(ptStockCur, p_state.getStochasticRealization());
            double espCondWithdrawal = gainWithdrawal + continuationWithdrawal ;
            if (espCondWithdrawal > espCondMax)
            {
                espCondMax = espCondWithdrawal;
                phiAdd = gainWithdrawal;
                ptStockMax(0) = ptStockCur(0);
            }
        }
        // for return
        p_state.setPtStock(ptStockMax);
        p_phiInOut(0) += phiAdd  ;
    }


    ///\brief store the simulator
    inline void setSimulator(const std::shared_ptr<ARModel> &p_simulator)
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
#endif /* OPTIMIZELAKE_H */
