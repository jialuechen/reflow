
#ifndef OPTIMIZEGASSTORAGE_H
#define OPTIMIZEGASSTORAGE_H
#include <memory>
#include <Eigen/Dense>
#include "reflow/core/utils/StateWithStocks.h"
#include "reflow/core/grids/SpaceGrid.h"
#include "reflow/regression/BaseRegression.h"
#include "reflow/core/grids/Interpolator.h"
#include "reflow/regression/ContinuationValue.h"
#include "reflow/regression/GridAndRegressedValue.h"
#include "reflow/dp/OptimizerDPBase.h"

/** \file OptimizeGasStorage.h
 *  \brief  Simple example of a gas storage optimizer
 *          - injection rate, withdrawal rates are independent of the storage level
 *          - the size of the storage is constant, minimum gas level is 0
 *          .
 *          Designed to work in parallel/multi threaded framework
 *  \author Xavier Warin
 */

/// \class OptimizeGasStorage OptimizeGasStorage.h
/// Defines a simple gas storage for optimization and simulation
/// No constraints on the storage at the end of optimization period (so the storage will be empty)
/// - when injecting the gain is  \f$ - C_{inj} ( S+ \kappa_{inj} )\f$
/// - when withdrawing the gain is  \f$  C_{with} ( S- \kappa_{with} )\f$
/// .
template< class Simulator>
class OptimizeGasStorage : public reflow::OptimizerDPBase
{
private :
    /// \brief Physical constraints
    //@{
    double m_injectionRate ; ///< injection  capacity (volume) per time step \f$ C_{inj} \f$
    double m_withdrawalRate ; ///< withdrawal rate (volume) per time step \f$ C_{with} \f$
    double m_injectionCost; ///< injection cost \f$ \kappa_{inj} \f$ per volume unit
    double m_withdrawalCost ; /// withdrawal cost  \f$ \kappa_{with} \f$ per volume unit
    //@}
    /// store the simulator
    std::shared_ptr<Simulator> m_simulator;

public :

    /// \brief Constructor
    /// \param  p_injectionRate     injection rate per time step
    /// \param  p_withdrawalRate    withdrawal rate between two time steps
    /// \param  p_injectionCost     injection cost
    /// \param  p_withdrawalCost    withdrawal cost
    OptimizeGasStorage(const double   &p_injectionRate, const double &p_withdrawalRate,
                       const double &p_injectionCost, const double &p_withdrawalCost):
        m_injectionRate(p_injectionRate), m_withdrawalRate(p_withdrawalRate), m_injectionCost(p_injectionCost), m_withdrawalCost(p_withdrawalCost) {}

    /// \brief define the diffusion cone for parallelism (not needed if not parallelism required )
    /// \param  p_regionByProcessor         region (min max) treated by the processor for the different regimes treated
    /// \return returns in each dimension the min max values in the stock that can be reached from the grid p_gridByProcessor for each regime
    std::vector< std::array< double, 2> > getCone(const  std::vector<  std::array< double, 2>  > &p_regionByProcessor) const
    {
        std::vector< std::array< double, 2> > extrGrid(1);
        extrGrid[0][0] = p_regionByProcessor[0][0] - m_withdrawalRate;
        extrGrid[0][1] = p_regionByProcessor[0][1] + m_injectionRate;
        return extrGrid;
    }

    /// \brief defines the dimension to split for MPI parallelism (not neeede if no parallelism )
    ///        For each dimension return true is the direction can be split
    Eigen::Array< bool, Eigen::Dynamic, 1> getDimensionToSplit() const
    {
        Eigen::Array< bool, Eigen::Dynamic, 1> bDim = Eigen::Array< bool, Eigen::Dynamic, 1>::Constant(1, true);
        return  bDim ;
    }

    /// \brief defines a step in optimization
    /// \param p_grid      grid at arrival step after command
    /// \param p_stock     coordinate of the stock point to treat at current time step
    /// \param p_condEsp   continuation values for each regime (permitting to interpolate in stocks the regresses values)
    /// \param p_phiIn     for each regime  gives the solution calculated at the previous step ( next time step by Dynamic Programming resolution)
    /// \return   a pair  :
    ///              - for each regimes (column) gives the solution for each particle (row)
    ///              - for each control (column) gives the optimal control for each particle (rows)
    ///              .
    std::pair< Eigen::ArrayXXd, Eigen::ArrayXXd> stepOptimize(const   std::shared_ptr< reflow::SpaceGrid> &p_grid, const Eigen::ArrayXd   &p_stock,
            const std::vector<reflow::ContinuationValue> &p_condEsp,
            const std::vector < std::shared_ptr< Eigen::ArrayXXd > > &p_phiIn) const
    {
        int nbSimul = m_simulator->getNbSimul();
        // actualization
        double actuStep = m_simulator->getActuStep(); // for one step
        std::pair< Eigen::ArrayXXd, Eigen::ArrayXXd> solutionAndControl;
        solutionAndControl.first.resize(nbSimul, 1);
        solutionAndControl.second.resize(nbSimul, 1);
        // Spot price : here given by a composition of
        //  -the method getParticles from the simulator : its gives the regression factor of the models
        //  -the method fromParticlesToSpot  reconstructing the spot
        Eigen::ArrayXd spotPrice =  m_simulator->fromParticlesToSpot(m_simulator->getParticles()).array();
        // level if injection
        // size of the stock
        double maxStorage = p_grid->getExtremeValues()[0][1];
        double injectionMax = std::min(maxStorage - p_stock(0), m_injectionRate);
        Eigen::ArrayXd injectionStock = p_stock + injectionMax;
        // level if withdrawal
        // level min of the stock
        double minStorage = p_grid->getExtremeValues()[0][0];
        double withdrawalMax = std::min(p_stock(0) - minStorage, m_withdrawalRate);
        Eigen::ArrayXd withdrawalStock = p_stock - withdrawalMax;
        if (reflow::isStrictlyLesser(injectionMax, 0.) && reflow::isStrictlyLesser(withdrawalMax, 0.))
        {
            // not an admissible point
            solutionAndControl.first.setConstant(-reflow::infty);
            solutionAndControl.second.setConstant(0.);
            return solutionAndControl;
        }
        // Suppose that non injection and no withdrawal
        ///////////////////////////////////////////////
        // create interpolator at current stock point
        Eigen::ArrayXd condExpSameStock, cashSameStock;
        if (p_grid->isInside(p_stock))
        {
            std::shared_ptr<reflow::Interpolator>  interpolatorCurrentStock = p_grid->createInterpolator(p_stock);
            // cash flow at current stock and previous step
            cashSameStock = interpolatorCurrentStock->applyVec(*p_phiIn[0]);
            // conditional expectation at current stock point
            condExpSameStock = actuStep * p_condEsp[0].getAllSimulations(*interpolatorCurrentStock) ;
        }
        //  injection
        ///////////////
        Eigen::ArrayXd  gainInjection, cashInjectionStock, condExpInjectionStock;
        if (reflow::isStrictlyLesser(0., injectionMax))
        {
            // interpolator for stock level if injection
            std::shared_ptr<reflow::Interpolator>  interpolatorInjectionStock = p_grid->createInterpolator(injectionStock);
            // cash flow  at previous step at injection level for all trajectories
            cashInjectionStock = interpolatorInjectionStock->applyVec(*p_phiIn[0]);
            // conditional expectation at injection stock level for all trajectories
            condExpInjectionStock = actuStep * p_condEsp[0].getAllSimulations(*interpolatorInjectionStock);
            // instantaneous gain if injection
            gainInjection =  - injectionMax * (spotPrice + m_injectionCost);
        }
        // withdrawal
        ///////////////
        Eigen::ArrayXd  gainWithdrawal, cashWithdrawalStock, condExpWithdrawalStock;
        if (reflow::isStrictlyLesser(0., withdrawalMax))
        {
            // interpolator for stock level if withdrawal
            std::shared_ptr<reflow::Interpolator> interpolatorWithdrawalStock = p_grid->createInterpolator(withdrawalStock);
            // cash flow  at previous step at withdrawal level for all trajectories
            cashWithdrawalStock = interpolatorWithdrawalStock->applyVec(*p_phiIn[0]);
            // conditional expectation at withdrawal stock level for all trajectories
            condExpWithdrawalStock = actuStep * p_condEsp[0].getAllSimulations(*interpolatorWithdrawalStock);
            // instantaneous gain if withdrawal
            gainWithdrawal =  withdrawalMax * (spotPrice - m_withdrawalCost);
        }
        // do the arbitrage
        //////////////////
        if ((gainWithdrawal.size() > 0) && (gainInjection.size() > 0))
        {
            // all point admissible
            for (int is = 0; is < spotPrice.size(); ++is)
            {
                solutionAndControl.first(is, 0) = actuStep * cashSameStock(is);
                solutionAndControl.second(is, 0) = 0.;
                double espCondMax = condExpSameStock(is);
                double espCondInjection = gainInjection(is) + condExpInjectionStock(is);
                if (espCondInjection > espCondMax)
                {
                    solutionAndControl.first(is, 0) =	gainInjection(is) + actuStep * cashInjectionStock(is);
                    solutionAndControl.second(is, 0) = injectionMax	;
                    espCondMax = espCondInjection;
                }
                double espCondWithdrawal = gainWithdrawal(is) + condExpWithdrawalStock(is);
                if (espCondWithdrawal > espCondMax)
                {
                    solutionAndControl.first(is, 0) = gainWithdrawal(is) + actuStep * cashWithdrawalStock(is);
                    solutionAndControl.second(is, 0) = -withdrawalMax;
                }
            }
        }
        else if (gainWithdrawal.size() > 0)
        {
            if (condExpSameStock.size() > 0)
            {
                for (int is = 0; is < spotPrice.size(); ++is)
                {
                    solutionAndControl.first(is, 0) = actuStep * cashSameStock(is);
                    solutionAndControl.second(is, 0) = 0.;
                    double espCondMax = condExpSameStock(is);
                    double espCondWithdrawal = gainWithdrawal(is) + condExpWithdrawalStock(is);
                    if (espCondWithdrawal > espCondMax)
                    {
                        solutionAndControl.first(is, 0) = gainWithdrawal(is) + actuStep * cashWithdrawalStock(is);
                        solutionAndControl.second(is, 0) = -withdrawalMax;
                    }
                }
            }
            else
            {
                for (int is = 0; is < spotPrice.size(); ++is)
                {
                    solutionAndControl.first(is, 0)  = gainWithdrawal(is) + actuStep * cashWithdrawalStock(is);
                    solutionAndControl.second(is, 0) = -withdrawalMax;
                }
            }
        }
        else if (gainInjection.size() > 0)
        {
            if (condExpSameStock.size() > 0)
            {
                for (int is = 0; is < spotPrice.size(); ++is)
                {
                    solutionAndControl.first(is, 0) = actuStep * cashSameStock(is);
                    solutionAndControl.second(is, 0) = 0.;
                    double espCondMax = condExpSameStock(is);
                    double espCondInjection = gainInjection(is) + condExpInjectionStock(is);
                    if (espCondInjection > espCondMax)
                    {
                        solutionAndControl.first(is, 0) =	gainInjection(is) + actuStep * cashInjectionStock(is);
                        solutionAndControl.second(is, 0) = injectionMax	;
                    }
                }
            }
            else
            {
                for (int is = 0; is < spotPrice.size(); ++is)
                {
                    solutionAndControl.first(is, 0) =	gainInjection(is) + actuStep * cashInjectionStock(is);
                    solutionAndControl.second(is, 0) = injectionMax	;
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
/// \param p_grid          grid at arrival step after command
/// \param p_continuation  defines the continuation operator for each regime
/// \param p_state         defines the state value (modified)
/// \param p_phiInOut      defines the value functions (modified): size number of functions to follow
    void stepSimulate(const std::shared_ptr< reflow::SpaceGrid>   &p_grid, const std::vector< reflow::GridAndRegressedValue  > &p_continuation,
                      reflow::StateWithStocks &p_state, Eigen::Ref<Eigen::ArrayXd> p_phiInOut) const
    {
        // optimal stock  attained
        Eigen::ArrayXd ptStockCur = p_state.getPtStock();
        Eigen::ArrayXd ptStockMax(ptStockCur);
        // actualization
        double actu = m_simulator->getActu(); // at time zero
        double actuStep = m_simulator->getActuStep(); // for one step
        // spot price
        double spotPrice = m_simulator->fromOneParticleToSpot(p_state.getStochasticRealization());
        // if do nothing
        double espCondMax = - reflow::infty;
        if (p_grid->isInside(ptStockCur))
        {
            double continuationDoNothing = actuStep * p_continuation[0].getValue(p_state.getPtStock(), p_state.getStochasticRealization());
            espCondMax = continuationDoNothing;
        }

        // gain to add at current point
        double phiAdd = 0;
        // size of the stock
        double maxStorage = p_grid->getExtremeValues()[0][1];
        // if injection
        double injectionMax = std::min(maxStorage - p_state.getPtStock()(0), m_injectionRate);
        // store storage level
        double currentLevel = ptStockMax(0);
        if (reflow::isStrictlyLesser(0., injectionMax))
        {
            double continuationInjection =  actuStep * p_continuation[0].getValue(p_state.getPtStock() + injectionMax, p_state.getStochasticRealization());
            double gainInjection = - injectionMax * (spotPrice + m_injectionCost);
            double espCondInjection = gainInjection + continuationInjection ;
            if (espCondInjection > espCondMax)
            {
                espCondMax = espCondInjection;
                phiAdd = gainInjection;
                ptStockMax(0) = currentLevel + injectionMax;
            }
        }

        // if withdrawal
        // level min of the stock
        double minStorage = p_grid->getExtremeValues()[0][0];
        double withdrawalMax = std::min(p_state.getPtStock()(0) - minStorage, m_withdrawalRate);
        if (reflow::isStrictlyLesser(0., withdrawalMax))
        {
            double gainWithdrawal =  withdrawalMax * (spotPrice - m_withdrawalCost);
            double continuationWithdrawal =  actuStep * p_continuation[0].getValue(p_state.getPtStock() - withdrawalMax, p_state.getStochasticRealization());
            double espCondWithdrawal = gainWithdrawal + continuationWithdrawal ;
            if (espCondWithdrawal > espCondMax)
            {
                espCondMax = espCondWithdrawal;
                phiAdd = gainWithdrawal;
                ptStockMax(0) = currentLevel - withdrawalMax;
            }
        }
        // for return
        p_state.setPtStock(ptStockMax);
        p_phiInOut(0) += phiAdd * actu ;
    }

    /// \brief Defines a step in simulation using interpolation in controls
    /// \param p_grid          grid at arrival step after command
    /// \param p_control       defines the controls
    /// \param p_state         defines the state value (modified)
    /// \param p_phiInOut      defines the value function (modified): size number of functions to follow
    virtual void stepSimulateControl(const std::shared_ptr< reflow::SpaceGrid>   &p_grid, const std::vector< reflow::GridAndRegressedValue  > &p_control,
                                     reflow::StateWithStocks &p_state,
                                     Eigen::Ref<Eigen::ArrayXd>  p_phiInOut) const
    {
        Eigen::ArrayXd ptStock = p_state.getPtStock();
        // actualization
        double actu = m_simulator->getActu(); // at time zero
        // spot price
        double spotPrice = m_simulator->fromOneParticleToSpot(p_state.getStochasticRealization());
        // optimal control
        double control = p_control[0].getValue(p_state.getPtStock(), p_state.getStochasticRealization());
        double maxStorage = p_grid->getExtremeValues()[0][1];
        double minStorage =  p_grid->getExtremeValues()[0][0];
        control = std::max(std::min(maxStorage - ptStock(0), control), minStorage - ptStock(0));
        if (control > 0)
            p_phiInOut(0) -= control * (spotPrice + m_injectionCost) * actu;
        else if (control < 0)
            p_phiInOut(0) -= control * (spotPrice - m_withdrawalCost) * actu;
        ptStock(0) += control ;
        ptStock(0) = std::min(std::max(minStorage, ptStock(0)), maxStorage); // avoid rounding error
        p_state.setPtStock(ptStock);
    }


    ///\brief store the simulator
    inline void setSimulator(const std::shared_ptr<Simulator> &p_simulator)
    {
        m_simulator = p_simulator ;
    }

    /// \brief get the simulator back
    inline std::shared_ptr< reflow::SimulatorDPBase > getSimulator() const
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
#endif /* OPTIMIZEGASSTORAGE_H */
