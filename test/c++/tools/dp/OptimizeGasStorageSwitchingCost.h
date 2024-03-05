
#ifndef OPTIMIZEGASSTORAGESWITCHINGCOST_H
#define OPTIMIZEGASSTORAGESWITCHINGCOST_H
#include <Eigen/Dense>
#include <memory>
#include "libflow/core/grids/LinearInterpolator.h"
#include "libflow/core/grids/Interpolator.h"
#include "libflow/core/utils/StateWithStocks.h"
#include "libflow/core/grids/SpaceGrid.h"
#include "libflow/core/grids/OneDimRegularSpaceGrid.h"
#include "libflow/core/grids/OneDimSpaceGrid.h"
#include "libflow/core/grids/OneDimData.h"
#include "libflow/regression/BaseRegression.h"
#include "libflow/regression/ContinuationValue.h"
#include "libflow/regression/GridAndRegressedValue.h"
#include "libflow/dp/OptimizerDPBase.h"

/** \file OptimizeGasStorageSwitchingCost.h
 *  \brief  Simple example of a gas storage optimizer
 *          - injection rate, withdrawal rates are independent of the storage level
 *          - the size of the storage is constant, minimum gas level is 0
 *          .
 *          This case includes switching cost such that the regimes are used.
 *          The state is given by the uncertainties of the price, the stock level, and the regime ( injection, withdrawal, do nothing)
 *          Designed to work in parallel/multi threaded framework, it is an example of a general switching problem
 *  \author Xavier Warin
 */

/// \class OptimizeGasStorageSwitchingCost OptimizeGasStorageSwitchingCost.h
/// Defines a simple gas storage for optimization and simulation
/// No constraints on the storage at the end of optimization period (so the storage will be empty)
/// - when injecting the gain is  \f$ - C_{inj} ( S+ \kappa_{inj} )\f$
/// - when withdrawing the gain is  \f$  C_{with} ( S- \kappa_{with} )\f$
/// .
/// Each change of regime is penalized by a fixed cost
///  - regime 0 corresponds to do nothing
///  - regime 1 corresponds to injection regime
///  - regime 2 corresponds to withdrawal regime
/// The number of regime may vary
///  - it can be 1 : then it is only allowed to do nothing
///  - it can be 2 : do nothing and withdrawal are allowed
///  - it can be 3 : all regimes au allowed
template< class Simulator>
class OptimizeGasStorageSwitchingCost : public libflow::OptimizerDPBase
{
private :
    /// \brief Physical constraints
    //@{
    double m_injectionRate ; ///< injection  capacity (volume) per time step \f$ C_{inj} \f$
    double m_withdrawalRate ; ///< withdrawal rate (volume) per time step \f$ C_{with} \f$
    double m_injectionCost; ///< injection cost \f$ \kappa_{inj} \f$ per volume unit
    double m_withdrawalCost ; ///< withdrawal cost  \f$ \kappa_{with} \f$ per volume unit
    double m_switchCost ; ///< switching cost when changing of regime
    //@}
    /// \brief store the simulator
    std::shared_ptr<Simulator> m_simulator;
    std::shared_ptr<libflow::OneDimData<libflow::OneDimSpaceGrid, int> > m_regime; ///< defines the number of regimes

public :

    /// \brief Constructor
    /// \param  p_injectionRate     injection rate per time step
    /// \param  p_withdrawalRate    withdrawal rate between two time steps
    /// \param  p_injectionCost     injection cost
    /// \param  p_withdrawalCost    withdrawal cost
    /// \param  p_switchCost         switching cost
    OptimizeGasStorageSwitchingCost(const double   &p_injectionRate, const double &p_withdrawalRate,
                                    const double &p_injectionCost, const double &p_withdrawalCost,
                                    const double &p_switchCost):
        m_injectionRate(p_injectionRate), m_withdrawalRate(p_withdrawalRate), m_injectionCost(p_injectionCost), m_withdrawalCost(p_withdrawalCost),
        m_switchCost(p_switchCost)
    {
        Eigen::ArrayXd tval(2);
        tval(0) = 0. ;
        tval(1) = libflow::infty;
        std::shared_ptr<libflow::OneDimSpaceGrid> timeGrid = std::make_shared<libflow::OneDimSpaceGrid>(tval);
        std::shared_ptr<std::vector< int > > values =  std::make_shared<std::vector< int > >(2);
        (*values)[0] = 3;
        (*values)[1] = 3;
        m_regime = std::make_shared<libflow::OneDimData<libflow::OneDimSpaceGrid, int> >(timeGrid, values);
    }

    /// \brief Second constructor with a number of regime changing
    /// \param  p_injectionRate     injection rate per time step
    /// \param  p_withdrawalRate    withdrawal rate between two time steps
    /// \param  p_injectionCost     injection cost
    /// \param  p_withdrawalCost    withdrawal cost
    /// \param  p_switchCost         switching cost
    /// \param  p_regime            give the number of regimes depending on date
    OptimizeGasStorageSwitchingCost(const double   &p_injectionRate, const double &p_withdrawalRate,
                                    const double &p_injectionCost, const double &p_withdrawalCost,
                                    const double &p_switchCost,
                                    const std::shared_ptr<libflow::OneDimData<libflow::OneDimSpaceGrid, int> > &p_regime):
        m_injectionRate(p_injectionRate), m_withdrawalRate(p_withdrawalRate), m_injectionCost(p_injectionCost), m_withdrawalCost(p_withdrawalCost),
        m_switchCost(p_switchCost),
        m_regime(p_regime)
    {}

    /// \brief define the diffusion cone for parallelism
    /// \param  p_regionByProcessor         region (min max) treated by the processor for the different regimes treated
    /// \return returns in each dimension the min max values in the stock that can be reached from the grid p_gridByProcessor for each regime
    std::vector< std::array< double, 2> > getCone(const  std::vector< std::array< double, 2>  > &p_regionByProcessor) const
    {
        std::vector< std::array< double, 2> > extrGrid(1);
        extrGrid[0][0] = p_regionByProcessor[0][0] - m_withdrawalRate;
        extrGrid[0][1] = p_regionByProcessor[0][1] + m_injectionRate;
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
    ///
    std::pair< Eigen::ArrayXXd, Eigen::ArrayXXd> stepOptimize(const std::shared_ptr< libflow::SpaceGrid> &p_grid, const Eigen::ArrayXd   &p_stock,
            const std::vector<libflow::ContinuationValue> &p_condEsp,
            const std::vector < std::shared_ptr< Eigen::ArrayXXd > > &p_phiIn) const
    {
        // actualization
        double actuStep = m_simulator->getActuStep(); // for one step
        // number of regime  allowed at the beginning of the time step
        int nbReg = getNbRegime();
        // number of regimes reached to test
        int nbRegReached = getNbRegimeReached();
        int nbSimul = m_simulator->getNbSimul();
        std::pair< Eigen::ArrayXXd, Eigen::ArrayXXd> solutionAndControl;
        solutionAndControl.first.resize(nbSimul, nbReg);
        solutionAndControl.second.resize(nbSimul, nbReg);
        // Spot price
        Eigen::ArrayXd spotPrice =  m_simulator->fromParticlesToSpot(m_simulator->getParticles()).array();

        // size of the stock
        double maxStorage = p_grid->getExtremeValues()[0][1];
        //  injection
        double injectionMax = std::min(maxStorage - p_stock(0), m_injectionRate);
        // level min of the stock
        double minStorage = p_grid->getExtremeValues()[0][0];
        double withdrawalMax = std::min(p_stock(0) - minStorage, m_withdrawalRate);

        if (libflow::isStrictlyLesser(injectionMax, 0.) && libflow::isStrictlyLesser(withdrawalMax, 0.))
        {
            // not an admissible point
            solutionAndControl.first.setConstant(-libflow::infty);
            solutionAndControl.second.setConstant(0.);
            return solutionAndControl;
        }

        // Suppose that non injection and no withdrawal
        Eigen::ArrayXd condExpSameStock, cashSameStock;
        if (p_grid->isInside(p_stock))
        {
            // create interpolator at current stock point
            std::shared_ptr<libflow::Interpolator>  interpolatorCurrentStock = p_grid->createInterpolator(p_stock);
            // cash flow at current stock and previous step
            cashSameStock = interpolatorCurrentStock->applyVec(*p_phiIn[0]);
            // conditional expectation at current stock point
            condExpSameStock =  actuStep * p_condEsp[0].getAllSimulations(*interpolatorCurrentStock) ;
        }
        Eigen::ArrayXd  cashInjectionStock, condExpInjectionStock;
        if ((libflow::isStrictlyLesser(0., injectionMax)) && (nbRegReached >= 2))
        {
            Eigen::ArrayXd injectionStock = p_stock + injectionMax;
            // interpolator for stock level if injection
            std::shared_ptr<libflow::Interpolator>  interpolatorInjectionStock = p_grid->createInterpolator(injectionStock);
            // cash flow  at previous step at injection level
            cashInjectionStock = interpolatorInjectionStock->applyVec(*p_phiIn[1]);
            // conditional expectation at injection stock level
            condExpInjectionStock = actuStep * p_condEsp[0].getAllSimulations(*interpolatorInjectionStock);
        }
        // withdrawal
        Eigen::ArrayXd  cashWithdrawalStock, condExpWithdrawalStock;
        if ((libflow::isStrictlyLesser(0., withdrawalMax)) && (nbRegReached == 3))
        {
            Eigen::ArrayXd withdrawalStock = p_stock - withdrawalMax;
            // interpolator for stock level if withdrawal
            std::shared_ptr<libflow::Interpolator>   interpolatorWithdrawalStock = p_grid->createInterpolator(withdrawalStock);
            // cash flow  at previous step at injection level
            cashWithdrawalStock = interpolatorWithdrawalStock->applyVec(*p_phiIn[2]);
            // conditional expectation at withdrawal stock level
            condExpWithdrawalStock = actuStep * p_condEsp[0].getAllSimulations(*interpolatorWithdrawalStock);
        }

        Eigen::ArrayXXd  gainInjection(spotPrice.size(), 3), gainWithdrawal(spotPrice.size(), 3), gainSameStock(spotPrice.size(), 3);
        /// suppose that current regime is 0 (Do Nothing)
        gainInjection.col(0) =  - injectionMax * (spotPrice + m_injectionCost) - m_switchCost;
        gainWithdrawal.col(0) =  withdrawalMax * (spotPrice - m_withdrawalCost) - m_switchCost;
        gainSameStock.col(0).setConstant(0.) ;
        // Regime 1) : injection
        gainInjection.col(1) =  - injectionMax * (spotPrice + m_injectionCost) ;
        gainWithdrawal.col(1) =  withdrawalMax * (spotPrice - m_withdrawalCost) - m_switchCost;
        gainSameStock.col(1).setConstant(- m_switchCost);
        // Regime 2) : withdrawal
        gainInjection.col(2) =  - injectionMax * (spotPrice + m_injectionCost) - m_switchCost ;
        gainWithdrawal.col(2) =  withdrawalMax * (spotPrice - m_withdrawalCost);
        gainSameStock.col(2).setConstant(- m_switchCost);

        // now arbitrage in each regime
        for (int iReg = 0; iReg < nbReg; ++ iReg)
        {
            if ((cashInjectionStock.size() > 0) && (cashWithdrawalStock.size() > 0))
            {
                if (nbRegReached == 3)
                {
                    // do the arbitrage
                    for (int is = 0; is < spotPrice.size(); ++is)
                    {
                        solutionAndControl.first(is, iReg) = gainSameStock(is, iReg) + actuStep * cashSameStock(is);
                        solutionAndControl.second(is, iReg) =  0.;
                        double espCondMax = condExpSameStock(is);
                        double espCondInjection = gainInjection(is, iReg) + condExpInjectionStock(is);
                        if (espCondInjection > espCondMax)
                        {
                            solutionAndControl.first(is, iReg) =	gainInjection(is, iReg) + actuStep * cashInjectionStock(is);
                            solutionAndControl.second(is, iReg) = injectionMax;
                            espCondMax = espCondInjection;
                        }
                        double espCondWithdrawal = gainWithdrawal(is, iReg) + condExpWithdrawalStock(is);
                        if (espCondWithdrawal > espCondMax)
                        {
                            solutionAndControl.first(is, iReg) = gainWithdrawal(is, iReg) + actuStep * cashWithdrawalStock(is);
                            solutionAndControl.second(is, iReg) = -withdrawalMax;
                        }
                    }
                }
                else if (nbRegReached == 2)
                {
                    // do the arbitrage
                    for (int is = 0; is < spotPrice.size(); ++is)
                    {
                        solutionAndControl.first(is, iReg) = gainSameStock(is, iReg) + actuStep * cashSameStock(is);
                        solutionAndControl.second(is, iReg) =  0.;
                        double espCondMax = condExpSameStock(is);
                        double espCondInjection = gainInjection(is, iReg) + condExpInjectionStock(is);
                        if (espCondInjection > espCondMax)
                        {
                            solutionAndControl.first(is, iReg) =	gainInjection(is, iReg) + actuStep * cashInjectionStock(is);
                            solutionAndControl.second(is, iReg) = injectionMax;
                        }
                    }
                }
                else
                    for (int is = 0; is < spotPrice.size(); ++is)
                    {
                        solutionAndControl.first(is, iReg) = gainSameStock(is, iReg) + actuStep * cashSameStock(is);
                        solutionAndControl.second(is, iReg) =  0.;
                    }
            }
            else if (cashWithdrawalStock.size() > 0)
            {
                if (cashSameStock.size() > 0)
                {
                    if (nbRegReached == 3)
                    {
                        // do the arbitrage
                        for (int is = 0; is < spotPrice.size(); ++is)
                        {
                            solutionAndControl.first(is, iReg) = gainSameStock(is, iReg) + actuStep * cashSameStock(is);
                            solutionAndControl.second(is, iReg) =  0.;
                            double espCondMax = condExpSameStock(is) - m_switchCost;
                            double espCondWithdrawal = gainWithdrawal(is, iReg) + condExpWithdrawalStock(is);
                            if (espCondWithdrawal > espCondMax)
                            {
                                solutionAndControl.first(is, iReg) = gainWithdrawal(is, iReg) + actuStep * cashWithdrawalStock(is);
                                solutionAndControl.second(is, iReg) = -withdrawalMax;
                            }
                        }
                    }
                    else
                    {
                        for (int is = 0; is < spotPrice.size(); ++is)
                        {
                            solutionAndControl.first(is, iReg) = gainSameStock(is, iReg) + actuStep * cashSameStock(is);
                            solutionAndControl.second(is, iReg) =  0.;
                        }
                    }
                }
                else
                {
                    if (nbRegReached == 3)
                        for (int is = 0; is < spotPrice.size(); ++is)
                        {
                            solutionAndControl.first(is, iReg) = gainWithdrawal(is, iReg) + actuStep * cashWithdrawalStock(is);
                            solutionAndControl.second(is, iReg) = -withdrawalMax;
                        }
                    else
                    {
                        solutionAndControl.first.col(iReg).setConstant(-libflow::infty);
                        solutionAndControl.second.col(iReg).setConstant(0.);
                    }
                }
            }
            else if (cashInjectionStock.size() > 0)
            {
                if (cashSameStock.size() > 0)
                {
                    if (nbRegReached >= 2)
                    {
                        // do the arbitrage
                        for (int is = 0; is < spotPrice.size(); ++is)
                        {
                            solutionAndControl.first(is, iReg) = gainSameStock(is, iReg) + actuStep * cashSameStock(is);
                            solutionAndControl.second(is, iReg) =  0.;
                            double espCondMax = condExpSameStock(is);
                            double espCondInjection = gainInjection(is, iReg) + condExpInjectionStock(is);
                            if (espCondInjection > espCondMax)
                            {
                                solutionAndControl.first(is, iReg) =	gainInjection(is, iReg) + actuStep * cashInjectionStock(is);
                                solutionAndControl.second(is, iReg) = injectionMax;
                            }
                        }
                    }
                    else
                    {
                        for (int is = 0; is < spotPrice.size(); ++is)
                        {
                            solutionAndControl.first(is, iReg) = gainSameStock(is, iReg) + actuStep * cashSameStock(is);
                            solutionAndControl.second(is, iReg) =  0.;
                        }
                    }
                }
                else
                {
                    if (nbRegReached >= 2)
                        for (int is = 0; is < spotPrice.size(); ++is)
                        {
                            solutionAndControl.first(is, iReg) =	gainInjection(is, iReg) + actuStep * cashInjectionStock(is);
                            solutionAndControl.second(is, iReg) = injectionMax;
                        }
                    else
                    {
                        solutionAndControl.first.col(iReg).setConstant(-libflow::infty);
                        solutionAndControl.second.col(iReg).setConstant(0.);
                    }
                }
            }
            else
            {
                // only same level
                if (cashSameStock.size() > 0)
                    for (int is = 0; is < spotPrice.size(); ++is)
                    {
                        solutionAndControl.first(is, iReg) = gainSameStock(is, iReg) + actuStep * cashSameStock(is);
                        solutionAndControl.second(is, iReg) =  0.;
                    }
                else
                {
                    solutionAndControl.first.col(iReg).setConstant(-libflow::infty);
                    solutionAndControl.second.col(iReg).setConstant(0.);
                }
            }
        }
        return solutionAndControl;
    }

    /// \brief get number of regimes
    inline int getNbRegime() const
    {
        // number of regime potentially reached at the current date
        return m_regime->get(m_simulator->getCurrentStep());
    }

    /// \brief get number of regime reached at the end of the resolution
    inline int getNbRegimeReached() const
    {
        // number of regime potentially reached at the current date
        return m_regime->get(m_simulator->getCurrentStep() + m_simulator->getStep());
    }

    /// \brief number of controls : here it is equal to the number of regimes
    ///                             each regime admits de control
    inline int getNbControl() const
    {
        return getNbRegime();
    }


/// \brief defines a step in simulation
/// Notice that this implementation is not optimal. In fact no interpolation is necessary for this asset.
/// This implementation is for test and example purpose
/// \param p_grid          grid at arrival step after command
/// \param p_continuation  defines the continuation operator for each regime
/// \param p_state         defines the state value (modified)
/// \param p_phiInOut      defines the value function (modified): size number of functions to follow
    void stepSimulate(const std::shared_ptr< libflow::SpaceGrid>   &p_grid, const std::vector< libflow::GridAndRegressedValue  > &p_continuation,
                      libflow::StateWithStocks &p_state,  Eigen::Ref<Eigen::ArrayXd> p_phiInOut) const
    {
        // actualization
        double actu = m_simulator->getActu(); // at time zero
        double actuStep = m_simulator->getActuStep(); // for one step
        // number of regimes to test
        int nbRegReached = getNbRegimeReached();
        // optimal stock  attained
        Eigen::ArrayXd ptStockMax(p_state.getPtStock());
        // spot price
        double spotPrice = m_simulator->fromOneParticleToSpot(p_state.getStochasticRealization());
        // if do nothing
        double espCondMax = - libflow::infty;
        // switching cost depending on the state
        double switchingToDoNothing =  m_switchCost;
        double switchingToInjection =  m_switchCost;
        double switchingToWithdrawal = m_switchCost;
        if (p_state.getRegime() == 0)
            switchingToDoNothing = 0;
        else if (p_state.getRegime() == 1)
            switchingToInjection = 0. ;
        else
            switchingToWithdrawal = 0. ;

        // gain to add at current point
        double phiAdd = 0;
        int newRegime = -1;
        if (p_grid->isInside(ptStockMax))
        {
            double continuationDoNothing = actuStep * p_continuation[0].getValue(p_state.getPtStock(), p_state.getStochasticRealization());
            espCondMax = continuationDoNothing - switchingToDoNothing;
            phiAdd -= switchingToDoNothing;
            newRegime = 0;
        }
        // size of the stock
        double maxStorage = p_grid->getExtremeValues()[0][1];
        // if injection
        double injectionMax = std::min(maxStorage - p_state.getPtStock()(0), m_injectionRate);
        // store storage level
        double currentLevel = ptStockMax(0);
        if ((libflow::isStrictlyLesser(0., injectionMax)) && (nbRegReached >= 2))
        {
            double gainInjection = - injectionMax * (spotPrice + m_injectionCost);
            double continuationInjection =  actuStep * p_continuation[1].getValue(p_state.getPtStock() + injectionMax, p_state.getStochasticRealization());
            double espCondInjection = gainInjection + continuationInjection  - switchingToInjection;
            if (espCondInjection > espCondMax)
            {
                espCondMax = espCondInjection;
                phiAdd = gainInjection - switchingToInjection;
                ptStockMax(0) = currentLevel + injectionMax;
                newRegime = 1;
            }
        }

        // if withdrawal
        // level min of the stock
        double minStorage = p_grid->getExtremeValues()[0][0];
        double withdrawalMax = std::min(p_state.getPtStock()(0) - minStorage, m_withdrawalRate);
        if ((libflow::isStrictlyLesser(0., withdrawalMax)) && (nbRegReached == 3))
        {
            double gainWithdrawal =  withdrawalMax * (spotPrice - m_withdrawalCost);
            double continuationWithdrawal =  actuStep * p_continuation[2].getValue(p_state.getPtStock() - withdrawalMax, p_state.getStochasticRealization());
            double espCondWithdrawal = gainWithdrawal + continuationWithdrawal - switchingToWithdrawal;

            if (espCondWithdrawal > espCondMax)
            {
                phiAdd = gainWithdrawal - switchingToWithdrawal;
                ptStockMax(0) = currentLevel - withdrawalMax;
                newRegime = 2;
            }

        }
        // for return
        p_state.setPtStock(ptStockMax);
        p_state.setRegime(newRegime);
        p_phiInOut(0) += phiAdd * actu ;
    }


    /// \brief Defines a step in simulation using interpolation in controls
    /// \param p_grid          grid at arrival step after command
    /// \param p_control       defines the controls
    /// \param p_state         defines the state value (modified)
    /// \param p_phiInOut      defines the value function (modified): size number of functions to follow
    virtual void stepSimulateControl(const std::shared_ptr< libflow::SpaceGrid>   &p_grid, const std::vector< libflow::GridAndRegressedValue  > &p_control,
                                     libflow::StateWithStocks &p_state,
                                     Eigen::Ref<Eigen::ArrayXd>  p_phiInOut) const
    {
        // actualization
        double actu = m_simulator->getActu(); // at time zero
        double actuStep = m_simulator->getActuStep(); // for one step
        Eigen::ArrayXd ptStock = p_state.getPtStock();
        int iReg = p_state.getRegime();
        // spot price
        double spotPrice = m_simulator->fromOneParticleToSpot(p_state.getStochasticRealization());
        // optimal control
        double control = p_control[iReg].getValue(p_state.getPtStock(), p_state.getStochasticRealization());
        double maxStorage = p_grid->getExtremeValues()[0][1];
        double minStorage =  p_grid->getExtremeValues()[0][0];
        control = std::max(std::min(maxStorage - ptStock(0), control), minStorage - ptStock(0));
        if (control > 0)
        {
            // already injection
            p_phiInOut(0) -= control * (spotPrice + m_injectionCost) * actu;
            p_state.setRegime(1);
            if (iReg != 1)
            {
                p_phiInOut(0) -= m_switchCost;
                p_state.setRegime(1);
            }
        }
        else if (control < 0)
        {
            p_phiInOut(0) -= control * (spotPrice - m_withdrawalCost) * actu;
            if (iReg != 2)
            {
                p_phiInOut(0) -= m_switchCost;
                p_state.setRegime(2);
            }
        }
        else
        {
            // do nothing
            if (iReg != 0)
            {
                p_phiInOut(0) -= m_switchCost;
                p_state.setRegime(0);
            }
        }
        ptStock(0) += control ;
        p_state.setPtStock(ptStock);
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
    inline int getSimuFuncSize() const
    {
        return 1;
    }
}
;
#endif /* OPTIMIZEGASSTORAGESWITCHINGCOST_H */
