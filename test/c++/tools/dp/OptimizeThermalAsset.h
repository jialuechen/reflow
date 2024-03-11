#ifndef OPTIMIZETHERMALASSET_H
#define OPTIMIZETHERMALASSET_H
#include <memory>
#include <iostream>
#include <Eigen/Dense>
#include "libflow/core/utils/StateWithIntState.h"
#include "libflow/core/grids/RegularSpaceIntGrid.h"
#include "libflow/regression/BaseRegression.h"
#include "libflow/dp/OptimizerSwitchBase.h"

template< class Simulator>
class OptimizeThermalAsset : public libflow::OptimizerSwitchBase
{
private :
    double  m_switchCostFromOff ; ///< cost for switching on
    /// store the simulator
    std::shared_ptr<Simulator> m_simulator;

public:
    OptimizeThermalAsset(const double &p_switchCostFromOff): m_switchCostFromOff(p_switchCostFromOff) {}

    /// \brief defines the diffusion cone for parallelism
    /// \param  p_iRegime                   regime used
    /// \param  p_regionByProcessor         region (min max) treated by the processor for the different regimes treated
    /// \return returns in each dimension the min max values in the stock that can be reached from the grid p_gridByProcessor for each regime
    std::vector< std::array<int, 2 > >  getCone(const int &p_iReg, const std::vector< std::array<int, 2 > > &p_regionByProcessor) const
    {
        std::vector< std::array<int, 2 > > retGrid(1);
        retGrid[0][0] = 0; // should be 0 when switching
        retGrid[0][1] = p_regionByProcessor[0][1] + 1; // if stay in same regime
        return retGrid;
    }

    /// \brief defines the dimension to split for MPI parallelism
    ///        For each regime : for each dimension return true is the direction can be split
    std::vector< Eigen::Array< bool, Eigen::Dynamic, 1> > getDimensionToSplit() const
    {
        std::vector< Eigen::Array< bool, Eigen::Dynamic, 1> > toSplitReg(2);
        toSplitReg[0] = Eigen::Array< bool, Eigen::Dynamic, 1>::Constant(1, true);
        toSplitReg[1] = Eigen::Array< bool, Eigen::Dynamic, 1>::Constant(1, true);
        return  toSplitReg;
    }

    /// \brief defines a step in optimization
    /// \param p_grid      grid at arrival step after command (integer states) for each regime
    /// \param p_iReg      regime treated
    /// \param p_state     coordinates of the deterministic integer state
    /// \param p_condExp   Conditional expectation operator
    /// \param p_phiIn     for each regime  gives the solution calculated at the previous step ( next time step by Dynamic Programming resolution) : structure of the 2D array ( nb simulation ,nb stocks )
    /// \return   a pair  :
    ///              - for each regimes (column) gives the solution for each particle (row)
    ///              - for each control (column) gives the optimal control for each particle (rows)
    ///              .
    virtual Eigen::ArrayXd    stepOptimize(const   std::vector<std::shared_ptr< libflow::RegularSpaceIntGrid>>  &p_grid,
                                           const   int &p_iReg,
                                           const   Eigen::ArrayXi  &p_state,
                                           const   std::shared_ptr< libflow::BaseRegression>  &p_condExp,
                                           const   std::vector < std::shared_ptr< Eigen::ArrayXXd > > &p_phiIn) const
    {
        int nbSimul = m_simulator->getNbSimul();
        // value to return
        Eigen::ArrayXd value(nbSimul);
        // spot
        Eigen::ArrayXXd spots = m_simulator->getSpot();
        // state after decision
        Eigen::ArrayXi stateAfter(p_state);
        // test the regime
        switch (p_iReg)
        {
        case 0 :
        {
            // first case we are on
            int nbMinOn = p_grid[0]->getSizeInDim(0);
            if (p_state(0) < (nbMinOn - 1))
            {
                // no choice : we have to go on
                stateAfter(0) = p_state(0) + 1;
                value = (spots.row(1) - spots.row(0)).transpose()  + p_phiIn[0]->col(p_grid[0]->globCoordPerDimToLocal(stateAfter));
            }
            else
            {
                // we can arbitrage
                // espected gain if on : state does not change
                Eigen::ArrayXd gainAfterOn = p_phiIn[0]->col(p_grid[0]->globCoordPerDimToLocal(stateAfter));
                Eigen::ArrayXd expGainIfStayOn = (spots.row(1) - spots.row(0)).transpose() + p_condExp->getAllSimulations(gainAfterOn);
                // if goes to off the state goes to 0
                stateAfter(0) = 0 ;
                Eigen::ArrayXd gainAfterOff = p_phiIn[1]->col(p_grid[1]->globCoordPerDimToLocal(stateAfter));
                Eigen::ArrayXd expGainIfGoesOff = p_condExp->getAllSimulations(gainAfterOff);
                for (int is = 0 ; is < nbSimul; ++is)
                {
                    if (expGainIfStayOn(is) < expGainIfGoesOff(is))
                    {
                        value(is) = gainAfterOff(is);
                    }
                    else
                    {
                        value(is) = (spots(1, is) - spots(0, is)) + gainAfterOn(is);
                    }
                }
            }
            break;
        }
        case 1 :
        {
            // first we are off
            int nbMinOff = p_grid[1]->getSizeInDim(0);
            if (p_state(0) < (nbMinOff - 1))
            {
                // no choice : go on
                stateAfter(0) = p_state(0) + 1;
                value = p_phiIn[1]->col(p_grid[1]->globCoordPerDimToLocal(stateAfter));
            }
            else
            {
                // can arbitrage
                // expected gain if off : state does not change
                Eigen::ArrayXd gainAfterOff = p_phiIn[1]->col(p_grid[1]->globCoordPerDimToLocal(stateAfter));
                Eigen::ArrayXd expGainIfStayOff =  p_condExp->getAllSimulations(gainAfterOff);
                //  expected gain if on after
                stateAfter(0) = 0 ;
                Eigen::ArrayXd gainAfterOn = p_phiIn[0]->col(p_grid[0]->globCoordPerDimToLocal(stateAfter));
                // if goes on : add cas generated  and switching cost
                Eigen::ArrayXd expGainIfGoesOn = (spots.row(1) - spots.row(0)).transpose() - m_switchCostFromOff + p_condExp->getAllSimulations(gainAfterOn);
                for (int is = 0 ; is < nbSimul; ++is)
                {
                    if (expGainIfGoesOn(is) < expGainIfStayOff(is))
                    {
                        value(is) = gainAfterOff(is);
                    }
                    else
                    {
                        value(is) = (spots(1, is) - spots(0, is)) - m_switchCostFromOff   + gainAfterOn(is);
                    }
                }
            }
            break;
        }
        default:
        {
            std::cout << "Regime not allowed " << std::endl ;
            abort();
        }
        }
        return value;
    }

    /// \brief defines a step in simulation
    /// Notice that this implementation is not optimal but is convenient if the control is discrete.
    /// By avoiding interpolation in control we avoid non admissible control
    /// Control are recalculated during simulation.
    /// \param p_grid          grid at arrival step after command
    /// \param p_condExp       Conditional expectation operator reconstructing conditionnal expectation from basis functions for each state
    /// \param p_basisFunc     Basis functions par each point of the grid state for each regime
    /// \param p_state         defines the state value (modified)
    /// \param p_phiInOut      defines the value functions (modified) : size number of functions  to follow
    virtual void stepSimulate(const std::vector<std::shared_ptr< libflow::RegularSpaceIntGrid> >  &p_grid,
                              const std::shared_ptr< libflow::BaseRegression>  &p_condExp,
                              const std::vector< Eigen::ArrayXXd >   &p_basisFunc,
                              libflow::StateWithIntState &p_state,
                              Eigen::Ref<Eigen::ArrayXd> p_phiInOut) const
    {
        // get regime
        int iReg = p_state.getRegime();
        // deterministic state
        Eigen::ArrayXi stateCur = p_state.getPtState();
        Eigen::ArrayXi stateAfter(stateCur);
        // store stochastic realization
        Eigen::ArrayXd stochRealization = p_state.getStochasticRealization();
        // spot
        Eigen::ArrayXd spot = m_simulator->fromOUToSpot(stochRealization);
        // test regime
        switch (iReg)
        {
        case 0:
        {
            // first case we are on
            int nbMinOn = p_grid[0]->getSizeInDim(0);
            if (stateCur(0) < (nbMinOn - 1))
            {
                // no choice : we have to go on
                stateAfter(0) = stateCur(0) + 1;
                p_phiInOut(0) += spot(1) - spot(0);
            }
            else
            {
                // we can arbitrage
                // espected gain if on : state does not change
                double expGainIfStayOn = (spot(1) - spot(0)) + p_condExp->getValue(stochRealization, p_basisFunc[0].col(p_grid[0]->globCoordPerDimToLocal(stateAfter)));
                stateAfter(0) = 0 ;
                double expGainIfGoesOff = p_condExp->getValue(stochRealization, p_basisFunc[1].col(p_grid[1]->globCoordPerDimToLocal(stateAfter)));
                if (expGainIfStayOn > expGainIfGoesOff)
                {
                    p_phiInOut(0) += (spot(1) - spot(0)) ;
                    stateAfter(0) =  nbMinOn - 1;
                }
                else
                {
                    stateAfter(0) = 0; // goes to 0 because of switch
                    p_state.setRegime(1); // switch state
                }
            }
            break;
        }
        case 1:
        {
            int nbMinOff = p_grid[1]->getSizeInDim(0);
            if (stateCur(0) < (nbMinOff - 1))
            {
                // no choice : we have to go on
                stateAfter(0) = stateCur(0) + 1;
            }
            else
            {
                //  arbitrage
                // stay off : tstae does not chaneg
                double  expGainIfStayOff =  p_condExp->getValue(stochRealization, p_basisFunc[1].col(p_grid[1]->globCoordPerDimToLocal(stateAfter)));
                // switch on : state goes to zero
                stateAfter(0) = 0;
                double expGainIfGoesOn = (spot(1) - spot(0)) - m_switchCostFromOff + p_condExp->getValue(stochRealization, p_basisFunc[0].col(p_grid[0]->globCoordPerDimToLocal(stateAfter)));
                if (expGainIfGoesOn > expGainIfStayOff)
                {
                    stateAfter(0) = 0;
                    p_phiInOut(0) += (spot(1) - spot(0)) - m_switchCostFromOff;
                    p_state.setRegime(0); // switch state to on
                }
                else
                {
                    stateAfter(0) = nbMinOff - 1;
                }
            }
            break;
        }
        default:
        {
            std::cout << "Impossible" << std::endl;
            abort();
        }
        }
        p_state.setPtState(stateAfter);
    }


    /// \brief Get the number of regimes allowed for the asset to be reached  at the current time step
    ///    If \f$ t \f$ is the current time, and $\f$ dt \f$  the resolution step,  this is the number of regime allowed on \f$[ t- dt, t[\f$
    virtual   int getNbRegime() const
    {
        return 2;
    }

    /// \brief get the simulator back
    std::shared_ptr< libflow::SimulatorDPBase > getSimulator() const
    {
        return m_simulator ;
    }
    ///\brief store the simulator
    inline void setSimulator(const std::shared_ptr<Simulator> &p_simulator)
    {
        m_simulator = p_simulator ;
    }

    /// \brief get size of the  function to follow in simulation
    int getSimuFuncSize() const
    {
        return 1;
    }

    /// \brief get the simulator back but the derived one
    inline std::shared_ptr< Simulator> getSimulatorDerived() const
    {
        return m_simulator ;
    }

    /// \brief Helper to revert simulator directly
    void resetSimulatorDirection(const bool &p_bForward)
    {
        m_simulator->resetDirection(p_bForward);
    }

};
#endif
