// Copyright (C) 2016 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#ifndef OPTIMIZEPORTFOLIODP_H
#define OPTIMIZEPORTFOLIODP_H
#include <Eigen/Dense>
#include "libflow/dp/OptimizerNoRegressionDPBase.h"
#include "libflow/core/utils/constant.h"
#include "libflow/core/utils/comparisonUtils.h"
#include "libflow/regression/LocalConstRegression.h"
#include "test/c++/tools/simulators/MMMSimulator.h"

/** \file OptimizePortfolioDP.h
 * \brief Optimize a simple portfolio by dynamic programming using constant regressor
 *        The command is taken constant per mesh
 * \author Xavier Warin
 */

/// \class OptimizePortfolioDP OptimizePortfolioDP.h
/// Defines an optimization for portfolio with Dynamic programming
/// \f$ dA_t = \theta A_t \frac{dS_t}{S_t} \f$
/// Optimize in expectation (maximization).

class OptimizePortfolioDP : public libflow::OptimizerNoRegressionDPBase
{
private :

    int m_nbStep ; // number of discretization steps
    std::shared_ptr<MMMSimulator> m_simulator; /// MMM simulator
    std::shared_ptr< Eigen::ArrayXXd >  m_currentSim ; /// current asset simulations
    std::shared_ptr< Eigen::ArrayXXd > m_nextSim ; /// asset simulations at next time step

public :

    /// \brief ructror
    /// \param p_nbStep  number of discretization step for portfolio
    OptimizePortfolioDP(const int &p_nbStep): m_nbStep(p_nbStep) {}

    /// \brief Before first time step resolution : store particles
    void initializeSimulation()
    {
        m_currentSim =  std::make_shared<Eigen::ArrayXXd>(Eigen::ArrayXXd(m_simulator->getParticles().array()));
    }

    /// \brief  Store current simulations, one step backward
    void oneStepBackward()
    {
        // swap pointer
        m_nextSim = m_currentSim;
        // first get particles
        m_currentSim =  std::make_shared<Eigen::ArrayXXd>(Eigen::ArrayXXd(m_simulator->stepBackwardAndGetParticles().array()));
    }

    /// \brief  get Current simulation
    std::shared_ptr< Eigen::ArrayXXd > getCurrentSim() const
    {
        return m_currentSim;
    }

    /// \brief define the diffusion cone for parallelism
    /// \param  p_regionByProcessor         region (min max) treated by the processor for the different regimes treated
    /// \return returns in each dimension the min max values in the stock that can be reached from the grid p_gridByProcessor for each regime
    std::vector< std::array< double, 2> > getCone(const  std::vector<  std::array< double, 2>  > &p_regionByProcessor) const
    {

        std::vector< std::array< double, 2> > extrGrid(1);
        // max variations estimated
        Eigen::ArrayXXd increment = (*m_nextSim - *m_currentSim) / (*m_currentSim);
        double maxDefPlus =  increment.maxCoeff();
        double maxDefMinus = increment.minCoeff();
        extrGrid[0][0] = p_regionByProcessor[0][0] * (1 + maxDefMinus);
        extrGrid[0][1] = p_regionByProcessor[0][1] * (1 + maxDefPlus);
        return extrGrid;
    }

    /// \brief defines a step in optimization
    /// \param p_stock          coordinates associated to the portfolio discretization
    /// \param p_valNext        Optimize values at next time step for each regime
    /// \param p_regressorCur   Regressor at the current date
    /// \return   a pair  :
    ///              - for each regimes (column) gives the solution for each particle (row)
    ///              - for each control (column) gives the optimal control for each particle (rows)
    ///              .
    virtual std::pair< Eigen::ArrayXXd, Eigen::ArrayXXd>   stepOptimize(const Eigen::ArrayXd   &p_stock,
            const std::vector< libflow::GridAndRegressedValue>  &p_valNext,
            std::shared_ptr< libflow::BaseRegression  >     p_regressorCur) const
    {
        // convert
        std::shared_ptr< libflow::LocalConstRegression> constRegressor  = std::static_pointer_cast<libflow::LocalConstRegression>(p_regressorCur);
        int nbSimul = m_simulator->getNbSimul();

        std::pair< Eigen::ArrayXXd, Eigen::ArrayXXd> solutionAndControl;
        // to store final solution (here one regime) and the optimal control
        solutionAndControl.first.resize(nbSimul, 1);
        solutionAndControl.second.resize(nbSimul, 1);
        // grid used for interpolation at next date
        std::shared_ptr< libflow::SpaceGrid >  grid = p_valNext[0].getGrid();
        // level min, max of the portfolio considered
        double minA = grid->getExtremeValues()[0][0];
        double maxA = grid->getExtremeValues()[0][1];
        // is current time 0 ?
        bool bZeroDate = libflow::isLesserOrEqual(m_simulator-> getCurrentStep(), 0.);
        // store number of cells
        int nbCell = constRegressor->getNumberOfFunction();
        // each simulation associated to cell
        const Eigen::ArrayXi &simToCell = constRegressor->getSimToCell();
        // store the number of commands
        Eigen::ArrayXd optCommand = Eigen::ArrayXd::Constant(nbCell, -libflow::infty);
        // cash expectation
        Eigen::ArrayXd espCash(nbCell);
        Eigen::ArrayXd espCashMax = Eigen::ArrayXd::Constant(nbCell, -libflow::infty);
        Eigen::ArrayXi nbsimPerCell(nbCell);
        // step
        double step = 1. / m_nbStep;
        Eigen::ArrayXd vNext(1); // for next asset value
        Eigen::ArrayXd ANext(1) ; // for next portfolio value
        // test control
        for (int istep = 0; istep <= m_nbStep; ++istep)
        {
            //command
            double command = istep * step;
            espCash.setConstant(0.);
            nbsimPerCell.setConstant(0);
            for (int is = 0 ; is < nbSimul; ++is)
            {
                int iCell = simToCell(is);
                // portfolio value reached
                ANext(0) = p_stock(0)  + p_stock(0) * command * ((*m_nextSim)(0, is) - (*m_currentSim)(0, is)) / (*m_currentSim)(0, is);
                if ((ANext(0) >=  minA) && (ANext(0) <= maxA))
                {
                    nbsimPerCell(iCell) += 1;
                    vNext(0) = (*m_nextSim)(0, is);
                    espCash(iCell) += p_valNext[0].getValue(ANext, vNext);
                }
            }
            if (bZeroDate)
            {
                espCash(0) /= nbsimPerCell(0);
                if (espCash(0) > espCashMax(0))
                {
                    espCashMax(0) = espCash(0);
                    optCommand(0) = command;
                }
            }
            else
            {
                espCash /= nbsimPerCell.cast<double>();
                for (int iCell = 0; iCell  < nbCell; ++iCell)
                {
                    if (espCash(iCell) > espCashMax(iCell))
                    {
                        espCashMax(iCell) = espCash(iCell);
                        optCommand(iCell) = command;
                    }
                }
            }
        }
        for (int is = 0 ; is < nbSimul; ++is)
        {
            solutionAndControl.first(is, 0) = espCashMax(simToCell(is));
            solutionAndControl.second(is, 0) = optCommand(simToCell(is)) ;
        }
        return solutionAndControl;
    }

    /// \brief Defines a step in simulation using interpolation in controls
    /// \param p_control       defines the controls
    /// \param p_state         defines the state value (modified)
    /// \param p_phiInOut      defines the value function (modified): size number of functions to follow
    virtual void stepSimulateControl(const std::shared_ptr< libflow::SpaceGrid> &, const std::vector< libflow::GridAndRegressedValue  > &p_control,
                                     libflow::StateWithStocks &p_state,
                                     Eigen::Ref<Eigen::ArrayXd>  p_phiInOut) const
    {
        Eigen::ArrayXd AValue = p_state.getPtStock();
        // get back asset  value et next asset valeur
        Eigen::ArrayXd assetValue = p_state.getStochasticRealization();
        // get back asset value
        double assetCurrent = assetValue(0);
        // get back next asset value
        double assetNext = assetValue(1);
        // control
        Eigen::ArrayXd assetCurrentTab(1);
        assetCurrentTab(0) = assetCurrent;
        // make sure control is stricly between 0 and 1
        double control = std::min(std::max(0., p_control[0].getValue(AValue, assetCurrentTab)), 1.);
        // one step
        AValue(0) *= 1. + control * (assetNext - assetCurrent) / assetCurrent;
        // for return
        p_state.setPtStock(AValue);
        p_phiInOut(0) = control ;
    }


    /// \brief get number of regimes (here 1)
    inline int getNbRegime() const
    {
        return 1;
    }

    /// \brief get number of time step
    inline int getNbStep() const
    {
        return m_simulator->getNbStep();
    }

    /// \brief number of controls
    inline int getNbControl() const
    {
        return 1;
    }

    ///\brief store the simulator
    inline void setSimulator(const std::shared_ptr<MMMSimulator> &p_simulator)
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
    /// \brief defines the dimension to split for MPI parallelism
    ///        For each dimension return true is the direction can be split
    Eigen::Array< bool, Eigen::Dynamic, 1> getDimensionToSplit() const
    {
        Eigen::Array< bool, Eigen::Dynamic, 1> bDim = Eigen::Array< bool, Eigen::Dynamic, 1>::Constant(1, true);
        return  bDim ;
    }
};

#endif
