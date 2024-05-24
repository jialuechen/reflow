
#ifndef OPTIMIZEGASSTORAGECUT_H
#define OPTIMIZEGASSTORAGECUT_H
#include "ClpSimplex.hpp"
#include <memory>
#include <Eigen/Dense>
#include "reflow/core/utils/StateWithStocks.h"
#include "reflow/core/grids/SpaceGrid.h"
#include "reflow/regression/BaseRegression.h"
#include "reflow/core/grids/Interpolator.h"
#include "reflow/regression/ContinuationCuts.h"
#include "reflow/dp/OptimizerDPCutBase.h"
#include "test/c++/tools/dp/OptimizeGasStorageCutBase.h"

/** \file OptimizeGasStorageCut.h
 *  \brief  Simple example of a gas storage optimizer
 *          - injection rate, withdrawal rates are independent of the storage level
 *          - the size of the storage is constant, minimum gas level is 0
 *          .
 *          Designed to work in parallel/multi threaded framework
 *          The local  optimization is solved using a LP on one time step
 *          Regression used for conditional expectation
 *          LP defined in base class
 *  \author Xavier Warin
 */

/// \class OptimizeGasStorageCut OptimizeGasStorageCut.h
/// Defines a simple gas storage for optimization and simulation using cuts and a LP solver
/// No constraints on the storage at the end of optimization period (so the storage will be empty)
/// - when injecting the gain is  \f$ - C_{inj} ( S+ \kappa_{inj} )\f$
/// - when withdrawing the gain is  \f$  C_{with} ( S- \kappa_{with} )\f$
/// .
template< class Simulator>
class OptimizeGasStorageCut : public reflow::OptimizerDPCutBase, OptimizeGasStorageCutBase
{
private :

    /// store the simulator
    std::shared_ptr<Simulator> m_simulator;

public :

    /// \brief Constructor
    /// \param  p_injectionRate     injection rate per time step
    /// \param  p_withdrawalRate    withdrawal rate between two time steps
    /// \param  p_injectionCost     injection cost
    /// \param  p_withdrawalCost    withdrawal cost
    OptimizeGasStorageCut(const double   &p_injectionRate, const double &p_withdrawalRate,
                          const double &p_injectionCost, const double &p_withdrawalCost):
        OptimizeGasStorageCutBase(p_injectionRate, p_withdrawalRate, p_injectionCost, p_withdrawalCost)  {}

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
    /// \return    For each regimes (column) gives the solution for each particle , and cut (row)
    ///            For a given simulation , cuts components (C) at a point stock \$ \bar S \f$  are given such that the cut is given by
    ///            \f$  C[0] + \sum_{i=1}^d C[i] (S_i - \bat S_i)   \f$
    Eigen::ArrayXXd  stepOptimize(const   std::shared_ptr< reflow::SpaceGrid> &p_grid, const Eigen::ArrayXd   &p_stock,
                                  const std::vector<reflow::ContinuationCuts> &p_condEsp) const
    {
        int nbSimul = m_simulator->getNbSimul();
        // Spot price : here given by a composition of
        //  -the method getParticles from the simulator : its gives the regression factor of the models
        //  -the method fromParticlesToSpot  reconstructing the spot
        Eigen::ArrayXd spotPrice =  m_simulator->fromParticlesToSpot(m_simulator->getParticles()).array();
        // constraints on storage commands
        // level if injection
        double maxStorage = p_grid->getExtremeValues()[0][1];
        double injectionMax = std::min(maxStorage - p_stock(0), m_injectionRate);
        // level withdrawal (max > 0)
        double minStorage = p_grid->getExtremeValues()[0][0];
        double withdrawalMax = std::min(p_stock(0) - minStorage, m_withdrawalRate);
        // hyperCube for cuts
        Eigen::ArrayXXd hyperCube(1, 2);
        hyperCube(0, 0) = p_stock(0) - withdrawalMax;
        hyperCube(0, 1) =  p_stock(0) + injectionMax;
        // get back cuts for all simulatons  \f$ \hat a_0 + \sum_{i=1}^d a_i x_i\f$
        Eigen::ArrayXXd  cuts = p_condEsp[0].getCutsAllSimulations(hyperCube);
        // nb cuts
        int nbDimCut = p_grid->getDimension() + 1;
        // solution for return (only one regime)
        Eigen::ArrayXXd solution(nbDimCut * nbSimul, 1);
        // cuts for on simulation
        Eigen::ArrayXXd cutsASim(nbDimCut, cuts.cols());
        // to store vales and derivatives
        Eigen::ArrayXd valueAndDerivatives(nbDimCut);
        Eigen::ArrayXd stateFollowing(p_stock.size());
        double gain ;
        // nest on  simulation
        for (int is  = 0; is < nbSimul; ++is)
        {
            for (int ic = 0; ic <  nbDimCut; ++ic)
                cutsASim.row(ic) = cuts.row(is + ic * nbSimul);

            // Solve LP
            createAndSolveLP(cutsASim, p_stock(0), p_grid, spotPrice(is), valueAndDerivatives, stateFollowing, gain);

            // copy
            for (int ic = 0; ic <  nbDimCut; ++ic)
                solution(ic * nbSimul + is, 0) = valueAndDerivatives(ic);
        }

        return solution;
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
    void stepSimulate(const std::shared_ptr< reflow::SpaceGrid>   &p_grid, const std::vector< reflow::ContinuationCuts  > &p_continuation,
                      reflow::StateWithStocks &p_state, Eigen::Ref<Eigen::ArrayXd> p_phiInOut) const
    {
        // optimal stock  attained
        Eigen::ArrayXd ptStockCur = p_state.getPtStock();
        Eigen::ArrayXd ptStockMax(ptStockCur);
        // initial storage
        double storageInit = p_state.getPtStock()(0);
        // spot price
        double spotPrice = m_simulator->fromOneParticleToSpot(p_state.getStochasticRealization());
        // if injection
        double maxStorage = p_grid->getExtremeValues()[0][1];
        double injectionMax = std::min(maxStorage - storageInit, m_injectionRate);
        // if withdrawal
        double minStorage = p_grid->getExtremeValues()[0][0];
        double withdrawalMax = std::min(storageInit - minStorage, m_withdrawalRate);
        // hyperCube for cuts
        Eigen::ArrayXXd hyperCube(1, 2);
        hyperCube(0, 0) = storageInit - withdrawalMax;
        hyperCube(0, 1) =  storageInit + injectionMax;
        // get back cuts for all simulatons  \f$ \hat a_0 + \sum_{i=1}^d a_i x_i\f$
        Eigen::ArrayXXd  cuts = p_continuation[0].getCutsASim(hyperCube, p_state.getStochasticRealization());
        // nb cuts
        int nbDimCut = p_grid->getDimension() + 1;
        // to store vales and derivatives
        Eigen::ArrayXd valueAndDerivatives(nbDimCut);
        Eigen::ArrayXd stateFollowing(p_grid->getDimension());
        double gain ;
        // Solve LP
        createAndSolveLP(cuts, storageInit, p_grid, spotPrice, valueAndDerivatives, stateFollowing, gain);

        // for return
        p_state.setPtStock(stateFollowing);
        p_phiInOut(0) += gain ;
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
#endif /* OPTIMIZEGASSTORAGECUT_H */
