
#ifndef DYNAMICHEDGEL2DIST_H
#define DYNAMICHEDGEL2DIST_H
#include <fstream>
#include <memory>
#include <functional>
#include "libflow/regression/BaseRegression.h"
#include "libflow/dp/FinalStepDPDist.h"
#include "libflow/dp/TransitionStepRegressionDPDist.h"
#include "libflow/core/parallelism/reconstructProc0Mpi.h"
#include "OptimizeOptionL2.h"

/* \file DynamicHedgeL2Dist.h
 * \brief Solve the global  hedging Asset problem
 *        according to the methodology in
 *        "Variance optimal hedging with application to Electricity markets", Xavier Warin, arXiv:1711.03733
 *        A simple grid  is used
 * \author Xavier Warin
 */

/// \brief Principal function to optimize  a problem
/// \param p_grid                  grid used for  deterministic state : position in futures
/// \param p_optimize              optimizer defining the optimisation between two time steps
/// \param p_regressor             regressor object
/// \param p_funcFinalValue        function defining the final value
/// \param p_ar                    archive to dump
///
template< class PriceModel>
double  DynamicHedgeL2Dist(const std::shared_ptr<libflow::FullGrid> &p_grid,
                           const std::shared_ptr< OptimizeOptionL2<PriceModel>  > &p_optimize,
                           std::shared_ptr<libflow::BaseRegression> &p_regressor,
                           const std::function<double(const int &, const Eigen::ArrayXd &, const Eigen::ArrayXd &)>   &p_funcFinalValue,
                           std::shared_ptr<gs::BinaryFileArchive>   &p_ar,
                           const boost::mpi::communicator &p_world)
{
    // from the optimizer get back the simulator
    std::shared_ptr< libflow::SimulatorDPBase> simulator = p_optimize->getSimulator();
    // store previous asset values
    Eigen::ArrayXXd assetPrev = simulator->getParticles();
    // to store difference
    Eigen::ArrayXXd diffAsset(assetPrev.rows(), simulator->getNbSimul());
    // number of step
    int nStepGlobal = simulator->getNbStep();
    std::vector< std::shared_ptr< Eigen::ArrayXXd > >  valuesNext = libflow::FinalStepDPDist(p_grid, p_optimize->getNbRegime(), p_optimize->getDimensionToSplit(), p_world)(p_funcFinalValue, assetPrev);
    // name for object in archive
    std::string nameAr = "Storage";
    // p_ar
    for (int iStep = 0; iStep < nStepGlobal; ++iStep)
    {
        if (p_world.rank() == 0)
            std::cout << " Step " << iStep << std::endl ;
        // set back and get particles
        Eigen::ArrayXXd  assetCurrent = simulator->stepBackwardAndGetParticles();
        // asset variations
        diffAsset = assetPrev  - assetCurrent;
        // update to store asset evolution and bounds on commands
        p_optimize->setCommandsAndAssetVar(diffAsset, assetCurrent);
        // conditional expectation operator
        p_regressor->updateSimulations(((iStep == (nStepGlobal - 1)) ? true : false), assetCurrent);
        // transition object
        libflow::TransitionStepRegressionDPDist transStep(p_grid, p_grid, std::static_pointer_cast<libflow::OptimizerDPBase>(p_optimize), p_world);
        std::pair< std::vector< std::shared_ptr< Eigen::ArrayXXd > >, std::vector< std::shared_ptr< Eigen::ArrayXXd > > > valuesAndControl  = transStep.oneStep(valuesNext, p_regressor);
        transStep.dumpContinuationValues(p_ar, nameAr, nStepGlobal - 1 - iStep, valuesNext, valuesAndControl.second, p_regressor, true);
        valuesNext = valuesAndControl.first;
        assetPrev = assetCurrent;
    }
    // reconstruct a small grid for interpolation
    Eigen::ArrayXd  noFuture = Eigen::ArrayXd::Zero(p_grid->getDimension());
    return libflow::reconstructProc0Mpi(noFuture, p_grid, valuesNext[0], p_optimize->getDimensionToSplit(), p_world).mean();
}

#endif

