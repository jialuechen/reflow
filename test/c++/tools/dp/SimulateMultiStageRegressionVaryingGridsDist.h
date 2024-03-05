// Copyright (C) 2023 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#ifndef SIMULATEMULTISTAGEREGRESSIONREGREGRESSIONVARYINGGRIDSDIST_H
#define SIMULATEMULTISTAGEREGRESSIONREGREGRESSIONVARYINGGRIDSDIST_H
#include <functional>
#include <memory>
#include <Eigen/Dense>
#include <boost/mpi.hpp>
#include "geners/BinaryFileArchive.hh"
#include "libflow/core/utils/comparisonUtils.h"
#include "libflow/core/grids/FullGrid.h"
#include "libflow/core/utils/StateWithStocks.h"
#include "libflow/dp/SimulateStepMultiStageRegressionDist.h"
#include "libflow/dp/OptimizerMultiStageDPBase.h"
#include "libflow/dp/SimulatorMultiStageDPBase.h"


/** \file SimulateMultiStageRegressionVaryingGridsDist.h
 *  \brief Defines a simple program showing how to use simulation
 *        A simple grid  is used
 *  \author Xavier Warin
 */


/// \brief Simulate the optimal strategy , mpi version
/// \param p_timeChangeGrid    date for changing grids
/// \param p_grids             grids depending on time
/// \param p_optimize          Optimiser defining the problem to solve
/// \param p_funcFinalValue     function defining the final value
/// \param p_pointStock         initial point stock
/// \param p_initialRegime      regime at initial date
/// \param p_fileToDump         name associated to dumped bellman values
/// \param p_bOneFile           do we store continuation values  in only one file
/// \param p_world              MPI communicator
double SimulateMultiStageRegressionVaryingGridsDist(const std::vector<double>    &p_timeChangeGrid,
        const std::vector<std::shared_ptr<libflow::FullGrid> >   &p_grids,
        const std::shared_ptr<libflow::OptimizerMultiStageDPBase > &p_optimize,
        const std::function<double(const int &, const Eigen::ArrayXd &, const Eigen::ArrayXd &)>  &p_funcFinalValue,
        const Eigen::ArrayXd &p_pointStock,
        const int &p_initialRegime,
        const std::string   &p_fileToDump,
        const bool &p_bOneFile,
        const boost::mpi::communicator &p_world)
{
    // from the optimizer get back the simulation
    std::shared_ptr< libflow::SimulatorMultiStageDPBase> simulator = p_optimize->getSimulator();
    int nbStep = simulator->getNbStep();
    std::vector< libflow::StateWithStocks> states;
    states.reserve(simulator->getNbSimul());
    for (int is = 0; is < simulator->getNbSimul(); ++is)
        states.push_back(libflow::StateWithStocks(p_initialRegime, p_pointStock, Eigen::ArrayXd::Zero(simulator->getDimension())));
    std::string toDump = p_fileToDump ;
    // test if one file generated
    if (!p_bOneFile)
        toDump +=  "_" + boost::lexical_cast<std::string>(p_world.rank());
    std::shared_ptr<gs::BinaryFileArchive> ar =  std::make_shared<gs::BinaryFileArchive>(toDump.c_str(), "r");
    // name for continuation object in archive
    std::string nameAr = "Continuation";
    // Name for deterministic continuation in archive
    std::string nameArContValDet =  "ContinuationDet";
    // cost function
    Eigen::ArrayXXd costFunction = Eigen::ArrayXXd::Zero(p_optimize->getSimuFuncSize(), simulator->getNbSimul());
    for (int istep = 0; istep < nbStep; ++istep)
    {
        // get time step
        double currTime = simulator->getCurrentStep() ;
        double nextTime = currTime + simulator->getStep() ;
        int iTimeNext = p_timeChangeGrid.size() - 1;
        while (libflow::isStrictlyLesser(nextTime, p_timeChangeGrid[iTimeNext]))
            iTimeNext--;
        int iTime = iTimeNext;
        while (libflow::isStrictlyLesser(currTime, p_timeChangeGrid[iTime]))
            iTime--;
        // Number of transition achieved for the current time step
        int  nbPeriodsOfCurrentStep = simulator->getNbPeriodsInTransition();
        std::vector<Eigen::ArrayXXd> costFunctionPeriod(nbPeriodsOfCurrentStep);
        for (int iPeriod = 0; iPeriod < nbPeriodsOfCurrentStep; ++iPeriod)
            costFunctionPeriod[iPeriod] = Eigen::ArrayXXd::Zero(p_optimize->getSimuFuncSize(), simulator->getNbSimul());
        costFunctionPeriod[0] = costFunction;
        std::string toStorBellDet = nameArContValDet + boost::lexical_cast<std::string>(nbStep - 1 - istep);

        libflow::SimulateStepMultiStageRegressionDist(ar, nbStep - 1 - istep, nameAr, toStorBellDet,
                p_grids[iTime], p_grids[iTimeNext], p_optimize, p_bOneFile, p_world).oneStep(states, costFunctionPeriod);

        // new stochastic state
        Eigen::ArrayXXd particles =  simulator->stepForwardAndGetParticles();
        for (int is = 0; is < simulator->getNbSimul(); ++is)
            states[is].setStochasticRealization(particles.col(is));

        // store current cost function
        costFunction = costFunctionPeriod[nbPeriodsOfCurrentStep - 1];
    }
    // final : accept to exercise if not already done entirely (here suppose one function to follow)
    for (int is = 0; is < simulator->getNbSimul(); ++is)
        costFunction(0, is) += p_funcFinalValue(states[is].getRegime(), states[is].getPtStock(), states[is].getStochasticRealization()) * simulator->getActu();

    return costFunction.mean();
}

#endif /* SIMULATEREGRESSIONVARYINGGRIDS_H */
