// Copyright (C) 2016 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#ifndef SIMULATEREGREGRESSIONVARYINGGRIDS_H
#define SIMULATEREGREGRESSIONVARYINGGRIDS_H
#include <functional>
#include <memory>
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "libflow/core/utils/comparisonUtils.h"
#include "libflow/core/utils/StateWithStocks.h"
#include "libflow/core/grids/FullGrid.h"
#include "libflow/regression/BaseRegression.h"
#include "libflow/dp/SimulateStepRegression.h"
#include "libflow/dp/OptimizerDPBase.h"
#include "libflow/dp/SimulatorDPBase.h"


/** \file SimulateRegression.h
 *  \brief Defines a simple program showing how to use simulation
 *        A simple grid  is used
 *  \author Xavier Warin
 */

/// \brief Simulate the optimal strategy
///        The geometry of the  stock is time dependent
/// \param p_timeChangeGrid    date for changing grids
/// \param p_grids             grids depending on time
/// \param p_optimize               optimizer defining the optimisation between two time steps
/// \param p_funcFinalValue         function defining the final value
/// \param p_pointStock             initial point stock
/// \param p_initialRegime          regime at initial date
/// \param p_fileToDump             name of the file used to dump continuation values in optimization
/// \param p_world                  MPI communicator
double SimulateRegressionVaryingGrids(const std::vector<double>    &p_timeChangeGrid,
                                      const std::vector<std::shared_ptr<libflow::FullGrid> >   &p_grids,
                                      const std::shared_ptr<libflow::OptimizerDPBase > &p_optimize,
                                      const std::function<double(const int &, const Eigen::ArrayXd &, const Eigen::ArrayXd &)>   &p_funcFinalValue,
                                      const Eigen::ArrayXd &p_pointStock,
                                      const int &p_initialRegime,
                                      const std::string   &p_fileToDump
#ifdef USE_MPI
                                      , const boost::mpi::communicator &p_world
#endif
                                     )
{
    // from the optimizer get back the simulation
    std::shared_ptr< libflow::SimulatorDPBase> simulator = p_optimize->getSimulator();
    int nbStep = simulator->getNbStep();
    std::vector< libflow::StateWithStocks> states;
    states.reserve(simulator->getNbSimul());
    for (int is = 0; is < simulator->getNbSimul(); ++is)
        states.push_back(libflow::StateWithStocks(p_initialRegime, p_pointStock, Eigen::ArrayXd::Zero(simulator->getDimension())));
    gs::BinaryFileArchive ar(p_fileToDump.c_str(), "r");
    // name for continuation object in archive
    std::string nameAr = "Continuation";
    // cost function
    Eigen::ArrayXXd costFunction = Eigen::ArrayXXd::Zero(p_optimize->getSimuFuncSize(), simulator->getNbSimul());

    // iterate on time steps
    for (int istep = 0; istep < nbStep; ++istep)
    {
        // get time step
        double nextTime = simulator->getCurrentStep() + simulator->getStep() ;
        int iTime = p_timeChangeGrid.size() - 1;
        while (libflow::isStrictlyLesser(nextTime, p_timeChangeGrid[iTime]))
            iTime--;       // conditional expectation operator
        libflow::SimulateStepRegression(ar, nbStep - 1 - istep, nameAr, p_grids[iTime], p_optimize
#ifdef USE_MPI
                                      , p_world
#endif
                                     ).oneStep(states, costFunction);
        // new stochastic state : update the uncertainties
        Eigen::ArrayXXd particles =  simulator->stepForwardAndGetParticles();
        for (int is = 0; is < simulator->getNbSimul(); ++is)
            states[is].setStochasticRealization(particles.col(is));
    }
    // final : accept to exercise if not already done entirely  (here suppose one function to follow)
    for (int is = 0; is < simulator->getNbSimul(); ++is)
        costFunction(0, is) += p_funcFinalValue(states[is].getRegime(), states[is].getPtStock(), states[is].getStochasticRealization()) * simulator->getActu();
    // average gain/cost
    return costFunction.mean();
}
#endif /* SIMULATEREGRESSIONVARYINGGRIDS_H */
