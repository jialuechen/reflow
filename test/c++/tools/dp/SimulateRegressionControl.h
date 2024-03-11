
#ifndef SIMULATEREGREGRESSIONCONTROL_H
#define SIMULATEREGREGRESSIONCONTROL_H
#include <Eigen/Dense>
#include <functional>
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <memory>
#include "geners/BinaryFileArchive.hh"
#include "libflow/core/utils/StateWithStocks.h"
#include "libflow/core/grids/SpaceGrid.h"
#include "libflow/regression/BaseRegression.h"
#include "libflow/dp/SimulateStepRegressionControl.h"
#include "libflow/dp/OptimizerBaseInterp.h"
#include "libflow/dp/SimulatorDPBase.h"

double SimulateRegressionControl(const std::shared_ptr<libflow::SpaceGrid> &p_grid,
                                 const std::shared_ptr<libflow::OptimizerBaseInterp > &p_optimize,
                                 const std::function<double(const int &, const Eigen::ArrayXd &, const Eigen::ArrayXd &)>   &p_funcFinalValue,
                                 const Eigen::ArrayXd &p_pointStock,
                                 const int &p_initialRegime,
                                 const std::string   &p_fileToDump
#ifdef USE_MPI
                                 , const boost::mpi::communicator &p_world
#endif
                                )
{
    // from the optimizer get back the simulator
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
        libflow::SimulateStepRegressionControl(ar, nbStep - 1 - istep, nameAr, p_grid, p_optimize
#ifdef USE_MPI
                                             , p_world
#endif
                                            ).oneStep(states, costFunction);
        // new stochastic state
        Eigen::ArrayXXd particles =  simulator->stepForwardAndGetParticles();
        for (int is = 0; is < simulator->getNbSimul(); ++is)
            states[is].setStochasticRealization(particles.col(is));

    }
    // final : accept to exercise if not already done entirely (here suppose one function to follow)
    for (int is = 0; is < simulator->getNbSimul(); ++is)
        costFunction(0, is) += p_funcFinalValue(states[is].getRegime(), states[is].getPtStock(), states[is].getStochasticRealization()) * simulator->getActu();
    // average gain/cost
    return costFunction.mean();
}
#endif /* SIMULATEREGRESSIONCONTROL_H */
