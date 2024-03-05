
#ifndef SIMULATEREGREGRESSIONWITHHEDGE_H
#define SIMULATEREGREGRESSIONWITHHEDGE_H
#include <Eigen/Dense>
#include <functional>
#include <memory>
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include "geners/BinaryFileArchive.hh"
#include "libflow/core/utils/StateWithStocks.h"
#include "libflow/core/grids/SpaceGrid.h"
#include "libflow/regression/BaseRegression.h"
#include "libflow/dp/SimulateStepRegression.h"
#include "libflow/dp/OptimizerDPBase.h"
#include "libflow/dp/SimulatorDPBase.h"


/** \file SimulateRegressionWithHedge.h
 *  \brief Defines a simple program showing how to use simulations
 *        A simple grid  is used.
 *       In this example the position in hedging is followed for the first component of the price
 *  \author Xavier Warin
 */

/// \brief Simulate the optimal strategy ,
/// \param p_grid                   grid used for  deterministic state (stocks for example)
/// \param p_optimize               optimizer defining the optimization between two time steps
/// \param p_funcFinalValue         function defining the final value
/// \param p_pointStock             initial point stock
/// \param p_initialRegime          regime at initial date
/// \param p_fileToDump             name of the file used to dump continuation values in optimization
/// \return an array with (average simulation value, average hedge position, std without hedge, std with hedge)
/// \param p_world                  MPI communicator
Eigen::Array4d  SimulateRegressionWithHedge(const std::shared_ptr<libflow::SpaceGrid> &p_grid,
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
    // hedge part
    Eigen::ArrayXd hedge = Eigen::ArrayXd::Zero(simulator->getNbSimul());
    // actualization for one step
    double actuStep = simulator->getActuStep();
    // iterate on time steps
    for (int istep = 0; istep < nbStep; ++istep)
    {
        libflow::SimulateStepRegression(ar, nbStep - 1 - istep, nameAr, p_grid, p_optimize
#ifdef USE_MPI
                                      , p_world
#endif
                                     ).oneStep(states, costFunction);
        // get asset value at the current date
        Eigen::ArrayXXd price = simulator->getParticles();
        // new stochastic state
        Eigen::ArrayXXd newPrice =  simulator->stepForwardAndGetParticles();
        // in cost function first is the pay off , in second place the delta for the first price in the simulation model -> move the hedge in the cost function
        for (int is = 0; is < simulator->getNbSimul(); ++is)
            hedge(is) += costFunction(1, is) * (newPrice(0, is) * actuStep - price(0, is));
        // stochastic state is updated
        for (int is = 0; is < simulator->getNbSimul(); ++is)
            states[is].setStochasticRealization(newPrice.col(is));
    }
    // final : accept to exercise if not already done entirely
    for (int is = 0; is < simulator->getNbSimul(); ++is)
        costFunction(0, is) += p_funcFinalValue(states[is].getRegime(), states[is].getPtStock(), states[is].getStochasticRealization()) * simulator->getActu();
    // average without hedge
    double meanWOHedge = costFunction.row(0).mean();
    // std deviation without hedge
    double stdWOHedge = sqrt(pow(costFunction.row(0), 2.).mean() - pow(meanWOHedge, 2.));
    for (int is = 0; is < simulator->getNbSimul(); ++is)
        costFunction(0, is) -= hedge(is);
    // average with  hedge
    double meanWHedge = costFunction.row(0).mean();
    // std deviation with  edge
    double stdWHedge = sqrt(pow(costFunction.row(0), 2.).mean() - pow(meanWOHedge, 2.));
    // return
    Eigen::Array4d ret;
    ret(0) = meanWHedge;
    ret(1) =  hedge.mean();
    ret(2) =  stdWOHedge;
    ret(3) =   stdWHedge;
    return ret;
}
#endif /* SIMULATEREGRESSIONWITHHEDGE_H */
