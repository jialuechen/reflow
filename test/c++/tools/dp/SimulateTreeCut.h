
#ifndef SIMULATEREGTREECUT_H
#define SIMULATEREGTREECUT_H
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <Eigen/Dense>
#include <functional>
#include <memory>
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include "geners/BinaryFileArchive.hh"
#include "libflow/core/grids/FullGrid.h"
#include "libflow/tree/StateTreeStocks.h"
#include "libflow/dp/SimulateStepTreeCut.h"
#include "libflow/dp/OptimizerDPCutTreeBase.h"
#include "libflow/dp/SimulatorDPBaseTree.h"


/** \file SimulateTreeCut.h
 *  \brief Defines a simple program showing how to use simulations when optimizaton achived with transition problems solved with cuts.
 *        A simple grid  is used, uncertainties are  discrete and defined on a tree
 *  \author Xavier Warin
 */

/// \brief Simulate the optimal strategy , Bellman cuts used to allow LP resolution of transition problems , uncertainties on a tree
/// \param p_grid                   grid used for  deterministic state (stocks for example)
/// \param p_optimize               optimizer defining the optimization between two time steps
/// \param p_funcFinalValue         function defining the final value cuts
/// \param p_pointStock             initial point stock
/// \param p_initialRegime          regime at initial date
/// \param p_fileToDump             name of the file used to dump continuation values in optimization
/// \param p_world                  MPI communicator
double SimulateTreeCut(const std::shared_ptr<libflow::FullGrid> &p_grid,
                       const std::shared_ptr<libflow::OptimizerDPCutTreeBase > &p_optimize,
                       const std::function< Eigen::ArrayXd(const int &, const Eigen::ArrayXd &, const Eigen::ArrayXd &)>   &p_funcFinalValue,
                       const Eigen::ArrayXd &p_pointStock,
                       const int &p_initialRegime,
                       const std::string   &p_fileToDump
#ifdef USE_MPI
                       , const boost::mpi::communicator &p_world
#endif
                      )
{
    // from the optimizer get back the simulator
    std::shared_ptr< libflow::SimulatorDPBaseTree> simulator = p_optimize->getSimulator();
    int nbStep = simulator->getNbStep();
    std::vector< libflow::StateTreeStocks> states;
    states.reserve(simulator->getNbSimul());
    for (int is = 0; is < simulator->getNbSimul(); ++is)
        states.push_back(libflow::StateTreeStocks(p_initialRegime, p_pointStock, 0));
    gs::BinaryFileArchive ar(p_fileToDump.c_str(), "r");
    // name for continuation object in archive
    std::string nameAr = "ContinuationTree";
    // cost function
    Eigen::ArrayXXd costFunction = Eigen::ArrayXXd::Zero(p_optimize->getSimuFuncSize(), simulator->getNbSimul());
    // iterate on time steps
    for (int istep = 0; istep < nbStep; ++istep)
    {
        libflow::SimulateStepTreeCut(ar, nbStep - 1 - istep, nameAr, p_grid, p_optimize
#ifdef USE_MPI
                                   , p_world
#endif
                                  ).oneStep(states, costFunction);
        // new date
        simulator->stepForward();
        for (int is = 0; is < simulator->getNbSimul(); ++is)
            states[is].setStochasticRealization(simulator->getNodeAssociatedToSim(is));
    }
    // final : accept to exercise if not already done entirely (here suppose one function to follow)
    for (int is = 0; is < simulator->getNbSimul(); ++is)
        costFunction(0, is) += p_funcFinalValue(states[is].getRegime(), states[is].getPtStock(), simulator->getValueAssociatedToNode(states[is].getStochasticRealization()))(0);
    // average gain/cost
    return costFunction.mean();
}
#endif /* SIMULATETREECUT_H */
