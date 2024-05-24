
#ifndef SIMULATEREGTREECONTROL_H
#define SIMULATEREGTREECONTROL_H
#include <Eigen/Dense>
#include <functional>
#include <memory>
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include "geners/BinaryFileArchive.hh"
#include "reflow/tree/StateTreeStocks.h"
#include "reflow/core/grids/SpaceGrid.h"
#include "reflow/dp/SimulateStepTreeControl.h"
#include "reflow/dp/OptimizerDPTreeBase.h"
#include "reflow/dp/SimulatorDPBaseTree.h"


double SimulateTreeControl(const std::shared_ptr<reflow::SpaceGrid> &p_grid,
                           const std::shared_ptr<reflow::OptimizerDPTreeBase > &p_optimize,
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
    std::shared_ptr< reflow::SimulatorDPBaseTree> simulator = p_optimize->getSimulator();
    int nbStep = simulator->getNbStep();
    std::vector< reflow::StateTreeStocks> states;
    states.reserve(simulator->getNbSimul());
    for (int is = 0; is < simulator->getNbSimul(); ++is)
        states.push_back(reflow::StateTreeStocks(p_initialRegime, p_pointStock, 0));
    gs::BinaryFileArchive ar(p_fileToDump.c_str(), "r");
    // name for continuation object in archive
    std::string nameAr = "Continuation";
    // cost function
    Eigen::ArrayXXd costFunction = Eigen::ArrayXXd::Zero(p_optimize->getSimuFuncSize(), simulator->getNbSimul());
    // iterate on time steps
    for (int istep = 0; istep < nbStep; ++istep)
    {
        reflow::SimulateStepTreeControl(ar, nbStep - 1 - istep, nameAr, p_grid, p_optimize
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
        costFunction(0, is) += p_funcFinalValue(states[is].getRegime(), states[is].getPtStock(), simulator->getValueAssociatedToNode(states[is].getStochasticRealization()));
    return costFunction.mean();
}
#endif /* SIMULATETREECONTROL_H */
