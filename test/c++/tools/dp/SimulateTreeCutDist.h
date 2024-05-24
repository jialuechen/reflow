
#ifndef SIMULATEREGTREECUTDIST_H
#define SIMULATEREGTREECUTDIST_H
#include <functional>
#include <memory>
#include <Eigen/Dense>
#include <boost/mpi.hpp>
#include "geners/BinaryFileArchive.hh"
#include "reflow/core/grids/FullGrid.h"
#include "reflow/tree/StateTreeStocks.h"
#include "reflow/dp/SimulateStepTreeCutDist.h"
#include "reflow/dp/OptimizerDPCutBase.h"
#include "reflow/dp/SimulatorDPBase.h"


double SimulateTreeCutDist(const std::shared_ptr<reflow::FullGrid> &p_grid,
                           const std::shared_ptr<reflow::OptimizerDPCutTreeBase > &p_optimize,
                           const std::function< Eigen::ArrayXd(const int &, const Eigen::ArrayXd &, const Eigen::ArrayXd &)>  &p_funcFinalValue,
                           const Eigen::ArrayXd &p_pointStock,
                           const int &p_initialRegime,
                           const std::string   &p_fileToDump,
                           const bool &p_bOneFile,
                           const boost::mpi::communicator &p_world)
{
    // from the optimizer get back the simulator
    std::shared_ptr< reflow::SimulatorDPBaseTree> simulator = p_optimize->getSimulator();
    int nbStep = simulator->getNbStep();
    std::vector< reflow::StateTreeStocks> states;
    states.reserve(simulator->getNbSimul());
    for (int is = 0; is < simulator->getNbSimul(); ++is)
        states.push_back(reflow::StateTreeStocks(p_initialRegime, p_pointStock, 0));
    std::string toDump = p_fileToDump ;
    // test if one file generated
    if (!p_bOneFile)
        toDump +=  "_" + boost::lexical_cast<std::string>(p_world.rank());
    gs::BinaryFileArchive ar(toDump.c_str(), "r");
    // name for continuation object in archive
    std::string nameAr = "ContinuationTree";
    // cost function
    Eigen::ArrayXXd costFunction = Eigen::ArrayXXd::Zero(p_optimize->getSimuFuncSize(), simulator->getNbSimul());
    for (int istep = 0; istep < nbStep; ++istep)
    {
        reflow::SimulateStepTreeCutDist(ar, nbStep - 1 - istep, nameAr, p_grid, p_optimize, p_bOneFile, p_world).oneStep(states, costFunction);

        // new date
        simulator->stepForward();
        for (int is = 0; is < simulator->getNbSimul(); ++is)
            states[is].setStochasticRealization(simulator->getNodeAssociatedToSim(is));
    }
    // final : accept to exercise if not already done entirely (here suppose one function to follow)
    for (int is = 0; is < simulator->getNbSimul(); ++is)
        costFunction(0, is) += p_funcFinalValue(states[is].getRegime(), states[is].getPtStock(), simulator->getValueAssociatedToNode(states[is].getStochasticRealization()))(0);

    return costFunction.mean();
}

#endif /* SIMULATETREECUTDIST_H */
