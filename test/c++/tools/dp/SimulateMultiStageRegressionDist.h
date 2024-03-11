
#ifndef SIMULATEMULTISTAGEREGRESSIONDIST_H
#define SIMULATEMULTISTAGEREGRESSIONDIST_H
#include <functional>
#include <memory>
#include <Eigen/Dense>
#include <boost/mpi.hpp>
#include "geners/BinaryFileArchive.hh"
#include "libflow/core/grids/FullGrid.h"
#include "libflow/core/utils/StateWithStocks.h"
#include "libflow/dp/SimulateStepMultiStageRegressionDist.h"
#include "libflow/dp/OptimizerMultiStageDPBase.h"
#include "libflow/dp/SimulatorMultiStageDPBase.h"


double SimulateMultiStageRegressionDist(const std::shared_ptr<libflow::FullGrid> &p_grid,
                                        const std::shared_ptr<libflow::OptimizerMultiStageDPBase > &p_optimize,
                                        const std::function<double(const int &, const Eigen::ArrayXd &, const Eigen::ArrayXd &)>  &p_funcFinalValue,
                                        const Eigen::ArrayXd &p_pointStock,
                                        const int &p_initialRegime,
                                        const std::string   &p_fileToDump,
                                        const bool &p_bOneFile,
                                        const boost::mpi::communicator &p_world)
{
    // from the optimizer get back the simulator
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
    std::cout << " FILEDUMPSIM " << toDump << std::endl ;
    // name for continuation object in archive
    std::string nameAr = "Continuation";
    // Name for deterministic continuation in archive
    std::string nameArContValDet =  "ContinuationDet";
    // cost function
    Eigen::ArrayXXd costFunction = Eigen::ArrayXXd::Zero(p_optimize->getSimuFuncSize(), simulator->getNbSimul());
    for (int istep = 0; istep < nbStep; ++istep)
    {
        // Number of transition achieved for the current time step
        int  nbPeriodsOfCurrentStep = simulator->getNbPeriodsInTransition();
        std::vector<Eigen::ArrayXXd> costFunctionPeriod(nbPeriodsOfCurrentStep);
        for (int iPeriod = 0; iPeriod < nbPeriodsOfCurrentStep; ++iPeriod)
            costFunctionPeriod[iPeriod] = Eigen::ArrayXXd::Zero(p_optimize->getSimuFuncSize(), simulator->getNbSimul());
        costFunctionPeriod[0] = costFunction;
        std::string toStorBellDet = nameArContValDet + boost::lexical_cast<std::string>(nbStep - 1 - istep);

        libflow::SimulateStepMultiStageRegressionDist(ar, nbStep - 1 - istep, nameAr, toStorBellDet, p_grid, p_grid, p_optimize, p_bOneFile, p_world).oneStep(states, costFunctionPeriod);

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

#endif /* SIMULATEMULTISTAGEREGRESSIONDIST_H */
