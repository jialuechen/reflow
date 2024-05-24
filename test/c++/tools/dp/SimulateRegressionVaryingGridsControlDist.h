
#ifndef SIMULATEREGREGRESSIONVARYINGGRIDSCONTROLDIST_H
#define SIMULATEREGREGRESSIONVARYINGGRIDSCONTROLDIST_H
#include <functional>
#include <memory>
#include <Eigen/Dense>
#include <boost/mpi.hpp>
#include "geners/BinaryFileArchive.hh"
#include "reflow/core/utils/comparisonUtils.h"
#include "reflow/core/grids/FullGrid.h"
#include "reflow/core/utils/StateWithStocks.h"
#include "reflow/dp/SimulateStepRegressionControlDist.h"
#include "reflow/dp/OptimizerDPBase.h"
#include "reflow/dp/SimulatorDPBase.h"


double SimulateRegressionVaryingGridsControlDist(const std::vector<double>    &p_timeChangeGrid,
        const std::vector<std::shared_ptr<reflow::FullGrid> >   &p_grids,
        const std::shared_ptr<reflow::OptimizerDPBase > &p_optimize,
        const std::function<double(const int &, const Eigen::ArrayXd &, const Eigen::ArrayXd &)>  &p_funcFinalValue,
        const Eigen::ArrayXd &p_pointStock,
        const int &p_initialRegime,
        const std::string   &p_fileToDump,
        const bool &p_bOneFile,
        const boost::mpi::communicator &p_world)
{
    // from the optimizer get back the simulation
    std::shared_ptr< reflow::SimulatorDPBase> simulator = p_optimize->getSimulator();
    int nbStep = simulator->getNbStep();
    std::vector< reflow::StateWithStocks> states;
    states.reserve(simulator->getNbSimul());
    for (int is = 0; is < simulator->getNbSimul(); ++is)
        states.push_back(reflow::StateWithStocks(p_initialRegime, p_pointStock, Eigen::ArrayXd::Zero(simulator->getDimension())));
    std::string toDump = p_fileToDump ;
    // test if one file generated
    if (!p_bOneFile)
        toDump +=  "_" + boost::lexical_cast<std::string>(p_world.rank());
    gs::BinaryFileArchive ar(toDump.c_str(), "r");
    // name for continuation object in archive
    std::string nameAr = "Continuation";
    // cost function
    Eigen::ArrayXXd costFunction = Eigen::ArrayXXd::Zero(p_optimize->getSimuFuncSize(), simulator->getNbSimul());
    for (int istep = 0; istep < nbStep; ++istep)
    {
        // get time step
        double nextTime = simulator->getCurrentStep() + simulator->getStep() ;
        int iTime = p_timeChangeGrid.size() - 1;
        while (reflow::isStrictlyLesser(nextTime, p_timeChangeGrid[iTime]))
            iTime--;       // conditional expectation operator
        // current time
        double timeCur = simulator->getCurrentStep();
        int iTimeCurrent = iTime;
        while (reflow::isStrictlyLesser(timeCur, p_timeChangeGrid[iTimeCurrent]))
            iTimeCurrent--;
        reflow::SimulateStepRegressionControlDist(ar, nbStep - 1 - istep, nameAr, p_grids[iTimeCurrent], p_grids[iTime], p_optimize, p_bOneFile, p_world).oneStep(states, costFunction);
        // new stochastic state
        Eigen::ArrayXXd particles =  simulator->stepForwardAndGetParticles();
        for (int is = 0; is < simulator->getNbSimul(); ++is)
            states[is].setStochasticRealization(particles.col(is));

    }
    // final : accept to exercise if not already done entirely
    for (int is = 0; is < simulator->getNbSimul(); ++is)
        costFunction(0, is) += p_funcFinalValue(states[is].getRegime(), states[is].getPtStock(), states[is].getStochasticRealization()) * simulator->getActu();

    return costFunction.mean();
}

#endif /* SIMULATEREGRESSIONVARYINGGRIDSCONTROLDIST_H */
