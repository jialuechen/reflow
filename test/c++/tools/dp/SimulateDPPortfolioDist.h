
#ifdef USE_MPI
#ifndef SIMULATEDPPORTFOLIODIST_H
#define SIMULATEDPPORTFOLIODIST_H
#include <functional>
#include <memory>
#include <Eigen/Dense>
#include <boost/mpi.hpp>
#include "geners/BinaryFileArchive.hh"
#include "libflow/core/grids/FullGrid.h"
#include "libflow/core/utils/StateWithStocks.h"
#include "libflow/dp/SimulateStepRegressionControlDist.h"
#include "libflow/dp/OptimizerNoRegressionDPBase.h"
#include "libflow/dp/SimulatorDPBase.h"

double SimulateDPPortfolioDist(const std::shared_ptr<libflow::FullGrid> &p_grid,
                               const std::shared_ptr<libflow::OptimizerNoRegressionDPBase > &p_optimize,
                               const std::function<double(const int &, const Eigen::ArrayXd &, const Eigen::ArrayXd &)>  &p_funcFinalValue,
                               const Eigen::ArrayXd &p_initialPortfolio,
                               const std::string   &p_fileToDump,
                               const bool &p_bOneFile,
                               const boost::mpi::communicator &p_world)
{
    // from the optimizer get back the simulator
    std::shared_ptr< libflow::SimulatorDPBase> simulator = p_optimize->getSimulator();
    int nbStep = simulator->getNbStep();
    std::vector< libflow::StateWithStocks> states;
    states.reserve(simulator->getNbSimul());
    Eigen::ArrayXXd particles =  simulator->getParticles();
    Eigen::ArrayXXd particlesNext =  simulator->stepForwardAndGetParticles();
    Eigen::ArrayXd  partStore(2);
    for (int is = 0; is < simulator->getNbSimul(); ++is)
    {
        partStore(0) = particles(0, is);
        partStore(1) = particlesNext(0, is);
        // only one regime
        int initialRegime = 0 ;
        states.push_back(libflow::StateWithStocks(initialRegime, p_initialPortfolio, partStore));
    }
    std::string toDump = p_fileToDump ;
    // test if one file generated
#ifdef USE_MPI
    if (!p_bOneFile)
        toDump +=  "_" + boost::lexical_cast<std::string>(p_world.rank());
#endif
    gs::BinaryFileArchive ar(toDump.c_str(), "r");
    // name for object in archive
    std::string nameAr = "OptimizePort";
    // store control on each simulation
    Eigen::ArrayXXd control = Eigen::ArrayXXd::Zero(p_optimize->getSimuFuncSize(), simulator->getNbSimul());
    for (int istep = 0; istep < nbStep; ++istep)
    {
        libflow::SimulateStepRegressionControlDist(ar, nbStep - 1 - istep, nameAr, p_grid, p_grid, p_optimize, p_bOneFile, p_world).oneStep(states, control);

        // new stochastic state
        if (istep < nbStep - 1)
        {
            particles = particlesNext;
            particlesNext =  simulator->stepForwardAndGetParticles();
            for (int is = 0; is < simulator->getNbSimul(); ++is)
            {
                partStore(0) = particles(0, is);
                partStore(1) = particlesNext(0, is);
                states[is].setStochasticRealization(partStore);
            }
        }
    }
    // final average of payoff
    Eigen::ArrayXd payOff(simulator->getNbSimul());
    for (int is = 0; is < simulator->getNbSimul(); ++is)
    {
        payOff(is) = p_funcFinalValue(states[is].getRegime(), states[is].getPtStock(), states[is].getStochasticRealization()) * simulator->getActu();
    }

    return payOff.mean();
}

#endif /* SIMULATEDPPORTFOLIODIST_H */
#endif
