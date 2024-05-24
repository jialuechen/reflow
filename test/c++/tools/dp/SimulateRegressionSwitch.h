#ifndef SIMULATEREGRESSIONSWITCH_H
#define SIMULATEREGRESSIONSWITCH_H
#include <fstream>
#include <utility>
#include <functional>
#include <memory>
#include <vector>
#include <string>
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "reflow/core/utils/StateWithIntState.h"
#include "reflow/core/grids/RegularSpaceIntGrid.h"
#include "reflow/regression/BaseRegression.h"
#include "reflow/dp/SimulateStepSwitch.h"
#include "reflow/dp/OptimizerSwitchBase.h"

template<  class Optimizer, class Simulator>
std::pair<  double, Eigen::ArrayXi > SimulateRegressionSwitch(const std::vector< std::shared_ptr<reflow::RegularSpaceIntGrid> >  &p_grid,
        const std::shared_ptr< Optimizer > &p_optimize,
        const Eigen::ArrayXi &p_pointState,
        const int &p_initialRegime,
        const std::string   &p_fileToDump,
        const std::string    &p_fileOutput,
        const int &p_nbSimulPlot
#ifdef USE_MPI
        , const boost::mpi::communicator &p_world
#endif
                                                             )
{
    // from the optimizer get back the simulator
    std::shared_ptr< Simulator> simulator = p_optimize->getSimulatorDerived();
    int nbStep = simulator->getNbStep();
    std::vector< reflow::StateWithIntState> states;
    states.reserve(simulator->getNbSimul());
    for (int is = 0; is < simulator->getNbSimul(); ++is)
        states.push_back(reflow::StateWithIntState(p_initialRegime, p_pointState, Eigen::ArrayXd::Zero(simulator->getDimension())));
    gs::BinaryFileArchive ar(p_fileToDump.c_str(), "r");
    // name for continuation object in archive
    std::string nameAr = "Continuation";
    // cost function
    Eigen::ArrayXXd costFunction = Eigen::ArrayXXd::Zero(p_optimize->getSimuFuncSize(), simulator->getNbSimul());
    // get back spot
    Eigen::ArrayXXd spot = simulator->getSpot();
    int nbSpot = spot.rows();
    std::shared_ptr< std::fstream> fileForRegimeFin ;
    std::vector< std::shared_ptr< std::fstream> > fileForSpotFin(nbSpot);
#ifdef USE_MPI
    if (p_world.rank() == 0)
    {
#endif
        std::string fileForRegime = p_fileOutput + "Regime";
        fileForRegimeFin = std::make_shared<std::fstream>(fileForRegime.c_str(), std::fstream::out);
        for (int is = 0; is < p_nbSimulPlot - 1; ++is)
            *fileForRegimeFin <<  p_initialRegime << " , ";
        *fileForRegimeFin <<  p_initialRegime  << std::endl ;
        for (int ispot = 0; ispot < nbSpot; ++ispot)
        {
            std::string fileForSpot = p_fileOutput + "Spot" + std::to_string(ispot);
            fileForSpotFin[ispot] = std::make_shared<std::fstream>(fileForSpot.c_str(), std::fstream::out);
            for (int is = 0; is < p_nbSimulPlot - 1; ++is)
                *fileForSpotFin[ispot] <<  spot(ispot, is) << " , ";
            *fileForSpotFin[ispot] <<  spot(ispot, p_nbSimulPlot - 1)  << std::endl ;
        }
#ifdef USE_MPI
    }
#endif

    // to store min in each regime to check constraints
    Eigen::ArrayXi minInEachRegime = Eigen::ArrayXi::Constant(p_optimize->getNbRegime(), 10000);
    Eigen::ArrayXi iRegPrev = Eigen::ArrayXi::Constant(simulator->getNbSimul(), p_initialRegime);
    Eigen::ArrayXi iRegLoc(simulator->getNbSimul());
    Eigen::ArrayXi iLast = Eigen::ArrayXi::Constant(simulator->getNbSimul(), 1);

    // iterate on time steps
    for (int istep = 0; istep < nbStep; ++istep)
    {
        reflow::SimulateStepSwitch(ar, nbStep - 1 - istep, nameAr, p_grid, std::static_pointer_cast<reflow::OptimizerSwitchBase>(p_optimize)
#ifdef USE_MPI
                                  , p_world
#endif
                                 ).oneStep(states, costFunction);
        // new stochastic state
        Eigen::ArrayXXd particles =  simulator->stepForwardAndGetParticles();
        for (int is = 0; is < simulator->getNbSimul(); ++is)
            states[is].setStochasticRealization(particles.col(is));
        // dump
#ifdef USE_MPI
        if (p_world.rank() == 0)
        {
#endif
            for (int is = 0; is < p_nbSimulPlot - 1; ++is)
                *fileForRegimeFin <<  states[is].getRegime() << " , ";
            *fileForRegimeFin <<  states[p_nbSimulPlot - 1].getRegime()  << std::endl ;
            spot = simulator->getSpot();
            for (int ispot = 0; ispot < nbSpot; ++ispot)
            {
                for (int is = 0; is < p_nbSimulPlot - 1; ++is)
                    *fileForSpotFin[ispot] <<  spot(ispot, is) << " , ";
                *fileForSpotFin[ispot] <<  spot(ispot, p_nbSimulPlot - 1)  << std::endl ;
            }
#ifdef USE_MPI
        }
#endif
        // to check constraints
        for (int is = 0; is < simulator->getNbSimul(); ++is)
        {
            int iReg = states[is].getRegime();
            if (iReg == iRegPrev(is))
            {
                iLast(is) += 1;
            }
            else
            {
                minInEachRegime(iRegPrev(is)) = std::min(minInEachRegime(iRegPrev(is)), iLast(is));
                iLast(is) = 1.;
            }
            iRegPrev(is) = iReg;
        }

    }
    // average gain/cost
    return std::make_pair(costFunction.mean(), minInEachRegime);
}
#endif /* SIMULATEREGRESSIONSWITCH_H */
