
#ifndef BACKWARDFORWARDSDDP_H
#define BACKWARDFORWARDSDDP_H
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <map>
#include <memory>
#include <boost/timer/timer.hpp>
#include <Eigen/Dense>
#include "geners/Record.hh"
#include "libflow/sddp/SDDPFinalCut.h"
#include "libflow/sddp/OptimizerSDDPBase.h"
#include "libflow/sddp/backwardSDDP.h"
#include "libflow/sddp/forwardSDDP.h"
#include "libflow/sddp/SDDPFinalCut.h"
#include "libflow/sddp/SDDPVisitedStatesGeners.h"

/** \file backwardForwardSDDP.h
 * \brief alternates forward and backward resolution
 * \author Xavier Warin
 */
namespace libflow
{


/// \brief Achieve forward and backward sweep by SDDP
/// \param  p_optimizer           defines the optimiser necessary to optimize a step for one simulation solving a LP
/// \param  p_nbSimulCheckForSimu defines the number of simulations to check convergence
/// \param  p_initialState        initial state at the beginning of simulation
/// \param  p_finalCut            object of final cuts
/// \param  p_dates               vector of exercised dates, last date corresponds to the final cut object
/// \param  p_meshForReg          number of mesh for regression in each direction
/// \param  p_nameRegressor       name of the archive to store regressors
/// \param  p_nameCut             name of the archive to store cuts
/// \param  p_nameVisitedStates   name of the archive to store visited states
/// \param  p_iter                maximum iteration of SDDP, on return the number of iterations achieved
/// \param  p_accuracy            accuracy asked , on return estimation of accuracy achieved (expressed in %)
/// \param  p_nStepConv           every p_nStepConv convergence is checked
/// \param  p_outputStream        dump all print messages
/// \param  p_world               MPI communicator
/// \param  p_bPrintTime          if true print time at each backward and forward step
/// \return backward and forward valorization
template<  class LocalRegressionForSDDP>
std::pair<double, double> backwardForwardSDDP(const std::shared_ptr<OptimizerSDDPBase> &p_optimizer,
        const int   &p_nbSimulCheckForSimu,
        const Eigen::ArrayXd &p_initialState,
        const SDDPFinalCut &p_finalCut,
        const Eigen::ArrayXd &p_dates,
        const Eigen::ArrayXi &p_meshForReg,
        const std::string &p_nameRegressor,
        const std::string &p_nameCut,
        const std::string &p_nameVisitedStates,
        int &p_iter,
        double &p_accuracy,
        const int &p_nStepConv,
        std::ostream &p_outputStream,
#ifdef USE_MPI
        const boost::mpi::communicator &p_world,
#endif
        bool  p_bPrintTime = false)
{
    // get back simulators
    std::shared_ptr<SimulatorSDDPBase> simulatorForOptim = p_optimizer->getSimulatorBackward();
    std::shared_ptr<SimulatorSDDPBase> simulatorForSim = p_optimizer->getSimulatorForward();

#ifdef USE_MPI
    int iTask = p_world.rank();
#else
    int iTask = 0;
#endif
    assert(almostEqual<double>(p_dates(0), 0., 10));
    // cpu timers
    boost::timer::cpu_timer globalTimer;

    if (iTask == 0)
    {
        // create archive for regressors
        gs::BinaryFileArchive archiveForRegressor(p_nameRegressor.c_str(), "w");
        // create a first set of admissible states
        gs::BinaryFileArchive archiveForInitialState(p_nameVisitedStates.c_str(), "w");
        // vector of states
        std::vector< std::unique_ptr< SDDPVisitedStates > > vecSetOfStates(p_dates.size() - 1);
        // create regressor : each date except the last
        for (int idate = p_dates.size() - 2; idate >  0; --idate)
        {
            simulatorForOptim->updateDateIndex(idate);
            // initial regressor
            Eigen::ArrayXXd particlesCurrent = simulatorForOptim->getParticles();
            LocalRegressionForSDDP regCurrent(false, particlesCurrent, p_meshForReg);
            archiveForRegressor << gs::Record(regCurrent, "Regressor", "Top");
            std::unique_ptr< SDDPVisitedStates> setOfStates = std::make_unique<SDDPVisitedStates>(regCurrent.getNbMeshTotal());
            // offset  in admissible dates
            std::shared_ptr< Eigen::ArrayXd > anAdmissibleState = std::make_shared<Eigen::ArrayXd>(p_optimizer->oneAdmissibleState(p_dates(idate)));
            setOfStates->addVisitedStateForAll(anAdmissibleState, regCurrent);
            vecSetOfStates[idate] = move(setOfStates);
        }
        simulatorForOptim->updateDateIndex(0);
        Eigen::ArrayXXd particlesInit = simulatorForOptim->getParticles();
        LocalRegressionForSDDP regInit(true, particlesInit, p_meshForReg);
        archiveForRegressor << gs::Record(regInit, "Regressor", "Top");
        std::unique_ptr< SDDPVisitedStates> setOfStates = std::make_unique<SDDPVisitedStates>(1);
        std::shared_ptr< Eigen::ArrayXd > anAdmissibleState =  std::make_shared< Eigen::ArrayXd >(p_initialState);
        setOfStates->addVisitedStateForAll(anAdmissibleState, regInit);
        vecSetOfStates[0] = move(setOfStates);

        // archive initial admissible states (forward order)
        for (size_t idate = 0; idate <= vecSetOfStates.size() - 1; ++idate)
            archiveForInitialState << gs::Record(*vecSetOfStates[idate], "States", "Top");
    }

#ifdef USE_MPI
    p_world.barrier();
#endif
    int iterMax = p_iter;
    p_iter = 0;
    double accuracy = p_accuracy;
    p_accuracy = 1e10;
    // archive to read regressor : currently all processor reads the regression
    std::shared_ptr<gs::BinaryFileArchive> archiveReadRegressor = std::make_shared<gs::BinaryFileArchive>(p_nameRegressor.c_str(), "r");
    // archive for cuts read write
    std::shared_ptr<gs::BinaryFileArchive> archiveForCuts;

    // only create for first task
    if (iTask == 0)
        archiveForCuts = std::make_shared<gs::BinaryFileArchive>(p_nameCut.c_str(), "w+");

    // to store the backward value
    double backwardValue = 0.;
    // forward value
    double forwardValueForConv = 0.;
    int istep = 0;
    // store evolution of convergence
    double backwardMinusForwardPrev = 0;
    while ((accuracy < p_accuracy) && (p_iter < iterMax))
    {
        // increase step
        istep += 1;
        // local timer
        boost::timer::cpu_timer localTimer;
        //  actualize time for simulators
        simulatorForOptim->resetTime();
        simulatorForSim->resetTime();
#ifdef USE_MPI
        p_world.barrier();
#endif
        // backward sweep
        backwardValue = backwardSDDP<LocalRegressionForSDDP>(p_optimizer, simulatorForOptim, p_dates,
                        p_initialState, p_finalCut, archiveReadRegressor,
                        p_nameVisitedStates, archiveForCuts,
#ifdef USE_MPI
                        p_world,
#endif
                        false);
        localTimer.stop();
        if (p_bPrintTime && (iTask == 0))
        {
            p_outputStream << " SDDP backward  iteration " << p_iter <<  " value " << backwardValue << " time " <<  localTimer.format() <<  std::endl ;
            std::cout << " SDDP backward  iteration " << p_iter <<   " value " << backwardValue << " time " <<  localTimer.format();
            std::cout.flush();
        }

#ifdef USE_MPI
        p_world.barrier();
#endif
        // forward sweep
        bool   bIncreaseCut  = true;

        localTimer.start();

        forwardSDDP<LocalRegressionForSDDP>(p_optimizer, simulatorForSim, p_dates, p_initialState, p_finalCut, bIncreaseCut, archiveReadRegressor,
                                            archiveForCuts, p_nameVisitedStates
#ifdef USE_MPI
                                            , p_world
#endif
                                           );

        localTimer.stop();
        if (p_bPrintTime && (iTask == 0))
        {
            p_outputStream << " SDDP forward iteration " << p_iter << " time " <<  localTimer.format() << std::endl ;
        }

#ifdef USE_MPI
        p_world.barrier();
#endif
        if ((istep == p_nStepConv) || (p_iter == 0))
        {
            istep = 0;
            int oldParticleNb = simulatorForSim->getNbSimul();
            simulatorForSim->updateSimulationNumberAndResetTime(p_nbSimulCheckForSimu);
            bIncreaseCut = false;
            forwardValueForConv =  forwardSDDP<LocalRegressionForSDDP>(p_optimizer, simulatorForSim, p_dates,
                                   p_initialState, p_finalCut, bIncreaseCut, archiveReadRegressor,
                                   archiveForCuts, p_nameVisitedStates
#ifdef USE_MPI
                                   , p_world
#endif
                                                                      );
            if (forwardValueForConv != 0.0)
                p_accuracy = fabs((backwardValue - forwardValueForConv) / forwardValueForConv);
            else
                p_accuracy = fabs(backwardValue);
            simulatorForSim->updateSimulationNumberAndResetTime(oldParticleNb);
            globalTimer.stop();
            if (iTask == 0)
            {
                p_outputStream << " ACCURACY " << p_accuracy  << " Backward " << backwardValue << " Forward " << forwardValueForConv << " p_iter "  << p_iter << " accuracy " <<  p_accuracy;
                p_outputStream <<  " GlobalTimer " << globalTimer.format() << std::endl ;
                std::cout <<  " ACCURACY " << p_accuracy  << " Backward " << backwardValue << " Forward " << forwardValueForConv << " p_iter "  << p_iter << " accuracy " <<  p_accuracy << " GlobalTimer " <<  globalTimer.format() << std::endl ;
            }
            globalTimer.resume();
            double backwardMinusForward = backwardValue - forwardValueForConv;
            if (p_iter > 0)
            {
                if (backwardMinusForward * backwardMinusForwardPrev < 0)
                {
                    if (iTask == 0)
                        p_outputStream << " Curve are crossing : increase sample and simulations to get more accurate solution, decrease step for checking convergence" << " GlobalTimer " <<  globalTimer.format() << std::endl ;
                    // exit
                    p_accuracy = 0.;
                }
            }
            backwardMinusForwardPrev = backwardMinusForward;
        }
        else if (p_iter > 0)
        {
            // add a check on curve
            double backwardMinusForward = backwardValue - forwardValueForConv;
            if (backwardMinusForward * backwardMinusForwardPrev < 0)
            {
                if (iTask == 0)
                    p_outputStream << " Curve are crossing : increase sample and simulations to get more accurate solution, decrease step for checking convergence" << std::endl ;
                // exit
                p_accuracy = 0.;
            }
        }
        p_iter += 1;
    }

#ifdef USE_MPI
    p_world.barrier();
#endif
    return std::make_pair(backwardValue, forwardValueForConv);

}

}

#endif /* BACKWARDFORWARDSDDP_H */
