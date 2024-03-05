
#ifndef BACKWARDFORWARDSDDPTREE_H
#define BACKWARDFORWARDSDDPTREE_H
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <map>
#include <memory>
#include <boost/timer/timer.hpp>
#include <Eigen/Dense>
#include "geners/Record.hh"
#include "libflow/sddp/OptimizerSDDPBase.h"
#include "libflow/sddp/backwardSDDPTree.h"
#include "libflow/sddp/forwardSDDPTree.h"
#include "libflow/sddp/SDDPFinalCutTree.h"
#include "libflow/sddp/SDDPVisitedStatesTreeGeners.h"

/** \file backwardForwardSDDPTree.h
 * \brief alternates forward and backward resolution with a description of the non convex/concave uncertainties with tree
 *        Conditional expectation are then calculated with the tree
 * \author Xavier Warin
 */
namespace libflow
{


/// \brief Achieve forward and backward sweep by SDDP with tree
/// \param  p_optimizer           defines the optimiser necessary to optimize a step for one simulation solving a LP
/// \param  p_nbSimulCheckForSimu defines the number of simulations to check convergence
/// \param  p_initialState        initial state at the beginning of simulation
/// \param  p_finalCut            object of final cuts
/// \param  p_dates               vector of exercised dates, last date corresponds to the final cut object
/// \param  p_nameCut             name of the archive to store cuts
/// \param  p_nameVisitedStates   name of the archive to store visited states
/// \param  p_iter                maximum iteration of SDDP, on return the number of iterations achieved
/// \param  p_accuracy            accuracy asked , on return estimation of accuracy achieved (expressed in %)
/// \param  p_nStepConv           every p_nStepConv convergence is checked
/// \param  p_stringStream        dump all print messages
/// \param  p_world            MPI communicator
/// \param  p_bPrintTime          if true print time at each backward and forward step
/// \return backward and forward valorization
std::pair<double, double> backwardForwardSDDPTree(std::shared_ptr<OptimizerSDDPBase>    &p_optimizer,
        const int   &p_nbSimulCheckForSimu,
        const Eigen::ArrayXd &p_initialState,
        const SDDPFinalCutTree &p_finalCut,
        const Eigen::ArrayXd &p_dates,
        const std::string &p_nameCut,
        const std::string &p_nameVisitedStates,
        int &p_iter,
        double &p_accuracy,
        const int &p_nStepConv,
        std::ostringstream &p_stringStream,
#ifdef USE_MPI
        const boost::mpi::communicator &p_world,
#endif
        bool  p_bPrintTime = false)
{
    // get back simulators
    std::shared_ptr<SimulatorSDDPBaseTree> simulatorForOptim =  std::static_pointer_cast<SimulatorSDDPBaseTree>(p_optimizer->getSimulatorBackward());
    std::shared_ptr<SimulatorSDDPBaseTree> simulatorForSim = std::static_pointer_cast<SimulatorSDDPBaseTree>(p_optimizer->getSimulatorForward());

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
        // create a first set of admissible states
        gs::BinaryFileArchive archiveForInitialState(p_nameVisitedStates.c_str(), "w");

        // vector of states
        std::vector< std::unique_ptr< SDDPVisitedStatesTree > > vecSetOfStates(p_dates.size() - 1);
        for (int idate = p_dates.size() - 2; idate >  0; --idate)
        {
            simulatorForOptim->updateDateIndex(idate);

            std::unique_ptr< SDDPVisitedStatesTree> setOfStates = std::make_unique<SDDPVisitedStatesTree>(simulatorForOptim->getNbNodes());
            // offset  in admissible dates
            std::shared_ptr< Eigen::ArrayXd > anAdmissibleState = std::make_shared<Eigen::ArrayXd>(p_optimizer->oneAdmissibleState(p_dates(idate)));
            setOfStates->addVisitedStateForAll(anAdmissibleState, simulatorForOptim->getNbNodes());
            vecSetOfStates[idate] = move(setOfStates);
        }
        simulatorForOptim->updateDateIndex(0);
        std::unique_ptr< SDDPVisitedStatesTree> setOfStates = std::make_unique<SDDPVisitedStatesTree>(1);
        std::shared_ptr< Eigen::ArrayXd > anAdmissibleState =  std::make_shared< Eigen::ArrayXd >(p_initialState);
        setOfStates->addVisitedStateForAll(anAdmissibleState, simulatorForOptim->getNbNodes());
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
    // archive for cuts read write
    std::shared_ptr<gs::BinaryFileArchive> archiveForCuts;

    // only create for first task
    if (iTask == 0)
        archiveForCuts = std::make_shared<gs::BinaryFileArchive>(p_nameCut.c_str(), "w+");

    // to store  all backward values
    std::vector<double> backwardValues(iterMax);
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

#ifdef USE_MPI
        p_world.barrier();
#endif
        // backward sweep
        backwardValues[p_iter] = backwardSDDPTree(p_optimizer, simulatorForOptim, p_dates,
                                 p_initialState, p_finalCut,
                                 p_nameVisitedStates, archiveForCuts, p_world, false);
        localTimer.stop();
        if (p_bPrintTime && (iTask == 0))
        {
            p_stringStream << " SDDP backward  iteration " << p_iter <<  " value " << backwardValues[p_iter] << " time " <<  localTimer.format() <<  std::endl ;
            std::cout << " SDDP backward  iteration " << p_iter <<   " value " << backwardValues[p_iter] << " time " <<  localTimer.format();
            std::cout.flush();
        }

#ifdef USE_MPI
        p_world.barrier();
#endif
        // forward sweep
        bool   bIncreaseCut  = true;

        localTimer.start();

        // reset forward simulator
        simulatorForSim->resetTime();

        forwardSDDPTree(p_optimizer, simulatorForSim, p_dates, p_initialState, p_finalCut, bIncreaseCut, archiveForCuts, p_nameVisitedStates, p_world);

        localTimer.stop();
        if (p_bPrintTime && (iTask == 0))
        {
            p_stringStream << " SDDP forward iteration " << p_iter << " time " <<  localTimer.format() << std::endl ;
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
            forwardValueForConv =  forwardSDDPTree(p_optimizer, simulatorForSim, p_dates, p_initialState, p_finalCut, bIncreaseCut,
                                                   archiveForCuts, p_nameVisitedStates, p_world);
            p_accuracy  = fabs((backwardValues[p_iter] - forwardValueForConv) / forwardValueForConv);
            simulatorForSim->updateSimulationNumberAndResetTime(oldParticleNb);
            globalTimer.stop();
            if (iTask == 0)
            {
                p_stringStream << " ACCURACY " << p_accuracy  << " Backward " << backwardValues[p_iter] << " Forward " << forwardValueForConv << " p_iter "  << p_iter << " accuracy " <<  p_accuracy;
                p_stringStream <<  " GlobalTimer " << globalTimer.format() << std::endl ;
                std::cout <<  " ACCURACY " << p_accuracy  << " Backward " << backwardValues[p_iter] << " Forward " << forwardValueForConv << " p_iter "  << p_iter << " accuracy " <<  p_accuracy << " GlobalTimer " <<  globalTimer.format() << std::endl ;
            }
            globalTimer.resume();
            double backwardMinusForward = backwardValues[p_iter] - forwardValueForConv;
            if (p_iter > 0)
            {
                if (backwardMinusForward * backwardMinusForwardPrev < 0)
                {
                    if (iTask == 0)
                        p_stringStream << " Curve are crossing : increase sample and simulations to get more accurate solution, decrease step for checking convergence" << " GlobalTimer " <<  globalTimer.format() << std::endl ;
                    // exit
                    p_accuracy = 0.;
                }
            }
            backwardMinusForwardPrev = backwardMinusForward;
        }
        else if (p_iter > 0)
        {
            // add a check on curve
            double backwardMinusForward = backwardValues[p_iter] - forwardValueForConv;
            if (backwardMinusForward * backwardMinusForwardPrev < 0)
            {
                if (iTask == 0)
                    p_stringStream << " Curve are crossing : increase sample and simulations to get more accurate solution, decrease step for checking convergence" << std::endl ;
                // exit
                p_accuracy = 0.;
            }
        }
        p_iter += 1;
    }

#ifdef USE_MPI
    p_world.barrier();
#endif
    return std::make_pair(backwardValues[p_iter - 1], forwardValueForConv);

}

}

#endif /* BACKWARDFORWARDSDDPTREE_H */
