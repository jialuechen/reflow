
#ifndef BACKWARDSDDPTREE_H
#define BACKWARDSDDPTREE_H
#include <memory>
#ifdef USE_MPI
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#endif
#include <boost/timer/timer.hpp>
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "geners/Record.hh"
#include "geners/Reference.hh"
#include "reflow/sddp/SimulatorSDDPBaseTree.h"
#include "reflow/sddp/OptimizerSDDPBase.h"
#include "reflow/sddp/SDDPFinalCutTree.h"
#include "reflow/sddp/SDDPCutTree.h"
#include "reflow/sddp/SDDPVisitedStatesTree.h"
#include "reflow/sddp/SDDPVisitedStatesTreeGeners.h"


/** \file backwardSDDPTree.h
 * \brief One sequence of backward resolution by SDDP specialized for trees
 * \author Xavier Warin
 */
namespace reflow
{
/// \brief Realize a backward sweep for SDDP with trees
/// \param p_optimizer         object defining a transition step for SDDP
/// \param p_simulator         simulates uncertainties for regressions, inflows etc....  Explore all nodes in tree.
/// \param p_dates             vector of exercised dates, last dates correspond to the final cut object
/// \param p_initialState      initial state at the beginning of simulation
/// \param p_finalCut          object of final cuts
/// \param p_nameVisitedStates name of the archive used to store visited states
/// \param p_archiveCut        archive storing cuts generated
/// \param  p_world            MPI communicator
/// \param p_bPrintTime        if true print time at each backward and forward step
/// \return value obtained by backward resolution
double 	backwardSDDPTree(std::shared_ptr<OptimizerSDDPBase>   &p_optimizer,
                         std::shared_ptr<SimulatorSDDPBaseTree> &p_simulator,
                         const Eigen::ArrayXd   &p_dates,
                         const Eigen::ArrayXd &p_initialState,
                         const SDDPFinalCutTree &p_finalCut,
                         const std::string &p_nameVisitedStates,
                         const std::shared_ptr<gs::BinaryFileArchive> &p_archiveCut,
#ifdef USE_MPI
                         const boost::mpi::communicator &p_world,
#endif
                         bool  p_bPrintTime = false)
{
    // to red cuts
    gs::BinaryFileArchive archiveVisitedStates(p_nameVisitedStates.c_str(), "r");

    // get number of sample used in optimization part
    int nbSample = p_simulator->getNbSample();
    // final cut
    std::unique_ptr< SDDPCutBaseTree > linCutNext = std::make_unique< SDDPFinalCutTree>(p_finalCut);
    // iterate over step
    for (int idate = p_dates.size() - 2; idate > 0 ; --idate)
    {
        // local timer
        boost::timer::cpu_timer localTimer;
        // update new date for optimizer and simulator
        p_optimizer->updateDates(p_dates(idate - 1), p_dates(idate));
        // first update simulator to calculate conditional expections with tree
        p_simulator->updateDateIndex(idate - 1);
        // get probability transition between date p_dates(idate - 1) and p_dates(idate)
        std::vector<double> probabilies = p_simulator->getProba();
        // Connection between nodes between date p_dates(idate - 1) and p_dates(idate)
        std::vector<std::vector< std::array<int, 2 >  > > connectionMatrix = p_simulator-> getConnected();
        /// set of nodes in tree  at date p_dates(idate-1)
        Eigen::ArrayXXd  nodes = p_simulator->getNodes();
        // number of nodes at date p_dates(idate)
        int nbNodesNext =  p_simulator-> getNbNodesNext();
        // nodes at p_dates(idate)
        Eigen::ArrayXXd  nodesNext = p_simulator->getNodesNext();
        // get back states at current step
        std::unique_ptr<SDDPVisitedStatesTree> VisitedStates = gs::Reference< SDDPVisitedStatesTree >(archiveVisitedStates, "States", "Top").get(idate - 1);
        // create SDDP cut object at the previous date with regressor at previous date
        std::unique_ptr<SDDPCutBaseTree> linCutPrev = std::make_unique<SDDPCutTree>(idate - 1, nbSample, probabilies, connectionMatrix, nodes);
        /// load existing cuts to prepare next time step
        linCutPrev->loadCuts(p_archiveCut
#ifdef USE_MPI
                             , p_world
#endif
                            );
        // create vector of LP (one for each sample)
        std::vector< std::tuple< std::shared_ptr<Eigen::ArrayXd>, int, int >  >  vecState = linCutPrev->createVectorStatesParticle(*VisitedStates);
        // spread between processors
        int nbLPTotal = vecState.size() * nbSample;
        // now place  simulator to get  p_dates(idate) for use in optimization obejct
        p_simulator->updateDateIndex(idate);
        // now that conditional expectation object
#ifdef USE_MPI
        int nbTask = p_world.size();
        int iTask = p_world.rank();
#else
        int nbTask = 1;
        int iTask = 0;
#endif
        int nsimPProc = (int)(nbLPTotal / nbTask);
        int nRest = nbLPTotal % nbTask;
        int iLPFirst = iTask * nsimPProc + (iTask < nRest ? iTask : nRest);
        int iLPLast  = iLPFirst + nsimPProc + (iTask < nRest ? 1 : 0);
        // to store cuts ::dimension of the problem  plus one by number of simulations
        Eigen::ArrayXXd cutPerSimPerProc(p_optimizer->getStateSize() + 1, iLPLast - iLPFirst);
        int ism;
        #pragma omp parallel  for schedule(dynamic)  private(ism)
        for (ism = 0; ism < iLPLast - iLPFirst; ++ism)
        {
            // current state and particle associated
            std::tuple< std::shared_ptr<Eigen::ArrayXd>, int, int >  aState = vecState[(ism + iLPFirst) / linCutPrev->getSample()];
            // sample number
            int isample = (ism + iLPFirst) % (nbSample * nbNodesNext);
            //  call to main optimizer
            // simulator is given, cuts at the next time step, the state vector use, the node in tree  associated to this optimization
            cutPerSimPerProc.col(ism) =  p_optimizer->oneStepBackward(*static_cast<SDDPCutOptBase *>(linCutNext.get()), aState, nodesNext.col(std::get<1>(aState)), isample);
            /// now using function value  and sensibility, create the cut (derivatives already calculated)
            std::shared_ptr<Eigen::ArrayXd> stateAlone = std::get<0>(aState);
            for (int ist = 0; ist < stateAlone->size(); ++ist)
                cutPerSimPerProc(0, ism) -= cutPerSimPerProc(ist + 1, ism) * (*stateAlone)(ist);
        }
        // conditional expectation of the cuts at previous time step
        linCutPrev->createAndStoreCuts(cutPerSimPerProc, *VisitedStates, vecState, p_archiveCut
#ifdef USE_MPI
                                       , p_world
#endif
                                      );
        linCutNext = move(linCutPrev);
        if (p_bPrintTime && (iTask == 0))
        {
            std::cout << "backward  : idate " << idate << " nb LP processor 0 " << iLPLast - iLPFirst <<  " time " <<  localTimer.format() <<  std::endl ;
            std::cout.flush();
        }
    }
    // update first  for optimizer and simulator : -1 indicate non previous date
    p_optimizer->updateDates(-1, p_dates(0));
    p_simulator->updateDateIndex(0);
    /// set of nodes in tree  at date p_dates(idate)
    Eigen::ArrayXXd  nodes = p_simulator->getNodes();
    // now just only one particle for first time step adn one LP
    std::shared_ptr<Eigen::ArrayXd> ptState = std::make_shared< Eigen::ArrayXd>(p_initialState);
    std::tuple< std::shared_ptr<Eigen::ArrayXd>, int, int >  aState = make_tuple(ptState, 0, 0);
    double valueEstimation = p_optimizer->oneStepBackward(*linCutNext, aState, nodes.col(std::get<1>(aState)), 0)(0);
    return valueEstimation;
}

}
#endif
