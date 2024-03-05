
#ifndef BACKWARDSDDP_H
#define BACKWARDSDDP_H
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
#include "libflow/sddp/SimulatorSDDPBase.h"
#include "libflow/sddp/OptimizerSDDPBase.h"
#include "libflow/sddp/SDDPFinalCut.h"
#include "libflow/sddp/SDDPLocalCut.h"
#include "libflow/sddp/SDDPVisitedStates.h"
#include "libflow/sddp/SDDPVisitedStatesGeners.h"


/** \file backwardSDDP.h
 * \brief One sequence of backward resolution by SDDP with regressor
 * \author Xavier Warin
 */
namespace libflow
{
/// \brief Realize a backward sweep for SDDP
/// \param p_optimizer        object defining a transition step for SDDP
/// \param p_simulator         simulates uncertainties for regressions, inflows etc.... In this part, simulations are the same between iterations
/// \param p_dates             vector of exercised dates, last dates correspond to the final cut object
/// \param p_initialState      initial state at the beginning of simulation
/// \param p_finalCut          object of final cuts
/// \param p_archiveRegresssor archive with regressor objects
/// \param p_nameVisitedStates name of the archive used to store visited states
/// \param p_archiveCut        archive storing cuts generated
/// \param  p_world            MPI communicator
/// \param  p_bPrintTime       if true print time at each backward and forward step
/// \return value obtained by backward resolution
template<  class LocalRegressionForSDDP>
double 	backwardSDDP(const std::shared_ptr<OptimizerSDDPBase> &p_optimizer,
                     const std::shared_ptr<SimulatorSDDPBase> &p_simulator,
                     const Eigen::ArrayXd   &p_dates,
                     const Eigen::ArrayXd &p_initialState,
                     const SDDPFinalCut &p_finalCut,
                     const std::shared_ptr<gs::BinaryFileArchive> &p_archiveRegresssor,
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
    // get number of simulations used for regressions
    int nbSimul =  p_simulator->getNbSimul();
    // final cut
    std::unique_ptr< SDDPCutBase > linCutNext = std::make_unique< SDDPFinalCut>(p_finalCut);
    // regressor at previous time step
    std::shared_ptr<LocalRegressionForSDDP> regressorNext(gs::Reference< LocalRegressionForSDDP >(*p_archiveRegresssor, "Regressor", "Top").get(0));
    // iterate over step
    for (int idate = p_dates.size() - 2; idate > 0 ; --idate)
    {
        // local timer
        boost::timer::cpu_timer localTimer;

        // update new date for optimizer and simulator
        p_optimizer->updateDates(p_dates(idate - 1), p_dates(idate));
        p_simulator->updateDateIndex(idate);

        // get back states at current step
        std::unique_ptr<SDDPVisitedStates> VisitedStates = gs::Reference< SDDPVisitedStates >(archiveVisitedStates, "States", "Top").get(idate - 1);
        // get back regressor at previous time step
        std::shared_ptr<LocalRegressionForSDDP> regressorPrev(gs::Reference< LocalRegressionForSDDP >(*p_archiveRegresssor, "Regressor", "Top").get(p_dates.size() - 1 - idate));

        // create SDDP cut object at the previous date with regressor at previous date
        std::unique_ptr<SDDPCutBase> linCutPrev = std::make_unique<SDDPLocalCut>(idate - 1, nbSample, regressorPrev);
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
            // int isample = (ism + iLPFirst) % linCutPrev->getSample();
            int isample = (ism + iLPFirst) % (nbSample * nbSimul);
            //  call to main optimizer
            // simulator is given, cuts at the next time step, the state vector use, the particle associated to this optimization
            cutPerSimPerProc.col(ism) =  p_optimizer->oneStepBackward(*static_cast<SDDPCutOptBase *>(linCutNext.get()), aState, regressorNext->getParticle(std::get<1>(aState)), isample);
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
        // swap pointer
        regressorNext	= move(regressorPrev);
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
    // now just only one particle for first time step adn one LP
    std::shared_ptr<Eigen::ArrayXd> ptState = std::make_shared< Eigen::ArrayXd>(p_initialState);
    std::tuple< std::shared_ptr<Eigen::ArrayXd>, int, int >  aState = make_tuple(ptState, 0, 0);
    double valueEstimation = p_optimizer->oneStepBackward(*linCutNext, aState, regressorNext->getParticle(std::get<1>(aState)), 0)(0);
    return valueEstimation;
}

}
#endif
