
#ifndef FORWARDSDDDPTREE_H
#define FORWARDSDDDPTREE_H
#include <memory>
#include "geners/BinaryFileArchive.hh"
#include "geners/Record.hh"
#include "geners/Reference.hh"
#include "reflow/sddp/SDDPFinalCutTree.h"
#include "reflow/sddp/SDDPCutBaseTree.h"
#include "reflow/sddp/SDDPVisitedStatesTreeGeners.h"
#include "reflow/sddp/SimulatorSDDPBaseTree.h"
#include "reflow/sddp/SDDPCutTree.h"

/** \file forwardSDDPTree.h
 * \brief On sequence of forward resolution by SDDP specialized with tree
 * \author Xavier Warin
 */

namespace reflow
{
/// \brief Achieve  a forward sweep for SDDP with trees
/// \param p_optimizer              object defining a transition step for SDDP
/// \param p_simulator              simulates uncertainties for regressions, inflows etc....
/// \param p_dates                  vector of exercised dates
/// \param p_initialState           initial state at the beginning of simulation
/// \param p_finalCut               storing final cuts
/// \param p_archiveCutToRead       archive storing cuts visited
/// \param p_nameVisitedStates      name of the archive used to store   visited states
/// \param p_bIncreaseCut           true if this simulation part create visited state for cut
/// \param  p_world            MPI communicator
/// \return value obtained with this simulations
double	forwardSDDPTree(std::shared_ptr<OptimizerSDDPBase>    &p_optimizer,
                        std::shared_ptr<SimulatorSDDPBaseTree>  &p_simulator,
                        const Eigen::ArrayXd &p_dates,
                        const Eigen::ArrayXd &p_initialState,
                        const SDDPFinalCutTree &p_finalCut,
                        const bool   &p_bIncreaseCut,
                        const std::shared_ptr<gs::BinaryFileArchive> &p_archiveCutToRead,
                        const std::string &p_nameVisitedStates
#ifdef USE_MPI
                        , const boost::mpi::communicator &p_world
#endif
                      )
{

    // to store cuts
    std::unique_ptr<gs::BinaryFileArchive> arVisitedStates;
#ifdef USE_MPI
    int nbTask = p_world.size();
    int iTask = p_world.rank();
#else
    int iTask = 0;
    int nbTask = 1 ;
#endif
    if ((iTask == 0) && (p_bIncreaseCut))
        arVisitedStates = std::make_unique<gs::BinaryFileArchive>(p_nameVisitedStates.c_str(), "w");

    int nsimPProc = (int)(p_simulator->getNbSimul() / nbTask);
    int nRest = p_simulator->getNbSimul()  % nbTask;
    int iLPFirst = iTask * nsimPProc + (iTask < nRest ? iTask : nRest);
    int iLPLast  = iLPFirst + nsimPProc + (iTask < nRest ? 1 : 0);
    // to store states visited
    Eigen::ArrayXXd statePrev(p_initialState.size(), iLPLast - iLPFirst);
    for (int is = 0; is < iLPLast - iLPFirst; ++is)
        statePrev.col(is) = p_initialState ;
    // to store gain
    double gainAccumulator = 0;
    for (int idate = 0; idate < p_dates.size() - 1; ++idate)
    {
        // update new date
        p_optimizer->updateDates(p_dates(idate), p_dates(idate + 1));

        // create SDPPCut object
        std::unique_ptr< SDDPCutBaseTree > linCut;
        if (idate <  p_dates.size() - 2)
        {
            /// set of nodes in tree  at date p_dates(idate)
            Eigen::ArrayXXd  nodes = p_simulator->getNodes();

            linCut  = std::make_unique< SDDPCutTree>(idate, nodes);
        }
        else
            linCut  = std::make_unique< SDDPFinalCutTree>(p_finalCut);

        // load cuts
        linCut->loadCuts(p_archiveCutToRead
#ifdef USE_MPI
                         , p_world
#endif
                        );

        // to store visited states
        SDDPVisitedStatesTree setOfStates(p_simulator->getNbNodes());

        int isim;
        #pragma omp parallel  for schedule(dynamic)  private(isim) reduction(+:gainAccumulator)
        for (isim = 0; isim < iLPLast - iLPFirst; ++isim)
        {
            // simulation number according to simulator
            int isimul = (iLPFirst + isim);
            // get back particle associated
            Eigen::ArrayXd aParticle(p_simulator->getOneParticle(isimul));

            std::shared_ptr<Eigen::ArrayXd > newState = std::make_shared<Eigen::ArrayXd>(statePrev.col(isim)) ;
            std::shared_ptr<Eigen::ArrayXd > newStateToStore = std::make_shared< Eigen::ArrayXd>(statePrev.col(isim)) ;
            double gainOrCost = p_optimizer->oneStepForward(aParticle, *newState, *newStateToStore, *static_cast<SDDPCutOptBase *>(linCut.get()), isimul);
            // accumulate gain
            gainAccumulator += gainOrCost;

            // add state and associated node in tree : here this is the state (for storage) reached at following date
            // associated to the node of uncertainty associated at current date
            if (p_bIncreaseCut)
            {
                setOfStates.addVisitedState(newStateToStore, p_simulator->getNodeAssociatedToSim(isimul));
            }
            // update state
            statePrev.col(isim) = *newState;
        }
        // step forward  for simulator : update positions  in tree
        p_simulator->stepForward();

        // store if root
#ifdef USE_MPI
        if (idate <  p_dates.size() - 2)
            if (p_bIncreaseCut)
                if (p_world.size() > 1)
                    setOfStates.sendToRoot(p_world);
        if (iTask == 0)
#endif
            if (p_bIncreaseCut)
                *arVisitedStates << gs::Record(setOfStates, "States", "Top");
    }
#ifdef USE_MPI
    gainAccumulator = boost::mpi::all_reduce(p_world, gainAccumulator, std::plus<double>());
#endif
    return gainAccumulator / p_simulator->getNbSimul() ;
}
}
#endif
