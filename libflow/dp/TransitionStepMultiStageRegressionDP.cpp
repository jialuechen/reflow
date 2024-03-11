// Copyright (C) 2023 EDF

// This code is published under the GNU Lesser General Public License (GNU LGPL)
#include <memory>
#include <boost/lexical_cast.hpp>
#include "geners/vectorIO.hh"
#include "geners/Record.hh"
#ifdef USE_MPI
#include <boost/mpi.hpp>
#include "libflow/core/parallelism/all_gatherv.hpp"
#endif
#ifdef _OPENMP
#include <omp.h>
#include "libflow/core/utils/OpenmpException.h"
#endif
#include "libflow/core/grids/GridIterator.h"
#include "libflow/dp/TransitionStepMultiStageRegressionDP.h"
#include "libflow/dp/OptimizerMultiStageDPBase.h"
#include "libflow/regression/ContinuationValue.h"
#include "libflow/regression/ContinuationValueGeners.h"
#include "libflow/regression/GridAndRegressedValue.h"
#include "libflow/regression/GridAndRegressedValueGeners.h"


using namespace Eigen;
using namespace libflow;
using namespace std;


TransitionStepMultiStageRegressionDP::TransitionStepMultiStageRegressionDP(const  shared_ptr<FullGrid> &p_pGridCurrent,
        const  shared_ptr<FullGrid> &p_pGridPrevious,
        const  shared_ptr<OptimizerMultiStageDPBase > &p_pOptimize
#ifdef USE_MPI
        , const boost::mpi::communicator &p_world
#endif
                                                                          ):
    m_pGridCurrent(p_pGridCurrent), m_pGridPrevious(p_pGridPrevious), m_pOptimize(p_pOptimize)
#ifdef USE_MPI
    , m_world(p_world)
#endif
{
}
TransitionStepMultiStageRegressionDP::TransitionStepMultiStageRegressionDP(const  shared_ptr<FullGrid> &p_pGridCurrent,
        const  shared_ptr<FullGrid> &p_pGridPrevious,
        const  shared_ptr<OptimizerMultiStageDPBase > &p_pOptimize,
        const  std::shared_ptr<gs::BinaryFileArchive>   &p_arGen,
        const  std::string &p_nameDump
#ifdef USE_MPI
        , const boost::mpi::communicator &p_world
#endif
                                                                          ):
    m_pGridCurrent(p_pGridCurrent), m_pGridPrevious(p_pGridPrevious), m_pOptimize(p_pOptimize), m_arGen(p_arGen), m_nameDump(p_nameDump)
#ifdef USE_MPI
    , m_world(p_world)
#endif
{
}


void TransitionStepMultiStageRegressionDP::oneStageInStep(const vector< shared_ptr< ArrayXXd > > &p_phiIn,
        vector< shared_ptr< ArrayXXd > > &p_phiOut,
        vector<shared_ptr<ContinuationValue> > &p_contVal,
        const shared_ptr<FullGrid> &p_pGridCurTrans,
        const shared_ptr<FullGrid> &p_pGridPrevTrans
#
#ifdef USE_MPI
        , vector< ArrayXXd >   &p_phiOutLoc,
        ArrayXi &p_ilocToGLobal,
        ArrayXi &p_ilocToGLobalGlob,
        ArrayXXd   &p_storeGlob
#endif
                                                         ) const
{
    // number of regimes at current time
    int nbDetRegimes = m_pOptimize->getNbDetRegime();

#ifdef USE_MPI
    int  rank = m_world.rank();
    int nbProc = m_world.size();
#endif
    // number of thread
#ifdef _OPENMP
    int nbThreads = omp_get_max_threads();
#else
    int nbThreads = 1;
#endif
    // create iterator on current grid treated for processor
    int iThread = 0 ;
#ifdef _OPENMP
    OpenmpException excep; // deal with exception in openmp
    #pragma omp parallel for  private(iThread)
#endif
    for (iThread = 0; iThread < nbThreads; ++iThread)
    {
#ifdef _OPENMP
        excep.run([&]
        {
#endif
            shared_ptr< GridIterator >  iterGridPoint = p_pGridCurTrans->getGridIterator();
            // account for mpi and threads
#ifdef USE_MPI
            iterGridPoint->jumpToAndInc(rank, nbProc, iThread);
#else
            iterGridPoint->jumpToAndInc(0, 1, iThread);
#endif
            // iterates on points of the grid
            while (iterGridPoint->isValid())
            {
                ArrayXd pointCoord = iterGridPoint->getCoordinate();
                // optimize the current point and the set of regimes
                ArrayXXd  solution = m_pOptimize->stepOptimize(p_pGridPrevTrans, pointCoord, p_contVal, p_phiIn);
#ifdef USE_MPI
                // copie solution
                int iposArray = iterGridPoint->getRelativePosition();
                p_ilocToGLobal(iposArray) = iterGridPoint->getCount();
                // copie solution
                for (int iReg = 0; iReg < nbDetRegimes; ++iReg)
                    p_phiOutLoc[iReg].col(iposArray) = solution.col(iReg);

#else
                // copie solution
                for (int iReg = 0; iReg < nbDetRegimes; ++iReg)
                    p_phiOut[iReg]->col(iterGridPoint->getCount()) = solution.col(iReg);
#endif
                iterGridPoint->nextInc(nbThreads);
            }
#ifdef _OPENMP
        });
#endif
    }
#ifdef _OPENMP
    excep.rethrow();
#endif

#ifdef USE_MPI
    boost::mpi::all_gatherv<int>(m_world, p_ilocToGLobal.data(), p_ilocToGLobal.size(), p_ilocToGLobalGlob.data());
    for (int iReg = 0; iReg < nbDetRegimes; ++iReg)
    {
        boost::mpi::all_gatherv<double>(m_world, p_phiOutLoc[iReg].data(), p_phiOutLoc[iReg].size(), p_storeGlob.data());
        for (int ipos = 0; ipos < p_ilocToGLobalGlob.size(); ++ipos)
            (*p_phiOut[iReg]).col(p_ilocToGLobalGlob(ipos)) = p_storeGlob.col(ipos);
    }
#endif

}


vector< shared_ptr< ArrayXXd > >  TransitionStepMultiStageRegressionDP::oneStep(const vector< shared_ptr< ArrayXXd > > &p_phiIn,
        const shared_ptr< BaseRegression>     &p_condExp) const
{
    // number of regimes at current time
    int nbRegimes = m_pOptimize->getNbRegime();
    // number of deterministic regimes
    int nbDetRegimes =  m_pOptimize->getNbDetRegime();
    //  check in debug the coherence
    assert(nbDetRegimes >= nbRegimes);
    vector< shared_ptr< ArrayXXd > >  phiOut(nbDetRegimes);

#ifdef USE_MPI
    int  rank = m_world.rank();
    int nbProc = m_world.size();
    //  allocate for solution
    int nbPointsCur = m_pGridCurrent->getNbPoints();
    int npointPProcCur = (int)(nbPointsCur / nbProc);
    int nRestPointCur = nbPointsCur % nbProc;
    int iFirstPointCur = rank * npointPProcCur + (rank < nRestPointCur ? rank : nRestPointCur);
    int iLastPointCur  = iFirstPointCur + npointPProcCur + (rank < nRestPointCur ? 1 : 0);
    vector< ArrayXXd> phiOutLoc(nbDetRegimes);
    ArrayXi ilocToGLobal(iLastPointCur - iFirstPointCur);
    for (int iReg = 0; iReg < nbDetRegimes; ++iReg)
        phiOutLoc[iReg].resize(p_condExp->getNbSimul(), iLastPointCur - iFirstPointCur);
    ArrayXi ilocToGLobalGlob(nbPointsCur);
    ArrayXXd storeGlob(p_condExp->getNbSimul(), nbPointsCur);

#endif

    //  allocate for solution
    for (int  iReg = 0; iReg < nbDetRegimes; ++iReg)
        phiOut[iReg] = make_shared< ArrayXXd >(p_condExp->getNbSimul(), m_pGridCurrent->getNbPoints());

    shared_ptr< SimulatorMultiStageDPBase > simulator = m_pOptimize->getSimulator();

    int  nbPeriodsOfCurrentStep = simulator->getNbPeriodsInTransition();

    //  create continuation for last period in stochastic
    vector< shared_ptr<ContinuationValue> > contVal(p_phiIn.size());
    for (size_t iReg = 0; iReg < static_cast<size_t>(nbRegimes); ++iReg)
        contVal[iReg] = make_shared<ContinuationValue>(m_pGridPrevious, p_condExp, *p_phiIn[iReg]);

    // set period number in simulator
    simulator->setPeriodInTransition(nbPeriodsOfCurrentStep - 1);

    // optimize for current step
    oneStageInStep(p_phiIn, phiOut, contVal, m_pGridCurrent, m_pGridPrevious
#ifdef USE_MPI

                   , phiOutLoc, ilocToGLobal, ilocToGLobalGlob, storeGlob
#endif
                  );

    // now iterate on deterministic period
    for (int iPeriod = nbPeriodsOfCurrentStep - 2; iPeriod >= 0; iPeriod--)
    {
        vector< shared_ptr<ContinuationValue> > contValDet(nbDetRegimes);
        for (size_t iReg = 0; iReg < static_cast<size_t>(nbDetRegimes); ++iReg)
            contValDet[iReg] = make_shared<ContinuationValue>(m_pGridCurrent, p_condExp, *phiOut[iReg]);

        // dump if necessary
        if (m_arGen)
        {
#ifdef USE_MPI
            if (rank == 0)
            {
#endif
                // store as an interpolator (interpolate on trajectories values)
                vector< GridAndRegressedValue > bellInterpolator(nbDetRegimes);
                for (size_t iReg = 0; iReg < static_cast<size_t>(nbDetRegimes); ++iReg)
                    bellInterpolator[iReg] = GridAndRegressedValue(m_pGridCurrent, p_condExp, *phiOut[iReg]);
                *m_arGen << gs::Record(bellInterpolator, m_nameDump.c_str(), boost::lexical_cast<string>(iPeriod).c_str());
#ifdef USE_MPI
            }
#endif
        }
        // local
        vector< shared_ptr< ArrayXXd > >  phiInLoc(nbDetRegimes);
        for (int iReg = 0; iReg < nbDetRegimes; ++iReg)
            phiInLoc[iReg] = make_shared< ArrayXXd >(*phiOut[iReg]);

        // set period number in simulator
        simulator->setPeriodInTransition(iPeriod);

        // optimize for current step
        oneStageInStep(phiInLoc, phiOut, contValDet, m_pGridCurrent, m_pGridCurrent
#ifdef USE_MPI

                       , phiOutLoc, ilocToGLobal, ilocToGLobalGlob, storeGlob
#endif
                      );
    }
    // if the number of determinist regime strictly above , juste select the good number
    if (nbDetRegimes > nbRegimes)
    {
        vector< shared_ptr< ArrayXXd > >  phiOutRed(nbRegimes);
        for (int iReg = 0; iReg < nbRegimes; ++iReg)
            phiOutRed[iReg] =  phiOut[iReg];
        return phiOutRed;
    }
    else
    {
        return phiOut;
    }
}

void TransitionStepMultiStageRegressionDP::dumpContinuationValues(shared_ptr<gs::BinaryFileArchive> p_ar, const string &p_name, const int &p_iStep, const vector< shared_ptr< ArrayXXd > > &p_phiIn, const  shared_ptr<BaseRegression>    &p_condExp) const
{
#ifdef USE_MPI
    if (m_world.rank() == 0)
    {
#endif
        vector< GridAndRegressedValue > contVal(p_phiIn.size());
        for (size_t iReg = 0; iReg < p_phiIn.size(); ++iReg)
            contVal[iReg] = GridAndRegressedValue(m_pGridPrevious, p_condExp, *p_phiIn[iReg]);
        string stepString = boost::lexical_cast<string>(p_iStep) ;
        *p_ar << gs::Record(contVal, (p_name + "Values").c_str(), stepString.c_str()) ;
        p_ar->flush() ; // necessary for python mapping
#ifdef USE_MPI
    }
#endif
}

void TransitionStepMultiStageRegressionDP::dumpBellmanValues(shared_ptr<gs::BinaryFileArchive> p_ar, const string &p_name, const int &p_iStep, const vector< shared_ptr< ArrayXXd > > &p_phiIn,
        const  shared_ptr<BaseRegression>    &p_condExp) const
{
#ifdef USE_MPI
    if (m_world.rank() == 0)
    {
#endif
        vector< GridAndRegressedValue > bellVal(p_phiIn.size());
        for (size_t iReg = 0; iReg < p_phiIn.size(); ++iReg)
            bellVal[iReg] = GridAndRegressedValue(m_pGridCurrent, p_condExp, *p_phiIn[iReg]);
        string stepString = boost::lexical_cast<string>(p_iStep) ;
        *p_ar << gs::Record(bellVal, (p_name + "Values").c_str(), stepString.c_str()) ;
        p_ar->flush() ; // necessary for python mapping
#ifdef USE_MPI
    }
#endif
}

