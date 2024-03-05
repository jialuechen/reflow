
#include <memory>
#include "geners/vectorIO.hh"
#include "geners/Record.hh"
#ifdef USE_MPI
#include "boost/mpi.hpp"
#include "libflow/core/parallelism/all_gatherv.hpp"
#endif
#ifdef _OPENMP
#include <omp.h>
#include "libflow/core/utils/OpenmpException.h"
#endif
#include "libflow/dp/TransitionStepTreeDPCut.h"
#include "libflow/core/grids/GridIterator.h"
#include "libflow/tree/ContinuationCutsTree.h"
#include "libflow/tree/ContinuationCutsTreeGeners.h"
#include "libflow/tree/GridTreeValue.h"
#include "libflow/tree/GridTreeValueGeners.h"


using namespace Eigen;
using namespace libflow;
using namespace std;


TransitionStepTreeDPCut::TransitionStepTreeDPCut(const  shared_ptr<FullGrid> &p_pGridCurrent,
        const  shared_ptr<FullGrid> &p_pGridPrevious,
        const  shared_ptr<OptimizerDPCutTreeBase > &p_pOptimize
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

vector< shared_ptr< ArrayXXd > >  TransitionStepTreeDPCut::oneStep(const vector< shared_ptr< ArrayXXd > > &p_phiIn,
        const shared_ptr< Tree>     &p_condExp) const
{
    // number of regimes at current time
    int nbRegimes = m_pOptimize->getNbRegime();
    vector< shared_ptr< ArrayXXd > >   phiOut(nbRegimes);
    // only if the processor is working
    if (m_pGridCurrent->getNbPoints() > 0)
    {

#ifdef USE_MPI
        int  rank = m_world.rank();
        int nbProc = m_world.size();
        //  allocate for solution
        int nbPointsCur = m_pGridCurrent->getNbPoints();
        int npointPProcCur = (int)(nbPointsCur / nbProc);
        int nRestPointCur = nbPointsCur % nbProc;
        int iFirstPointCur = rank * npointPProcCur + (rank < nRestPointCur ? rank : nRestPointCur);
        int iLastPointCur  = iFirstPointCur + npointPProcCur + (rank < nRestPointCur ? 1 : 0);
        std::vector< ArrayXXd> phiOutLoc(nbRegimes);
        ArrayXi ilocToGLobal(iLastPointCur - iFirstPointCur);
        for (int iReg = 0; iReg < nbRegimes; ++iReg)
            phiOutLoc[iReg].resize(p_condExp->getNbNodes() * (m_pGridCurrent->getDimension() + 1), iLastPointCur - iFirstPointCur);

#endif

        //  allocate for solution
        for (int  iReg = 0; iReg < nbRegimes; ++iReg)
            phiOut[iReg] = make_shared< ArrayXXd >(p_condExp->getNbNodes() * (m_pGridCurrent->getDimension() + 1), m_pGridCurrent->getNbPoints());

        // number of thread
#ifdef _OPENMP
        int nbThreads = omp_get_max_threads();
#else
        int nbThreads = 1;
#endif
        //  create continuation values
        vector< ContinuationCutsTree > contVal(p_phiIn.size());
        for (size_t iReg = 0; iReg < p_phiIn.size(); ++iReg)
            contVal[iReg] = ContinuationCutsTree(m_pGridPrevious, p_condExp, *p_phiIn[iReg]);

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
                shared_ptr< GridIterator >  iterGridPoint = m_pGridCurrent->getGridIterator();
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
                    // optimize the current point and the set of regimes -> get back cuts per simulation and stock point
                    ArrayXXd   solution = m_pOptimize->stepOptimize(m_pGridPrevious, pointCoord, contVal);
#ifdef USE_MPI
                    // copie solution
                    int iposArray = iterGridPoint->getRelativePosition();
                    ilocToGLobal(iposArray) = iterGridPoint->getCount();
                    // copie solution
                    for (int iReg = 0; iReg < nbRegimes; ++iReg)
                        phiOutLoc[iReg].col(iposArray) = solution.col(iReg);
#else
                    // copie solution
                    for (int iReg = 0; iReg < nbRegimes; ++iReg)
                        (*phiOut[iReg]).col(iterGridPoint->getCount()) = solution.col(iReg);

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
        ArrayXi ilocToGLobalGlob(nbPointsCur);
        boost::mpi::all_gatherv<int>(m_world, ilocToGLobal.data(), ilocToGLobal.size(), ilocToGLobalGlob.data());
        ArrayXXd storeGlob(p_condExp->getNbNodes() * (m_pGridCurrent->getDimension() + 1), nbPointsCur);
        for (int iReg = 0; iReg < nbRegimes; ++iReg)
        {
            boost::mpi::all_gatherv<double>(m_world, phiOutLoc[iReg].data(), phiOutLoc[iReg].size(), storeGlob.data());
            for (int ipos = 0; ipos < ilocToGLobalGlob.size(); ++ipos)
                (*phiOut[iReg]).col(ilocToGLobalGlob(ipos)) = storeGlob.col(ipos);
        }
#endif

    }
    return phiOut;
}

void TransitionStepTreeDPCut::dumpContinuationCutsValues(std::shared_ptr<gs::BinaryFileArchive> p_ar, const string &p_name, const int &p_iStep, const vector< shared_ptr< ArrayXXd > > &p_phiIn, const shared_ptr< Tree>     &p_condExp) const
{
#ifdef USE_MPI
    if (m_world.rank() == 0)
    {
#endif
        vector< ContinuationCutsTree > contVal(p_phiIn.size());
        for (size_t iReg = 0; iReg < p_phiIn.size(); ++iReg)
        {
            contVal[iReg] =  ContinuationCutsTree(m_pGridPrevious, p_condExp,  *p_phiIn[iReg]);
        }
        string stepString = boost::lexical_cast<string>(p_iStep) ;
        *p_ar << gs::Record(contVal, (p_name + "Values").c_str(), stepString.c_str()) ;
        p_ar->flush() ; // necessary for python mapping
#ifdef USE_MPI
    }
#endif
}

