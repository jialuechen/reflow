
#include <memory>
#include <boost/lexical_cast.hpp>
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
#include "libflow/core/grids/GridIterator.h"
#include "libflow/dp/TransitionStepTreeDP.h"
#include "libflow/tree/ContinuationValueTree.h"
#include "libflow/tree/ContinuationValueTreeGeners.h"
#include "libflow/tree/GridTreeValue.h"
#include "libflow/tree/GridTreeValueGeners.h"


using namespace Eigen;
using namespace libflow;
using namespace std;


TransitionStepTreeDP::TransitionStepTreeDP(const  shared_ptr<FullGrid> &p_pGridCurrent,
        const  shared_ptr<FullGrid> &p_pGridPrevious,
        const  shared_ptr<OptimizerDPTreeBase > &p_pOptimize
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

std::pair< vector< shared_ptr< ArrayXXd > >, vector<  shared_ptr< ArrayXXd > > > TransitionStepTreeDP::oneStep(const vector< shared_ptr< ArrayXXd > > &p_phiIn,
        const shared_ptr< Tree>     &p_condExp) const
{
    // number of regimes at current time
    int nbRegimes = m_pOptimize->getNbRegime();
    vector< shared_ptr< ArrayXXd > >  phiOut(nbRegimes);
    int nbControl =  m_pOptimize->getNbControl();
    vector< shared_ptr< ArrayXXd > >  controlOut(nbControl);
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
        std::vector< ArrayXXd> phiOutLoc(nbRegimes), controlOutLoc(nbControl);
        ArrayXi ilocToGLobal(iLastPointCur - iFirstPointCur);
        for (int iReg = 0; iReg < nbRegimes; ++iReg)
            phiOutLoc[iReg].resize(p_condExp->getNbNodes(), iLastPointCur - iFirstPointCur);
        for (int iCont = 0; iCont < nbControl; ++iCont)
            controlOutLoc[iCont].resize(p_condExp->getNbNodes(), iLastPointCur - iFirstPointCur);
#endif

        //  allocate for solution
        for (int  iReg = 0; iReg < nbRegimes; ++iReg)
            phiOut[iReg] = make_shared< ArrayXXd >(p_condExp->getNbNodes(), m_pGridCurrent->getNbPoints());
        for (int iCont = 0; iCont < nbControl; ++iCont)
            controlOut[iCont] = make_shared< ArrayXXd >(p_condExp->getNbNodes(), m_pGridCurrent->getNbPoints());

        // number of thread
#ifdef _OPENMP
        int nbThreads = omp_get_max_threads();
#else
        int nbThreads = 1;
#endif
        //  create continuation values
        vector< ContinuationValueTree > contVal(p_phiIn.size());
        for (size_t iReg = 0; iReg < p_phiIn.size(); ++iReg)
        {
            contVal[iReg] = ContinuationValueTree(m_pGridPrevious, p_condExp, *p_phiIn[iReg]);
        }

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
                    // optimize the current point and the set of regimes
                    std::pair< ArrayXXd, ArrayXXd>  solutionAndControl = m_pOptimize->stepOptimize(m_pGridPrevious, pointCoord, contVal);
#ifdef USE_MPI
                    // copie solution
                    int iposArray = iterGridPoint->getRelativePosition();
                    ilocToGLobal(iposArray) = iterGridPoint->getCount();
                    // copie solution
                    for (int iReg = 0; iReg < nbRegimes; ++iReg)
                        phiOutLoc[iReg].col(iposArray) = solutionAndControl.first.col(iReg);
                    for (int iCont = 0; iCont < nbControl; ++iCont)
                        controlOutLoc[iCont].col(iposArray) = solutionAndControl.second.col(iCont);
#else
                    // copie solution
                    for (int iReg = 0; iReg < nbRegimes; ++iReg)
                        (*phiOut[iReg]).col(iterGridPoint->getCount()) = solutionAndControl.first.col(iReg);
                    for (int iCont = 0; iCont < nbControl; ++iCont)
                        (*controlOut[iCont]).col(iterGridPoint->getCount()) = solutionAndControl.second.col(iCont);
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
        ArrayXXd storeGlob(p_condExp->getNbNodes(), nbPointsCur);
        for (int iReg = 0; iReg < nbRegimes; ++iReg)
        {
            boost::mpi::all_gatherv<double>(m_world, phiOutLoc[iReg].data(), phiOutLoc[iReg].size(), storeGlob.data());
            for (int ipos = 0; ipos < ilocToGLobalGlob.size(); ++ipos)
                (*phiOut[iReg]).col(ilocToGLobalGlob(ipos)) = storeGlob.col(ipos);
        }
        for (int iCont = 0 ; iCont < nbControl; ++iCont)
        {
            boost::mpi::all_gatherv<double>(m_world, controlOutLoc[iCont].data(), controlOutLoc[iCont].size(), storeGlob.data());
            for (int ipos = 0; ipos <  ilocToGLobalGlob.size(); ++ipos)
                (*controlOut[iCont]).col(ilocToGLobalGlob(ipos)) = storeGlob.col(ipos);
        }
#endif

    }
    return make_pair(phiOut, controlOut);
}

void TransitionStepTreeDP::dumpContinuationValues(std::shared_ptr<gs::BinaryFileArchive> p_ar, const string &p_name,
        const int &p_iStep, const vector< shared_ptr< ArrayXXd > > &p_phiIn,
        const vector< shared_ptr< ArrayXXd > > &p_control,
        const std::shared_ptr< Tree>     &p_tree) const
{
#ifdef USE_MPI
    if (m_world.rank() == 0)
    {
#endif
        vector< GridTreeValue > contVal(p_phiIn.size());
        for (size_t iReg = 0; iReg < p_phiIn.size(); ++iReg)
        {
            ArrayXXd  valExp = p_tree->expCondMultiple(p_phiIn[iReg]->transpose()).transpose();
            contVal[iReg] = GridTreeValue(m_pGridPrevious, valExp);
        }
        string stepString = boost::lexical_cast<string>(p_iStep) ;
        *p_ar << gs::Record(contVal, (p_name + "Values").c_str(), stepString.c_str()) ;
        vector< GridTreeValue > controlVal(p_control.size());
        for (size_t iReg = 0; iReg < p_control.size(); ++iReg)
            controlVal[iReg] = GridTreeValue(m_pGridCurrent,  *p_control[iReg]);
        *p_ar << gs::Record(controlVal, (p_name + "Control").c_str(), stepString.c_str()) ;
        p_ar->flush() ; // necessary for python mapping
#ifdef USE_MPI
    }
#endif
}

