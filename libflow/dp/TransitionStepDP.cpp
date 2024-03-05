// Copyright (C) 2016 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#include <memory>
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
#include "libflow/dp/TransitionStepDP.h"
#include "libflow/regression/GridAndRegressedValue.h"
#include "libflow/regression/GridAndRegressedValueGeners.h"


using namespace Eigen;
using namespace libflow;
using namespace std;


TransitionStepDP::TransitionStepDP(const  shared_ptr<FullGrid> &p_pGridCurrent,
                                   const  shared_ptr<FullGrid> &p_pGridPrevious,
                                   const  std::shared_ptr<BaseRegression> &p_regressorCurrent,
                                   const  std::shared_ptr<BaseRegression> &p_regressorPrevious,
                                   const  shared_ptr<OptimizerNoRegressionDPBase > &p_pOptimize
#ifdef USE_MPI
                                   , const boost::mpi::communicator &p_world
#endif
                                  ):
    m_pGridPrevious(p_pGridPrevious), m_pGridCurrent(p_pGridCurrent),   m_regressorPrevious(p_regressorPrevious), m_regressorCurrent(p_regressorCurrent),
    m_pOptimize(p_pOptimize)
#ifdef USE_MPI
    , m_world(p_world)
#endif
{
}

std::pair< shared_ptr< vector<  Eigen::ArrayXXd > >, shared_ptr< vector<  Eigen::ArrayXXd > >  > TransitionStepDP::oneStep(const vector<  Eigen::ArrayXXd > &p_phiIn) const
{
    // number of regimes at current time
    int nbRegimes = m_pOptimize->getNbRegime();
    shared_ptr< vector< ArrayXXd > >  phiOut = make_shared<vector< ArrayXXd > > (nbRegimes);
    int nbControl =  m_pOptimize->getNbControl();
    shared_ptr< vector<  ArrayXXd > >  controlOut = make_shared<vector< ArrayXXd > > (nbControl);
    // only if the processor is working
    if (m_pGridCurrent->getNbPoints() > 0)
    {
        // Create Regressor using the values p_phin
        std::vector<GridAndRegressedValue> valGridReg(nbRegimes);
        for (int ireg = 0; ireg < nbRegimes; ++ireg)
        {
            valGridReg[ireg] = GridAndRegressedValue(m_pGridPrevious, m_regressorPrevious);
            valGridReg[ireg].setRegressedValues(p_phiIn[ireg]);
        }
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
            phiOutLoc[iReg].resize(m_regressorCurrent->getNumberOfFunction(), iLastPointCur - iFirstPointCur);
        for (int iCont = 0; iCont < nbControl; ++iCont)
            controlOutLoc[iCont].resize(m_regressorCurrent->getNumberOfFunction(), iLastPointCur - iFirstPointCur);
#endif

        //  allocate for solution
        for (int  iReg = 0; iReg < nbRegimes; ++iReg)
            (*phiOut)[iReg] = ArrayXXd(m_regressorCurrent->getNumberOfFunction(), m_pGridCurrent->getNbPoints());
        for (int iCont = 0; iCont < nbControl; ++iCont)
            (*controlOut)[iCont] = ArrayXXd(m_regressorCurrent->getNumberOfFunction(), m_pGridCurrent->getNbPoints());

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
                    std::pair< ArrayXXd, ArrayXXd>  solutionAndControl = m_pOptimize->stepOptimize(pointCoord, valGridReg, m_regressorCurrent);
#ifdef USE_MPI
                    // copie solution
                    int iposArray = iterGridPoint->getRelativePosition();
                    ilocToGLobal(iposArray) = iterGridPoint->getCount();
                    // copie solution
                    for (int iReg = 0; iReg < nbRegimes; ++iReg)
                        phiOutLoc[iReg].col(iposArray) = m_regressorCurrent->getCoordBasisFunction(solutionAndControl.first.col(iReg));
                    for (int iCont = 0; iCont < nbControl; ++iCont)
                        controlOutLoc[iCont].col(iposArray) =  m_regressorCurrent->getCoordBasisFunction(solutionAndControl.second.col(iCont));

#else
                    // copie solution
                    for (int iReg = 0; iReg < nbRegimes; ++iReg)
                        (*phiOut)[iReg].col(iterGridPoint->getCount()) = m_regressorCurrent->getCoordBasisFunction(solutionAndControl.first.col(iReg));
                    for (int iCont = 0; iCont < nbControl; ++iCont)
                        (*controlOut)[iCont].col(iterGridPoint->getCount()) = m_regressorCurrent->getCoordBasisFunction(solutionAndControl.second.col(iCont));
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
        ArrayXXd storeGlob(m_regressorCurrent->getNumberOfFunction(), nbPointsCur);
        for (int iReg = 0; iReg < nbRegimes; ++iReg)
        {
            boost::mpi::all_gatherv<double>(m_world, phiOutLoc[iReg].data(), phiOutLoc[iReg].size(), storeGlob.data());
            for (int ipos = 0; ipos < ilocToGLobalGlob.size(); ++ipos)
                (*phiOut)[iReg].col(ilocToGLobalGlob(ipos)) = storeGlob.col(ipos);
        }
        for (int iCont = 0 ; iCont < nbControl; ++iCont)
        {
            boost::mpi::all_gatherv<double>(m_world, controlOutLoc[iCont].data(), controlOutLoc[iCont].size(), storeGlob.data());
            for (int ipos = 0; ipos <  ilocToGLobalGlob.size(); ++ipos)
                (*controlOut)[iCont].col(ilocToGLobalGlob(ipos)) = storeGlob.col(ipos);
        }
#endif
    }
    return make_pair(phiOut, controlOut);
}

void TransitionStepDP::dumpValues(std::shared_ptr<gs::BinaryFileArchive> p_ar, const string &p_name, const int &p_iStep,
                                  const vector< ArrayXXd   > &p_control) const
{
#ifdef USE_MPI
    if (m_world.rank() == 0)
    {
#endif
        std::vector<GridAndRegressedValue> contGridReg(p_control.size());
        for (size_t ireg = 0; ireg < p_control.size(); ++ireg)
        {
            contGridReg[ireg] = GridAndRegressedValue(m_pGridCurrent, m_regressorCurrent);
            contGridReg[ireg].setRegressedValues(p_control[ireg]);
        }
        string stepString = boost::lexical_cast<string>(p_iStep) ;
        *p_ar << gs::Record(contGridReg, (p_name + "Control").c_str(), stepString.c_str()) ;
        p_ar->flush() ; // necessary for python mapping

#ifdef USE_MPI
    }
#endif
}

