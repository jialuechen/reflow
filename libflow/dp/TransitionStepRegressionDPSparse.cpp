
#include <memory>
#ifdef USE_MPI
#include "boost/mpi.hpp"
#include "libflow/core/parallelism/all_gatherv.hpp"
#endif
#ifdef _OPENMP
#include <omp.h>
#include "libflow/core/utils/OpenmpException.h"
#endif
#include "geners/Record.hh"
#include "libflow/core/grids/SparseGridIterator.h"
#include "libflow/dp/TransitionStepRegressionDPSparse.h"
#include "libflow/regression/ContinuationValue.h"
#include "libflow/regression/ContinuationValueGeners.h"
#include "libflow/regression/GridAndRegressedValue.h"
#include "libflow/regression/GridAndRegressedValueGeners.h"


using namespace Eigen;
using namespace libflow;
using namespace std;


TransitionStepRegressionDPSparse::TransitionStepRegressionDPSparse(const  shared_ptr<SparseSpaceGrid> &p_pGridCurrent,
        const  shared_ptr<SparseSpaceGrid> &p_pGridPrevious,
        const  shared_ptr<OptimizerDPBase > &p_pOptimize
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

std::pair< vector< shared_ptr< ArrayXXd > >, vector<  shared_ptr< ArrayXXd > > > TransitionStepRegressionDPSparse::oneStep(const vector< shared_ptr< ArrayXXd > > &p_phiIn,
        const shared_ptr< BaseRegression>     &p_condExp) const
{
    int rank = 0 ;
    int nbProc = 1;
    int nbSimul =  p_condExp->getNbSimul();
#ifdef USE_MPI
    rank = m_world.rank();
    nbProc = m_world.size();
#endif
    // number of regimes at current time
    int nbRegimes = m_pOptimize->getNbRegime();
    vector< shared_ptr< ArrayXXd > >  phiOut(nbRegimes);
    // to store hierarchical values of cash
    vector< shared_ptr< ArrayXXd > > cashHierar(p_phiIn.size());
    // parallelism
    int nsimPProc = (int)(nbSimul / nbProc);
    int nRestSim = nbSimul % nbProc;
    int iFirstSim = rank * nsimPProc + (rank < nRestSim ? rank : nRestSim);
    int iLastSim  = iFirstSim + nsimPProc + (rank < nRestSim ? 1 : 0);
    for (int iReg = 0; iReg <  nbRegimes; ++iReg)
    {
        cashHierar[iReg] = make_shared<ArrayXXd>(p_phiIn[iReg]->rows(), p_phiIn[iReg]->cols());
        ArrayXXd valHierar(p_phiIn[iReg]->cols(), iLastSim - iFirstSim);
        ArrayXd valHierarCol(p_phiIn[iReg]->cols());
        for (int is = iFirstSim; is < iLastSim; ++is)
        {
            // Hierarchize
            valHierarCol = p_phiIn[iReg]->row(is).transpose();
            m_pGridPrevious->toHierarchize(valHierarCol);
            valHierar.col(is - iFirstSim)  = valHierarCol;
        }
#ifdef USE_MPI
        if (m_world.size() > 1)
        {
            ArrayXXd valHierarShared(p_phiIn[iReg]->cols(), nbSimul);
            boost::mpi::all_gatherv<double>(m_world, valHierar.data(), valHierar.size(), valHierarShared.data());
            *cashHierar[iReg] = valHierarShared.transpose();
        }
        else
#endif
            *cashHierar[iReg] = valHierar.transpose();
    }
    int nbControl =  m_pOptimize->getNbControl();
    vector< shared_ptr< ArrayXXd > >  controlOut(nbControl);
    // only if the processor is working
    if (m_pGridCurrent->getNbPoints() > 0)
    {

        // number of thread
#ifdef _OPENMP
        int nbThreads = omp_get_max_threads();
#else
        int nbThreads = 1;
#endif
        //  create continuation values on Hierarchical values
        vector< ContinuationValue > contVal(p_phiIn.size());
        int nbPointsPrev = m_pGridPrevious->getNbPoints();
        int npointPProcPrev = (int)(nbPointsPrev / nbProc);
        int nRestPointPrev = nbPointsPrev % nbProc;
        int iFirstPointPrev = rank * npointPProcPrev + (rank < nRestPointPrev ? rank : nRestPointPrev);
        int iLastPointPrev  = iFirstPointPrev + npointPProcPrev + (rank < nRestPointPrev ? 1 : 0);
        for (size_t iReg = 0; iReg < p_phiIn.size(); ++iReg)
        {
            ArrayXXd regressed = p_condExp->getCoordBasisFunctionMultiple(p_phiIn[iReg]->block(0, iFirstPointPrev, p_phiIn[iReg]->rows(), iLastPointPrev - iFirstPointPrev).transpose());
#ifdef USE_MPI
            if (m_world.size() > 1)
            {
                ArrayXXd regressedShared(regressed.cols(), nbPointsPrev);
                ArrayXXd regrTranspose = regressed.transpose();
                boost::mpi::all_gatherv<double>(m_world, regrTranspose.data(), regrTranspose.size(), regressedShared.data());
                regressed = regressedShared.transpose();
            }
#endif
            // don't parallelize because hierarchize the regression coefficients and not the simulations
            ArrayXd vHierarLoc(regressed.cols());
            for (int iFonc = 0 ; iFonc < regressed.cols(); ++iFonc)
            {
                vHierarLoc = regressed.col(iFonc);
                m_pGridPrevious->toHierarchize(vHierarLoc);
                regressed.col(iFonc) = vHierarLoc;
            }
            contVal[iReg].loadForSimulation(m_pGridPrevious, p_condExp, regressed.transpose());
        }

        //  allocate for solution
#ifdef USE_MPI
        int nbPointsCur = m_pGridCurrent->getNbPoints();
        int npointPProcCur = (int)(nbPointsCur / nbProc);
        int nRestPointCur = nbPointsCur % nbProc;
        int iFirstPointCur = rank * npointPProcCur + (rank < nRestPointCur ? rank : nRestPointCur);
        int iLastPointCur  = iFirstPointCur + npointPProcCur + (rank < nRestPointCur ? 1 : 0);
        std::vector< ArrayXXd> phiOutLoc(nbRegimes), controlOutLoc(nbControl);
        ArrayXi ilocToGLobal(iLastPointCur - iFirstPointCur);
        for (int  iReg = 0; iReg < nbRegimes; ++iReg)
            phiOutLoc[iReg].resize(p_condExp->getNbSimul(), iLastPointCur - iFirstPointCur);
        for (int iCont = 0; iCont < nbControl; ++iCont)
            controlOutLoc[iCont].resize(p_condExp->getNbSimul(), iLastPointCur - iFirstPointCur);
#endif
        for (int  iReg = 0; iReg < nbRegimes; ++iReg)
            phiOut[iReg] = make_shared< ArrayXXd >(p_condExp->getNbSimul(), m_pGridCurrent->getNbPoints());
        for (int iCont = 0; iCont < nbControl; ++iCont)
            controlOut[iCont] = make_shared< ArrayXXd >(p_condExp->getNbSimul(), m_pGridCurrent->getNbPoints());

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
                shared_ptr< GridIterator > iterGridPoint = m_pGridCurrent->getGridIterator();
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
                    std::pair< ArrayXXd, ArrayXXd>  solutionAndControl = m_pOptimize->stepOptimize(m_pGridPrevious, pointCoord, contVal, cashHierar);
                    // copie solution
#ifdef USE_MPI
                    int iposArray = iterGridPoint->getRelativePosition();
                    ilocToGLobal(iposArray) = iterGridPoint->getCount();
                    for (int iReg = 0; iReg < nbRegimes; ++iReg)
                        phiOutLoc[iReg].col(iposArray) = solutionAndControl.first.col(iReg);
                    for (int iCont = 0; iCont < nbControl; ++iCont)
                        controlOutLoc[iCont].col(iposArray) = solutionAndControl.second.col(iCont);
#else
                    for (int iReg = 0; iReg < nbRegimes; ++iReg)
                        phiOut[iReg]->col(iterGridPoint->getCount()) = solutionAndControl.first.col(iReg);
                    for (int iCont = 0; iCont < nbControl; ++iCont)
                        controlOut[iCont]->col(iterGridPoint->getCount()) = solutionAndControl.second.col(iCont);
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
        ArrayXXd storeGlob(p_condExp->getNbSimul(), nbPointsCur);
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

void TransitionStepRegressionDPSparse::dumpContinuationValues(shared_ptr<gs::BinaryFileArchive> p_ar, const string &p_name, const int &p_iStep, const vector< shared_ptr< ArrayXXd > > &p_phiIn,
        const vector< shared_ptr< ArrayXXd > > &p_control, const  shared_ptr<BaseRegression>    &p_condExp) const
{
    int nbProc = 1;
    int rank = 0 ;
#ifdef USE_MPI
    rank = m_world.rank();
    nbProc = m_world.size();
#endif

    // hierarchize all values
    vector< GridAndRegressedValue > contVal(p_phiIn.size());
    int nbPoints = m_pGridPrevious->getNbPoints();
    int npointPProc = (int)(nbPoints / nbProc);
    int nRestPoint = nbPoints % nbProc;
    int iFirstPoint = rank * npointPProc + (rank < nRestPoint ? rank : nRestPoint);
    int iLastPoint  = iFirstPoint + npointPProc + (rank < nRestPoint ? 1 : 0);
    for (size_t iReg = 0; iReg < p_phiIn.size(); ++iReg)
    {
        // first regress the cash values on all stock points
        ArrayXXd regressed = p_condExp->getCoordBasisFunctionMultiple(p_phiIn[iReg]->block(0, iFirstPoint, p_phiIn[iReg]->rows(), iLastPoint - iFirstPoint).transpose());
#ifdef USE_MPI
        if (m_world.size() > 1)
        {
            ArrayXXd regressedShared(regressed.cols(), nbPoints);
            ArrayXXd regrTranspose = regressed.transpose();
            boost::mpi::all_gatherv<double>(m_world, regrTranspose.data(), regrTranspose.size(), regressedShared.data());
            regressed = regressedShared.transpose();
        }
#endif
        contVal[iReg] = GridAndRegressedValue(m_pGridPrevious, p_condExp);
        contVal[iReg].setRegressedValues(regressed.transpose());
    }
    string stepString = boost::lexical_cast<string>(p_iStep) ;
#ifdef USE_MPI
    if (rank == 0)
#endif
        *p_ar << gs::Record(contVal, (p_name + "Values").c_str(), stepString.c_str()) ;

    // Hierarchize control
    nbPoints = m_pGridCurrent->getNbPoints();
    npointPProc = (int)(nbPoints / nbProc);
    nRestPoint = nbPoints % nbProc;
    iFirstPoint = rank * npointPProc + (rank < nRestPoint ? rank : nRestPoint);
    iLastPoint  = iFirstPoint + npointPProc + (rank < nRestPoint ? 1 : 0);
    // number of controls
    int nbControl = p_control.size();
    vector< GridAndRegressedValue > controlVal(nbControl);
    for (int iCont = 0; iCont < nbControl; ++iCont)
    {
        // first regress the cash values on all stock points
        ArrayXXd contRegressed = p_condExp->getCoordBasisFunctionMultiple(p_control[iCont]->block(0, iFirstPoint, p_control[iCont]->rows(), iLastPoint - iFirstPoint).transpose());
#ifdef USE_MPI
        if (m_world.size() > 1)
        {
            ArrayXXd regressedShared(contRegressed.cols(), nbPoints);
            ArrayXXd regrTranspose = contRegressed.transpose();
            boost::mpi::all_gatherv<double>(m_world, regrTranspose.data(), regrTranspose.size(), regressedShared.data());
            contRegressed = regressedShared.transpose();
        }
#endif
        controlVal[iCont] = GridAndRegressedValue(m_pGridCurrent, p_condExp);
        controlVal[iCont].setRegressedValues(contRegressed.transpose());
    }
#ifdef USE_MPI
    if (rank == 0)
    {
#endif
        *p_ar << gs::Record(controlVal, (p_name + "Control").c_str(), stepString.c_str()) ;
        p_ar->flush() ; // necessary for python mapping
#ifdef USE_MPI
    }
#endif

}

