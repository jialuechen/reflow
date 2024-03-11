
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
#include "libflow/core/grids/FullRegularIntGridIterator.h"
#include "libflow/core/utils/eigenGeners.h"
#include "libflow/regression/BaseRegressionGeners.h"
#include "libflow/dp/TransitionStepRegressionSwitch.h"
#include "libflow/dp/OptimizerSwitchBase.h"



using namespace Eigen;
using namespace libflow;
using namespace std;


TransitionStepRegressionSwitch::TransitionStepRegressionSwitch(const  vector< shared_ptr<RegularSpaceIntGrid> >  &p_pGridCurrent,
        const  vector< shared_ptr<RegularSpaceIntGrid> >   &p_pGridPrevious,
        const  shared_ptr<OptimizerSwitchBase > &p_pOptimize
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

vector< shared_ptr< ArrayXXd > >  TransitionStepRegressionSwitch::oneStep(const vector< shared_ptr< ArrayXXd > > &p_phiIn,
        const shared_ptr< BaseRegression>     &p_condExp) const
{
    // number of regimes at current time
    int nbRegimes = m_pOptimize->getNbRegime();
    vector< shared_ptr< ArrayXXd > >  phiOut(nbRegimes);
    vector< ArrayXXd> phiOutLoc(nbRegimes);
    // Organize the data splitting : spread the incoming values on an extended grid
    for (int  iReg  = 0; iReg < nbRegimes ; ++iReg)
    {
        // only if the processor is working
        if (m_pGridCurrent[iReg]->getNbPoints() > 0)
        {

#ifdef USE_MPI
            int  rank = m_world.rank();
            int nbProc = m_world.size();
            //  allocate for solution
            int nbPointsCur = m_pGridCurrent[iReg]->getNbPoints();
            int npointPProcCur = (int)(nbPointsCur / nbProc);
            int nRestPointCur = nbPointsCur % nbProc;
            int iFirstPointCur = rank * npointPProcCur + (rank < nRestPointCur ? rank : nRestPointCur);
            int iLastPointCur  = iFirstPointCur + npointPProcCur + (rank < nRestPointCur ? 1 : 0);
            ArrayXi ilocToGLobal(iLastPointCur - iFirstPointCur);
            phiOutLoc[iReg].resize(p_condExp->getNbSimul(), iLastPointCur - iFirstPointCur);
#endif
            //  allocate for solution
            phiOut[iReg] = make_shared< ArrayXXd >(p_condExp->getNbSimul(), m_pGridCurrent[iReg]->getNbPoints());

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
                    FullRegularIntGridIterator  iterGridPoint = m_pGridCurrent[iReg]->getGridIterator();
                    // account for mpi and threads
#ifdef USE_MPI
                    iterGridPoint.jumpToAndInc(rank, nbProc, iThread);
#else
                    iterGridPoint.jumpToAndInc(0, 1, iThread);
#endif
                    // iterates on points of the grid
                    while (iterGridPoint.isValid())
                    {
                        ArrayXi pointCoord = iterGridPoint.getIntCoordinate();
                        // optimize the current point and the set of regimes
                        ArrayXd  solution = m_pOptimize->stepOptimize(m_pGridPrevious, iReg, pointCoord, p_condExp, p_phiIn);
#ifdef USE_MPI
                        // copie solution
                        int iposArray = iterGridPoint.getRelativePosition();
                        ilocToGLobal(iposArray) = iterGridPoint.getCount();
                        // copie solution
                        phiOutLoc[iReg].col(iposArray) = solution;
#else
                        // copie solution
                        (*phiOut[iReg]).col(iterGridPoint.getCount()) = solution;
#endif
                        iterGridPoint.nextInc(nbThreads);
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
            boost::mpi::all_gatherv<double>(m_world, phiOutLoc[iReg].data(), phiOutLoc[iReg].size(), storeGlob.data());
            for (int ipos = 0; ipos < ilocToGLobalGlob.size(); ++ipos)
                (*phiOut[iReg]).col(ilocToGLobalGlob(ipos)) = storeGlob.col(ipos);
#endif
        }

    }
    return phiOut;
}

void TransitionStepRegressionSwitch::dumpContinuationValues(shared_ptr<gs::BinaryFileArchive> p_ar, const string &p_name, const int &p_iStep, const vector< shared_ptr< ArrayXXd > > &p_phiIn, const  shared_ptr<BaseRegression>    &p_condExp) const
{
#ifdef USE_MPI
    if (m_world.rank() == 0)
    {
#endif
        string stepString = boost::lexical_cast<string>(p_iStep) ;
        // store regressor
        *p_ar <<  gs::Record(dynamic_cast<const BaseRegression &>(*p_condExp), "regressor", stepString.c_str()) ;
        for (size_t iReg = 0; iReg <  p_phiIn.size(); ++iReg)
        {
            // calculate function basis of regressed values of values at next date
            ArrayXXd basisValues = p_condExp->getCoordBasisFunctionMultiple(p_phiIn[iReg]->transpose()).transpose();
            *p_ar << gs::Record(basisValues, (p_name + "basisValues").c_str(), stepString.c_str()) ;
        }
        p_ar->flush() ; // necessary for python mapping
#ifdef USE_MPI
    }
#endif
}

