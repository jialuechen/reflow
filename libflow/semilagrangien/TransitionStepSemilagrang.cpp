
#include <memory>
#include <Eigen/Dense>
#include <boost/lexical_cast.hpp>
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
#include "libflow/semilagrangien/SemiLagrangEspCond.h"
#include "libflow/semilagrangien/TransitionStepSemilagrang.h"
#include "libflow/core/utils/eigenGeners.h"
#include "libflow/core/grids/SpaceGridGeners.h"
#include "libflow/core/grids/RegularSpaceGridGeners.h"
#include "libflow/core/grids/GeneralSpaceGridGeners.h"
#include "libflow/core/grids/SparseSpaceGridNoBoundGeners.h"
#include "libflow/core/grids/SparseSpaceGridBoundGeners.h"

using namespace libflow;
using namespace Eigen ;
using namespace std;

TransitionStepSemilagrang::TransitionStepSemilagrang(const  shared_ptr<SpaceGrid> &p_gridCurrent,
        const  shared_ptr<SpaceGrid> &p_gridPrevious,
        const  shared_ptr<OptimizerSLBase > &p_optimize
#ifdef USE_MPI
        , const boost::mpi::communicator &p_world
#endif
                                                    ):
    m_gridCurrent(p_gridCurrent),  m_gridPrevious(p_gridPrevious), m_optimize(p_optimize)
#ifdef USE_MPI
    , m_world(p_world)
#endif

{

}


pair< vector< shared_ptr< ArrayXd > >, vector<  shared_ptr< ArrayXd > > >  TransitionStepSemilagrang::oneStep(const vector< shared_ptr< ArrayXd > > &p_phiIn, const double &p_time,  const function<double(const int &, const Eigen::ArrayXd &)> &p_boundaryFunc) const
{
    // number of regimes at current time
    int nbRegimes = m_optimize->getNbRegime();
    vector< shared_ptr< ArrayXd > >  phiOut(nbRegimes);
    int nbControl =  m_optimize->getNbControl();
    vector< shared_ptr< ArrayXd > >  controlOut(nbControl);
    vector< shared_ptr< SemiLagrangEspCond > >  semilag(p_phiIn.size());
    // only if the processor is working
    if (m_gridCurrent->getNbPoints() > 0)
    {
        // create spectral operateur
        std::vector<std::shared_ptr<InterpolatorSpectral> > vecInterpolator(p_phiIn.size());
        // create a semi lagrangian object
        for (size_t iReg = 0; iReg < p_phiIn.size(); ++iReg)
        {
            vecInterpolator[iReg] = m_gridPrevious->createInterpolatorSpectral(*p_phiIn[iReg]);
            semilag[iReg] = make_shared<SemiLagrangEspCond>(vecInterpolator[iReg], m_gridPrevious->getExtremeValues(), m_optimize->getBModifVol());
        }

#ifdef USE_MPI
        int rank = m_world.rank();
        int nbProc = m_world.size();
        //  allocate for solution
        int nbPointsCur = m_gridCurrent->getNbPoints();
        int npointPProcCur = (int)(nbPointsCur / nbProc);
        int nRestPointCur = nbPointsCur % nbProc;
        int iFirstPointCur = rank * npointPProcCur + (rank < nRestPointCur ? rank : nRestPointCur);
        int iLastPointCur  = iFirstPointCur + npointPProcCur + (rank < nRestPointCur ? 1 : 0);
        std::vector< ArrayXd> phiOutLoc(nbRegimes), controlOutLoc(nbControl);
        ArrayXi ilocToGLobal(iLastPointCur - iFirstPointCur);
        for (int iReg = 0; iReg < nbRegimes; ++iReg)
            phiOutLoc[iReg].resize(iLastPointCur - iFirstPointCur);
        for (int iCont = 0; iCont < nbControl; ++iCont)
            controlOutLoc[iCont].resize(iLastPointCur - iFirstPointCur);

#endif

        //  allocate for solution
        for (int iReg = 0; iReg < nbRegimes; ++iReg)
            phiOut[iReg] = make_shared< ArrayXd >(m_gridCurrent->getNbPoints());
        for (int iCont = 0; iCont < nbControl; ++iCont)
            controlOut[iCont] = make_shared< ArrayXd >(m_gridCurrent->getNbPoints());


        // number of thread
#ifdef _OPENMP
        int nbThreads = omp_get_max_threads();
#else
        int nbThreads = 1;
#endif
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
                shared_ptr< GridIterator >  iterGridPoint = m_gridCurrent->getGridIterator();
#ifdef USE_MPI
                // account for mpi and threads
                iterGridPoint->jumpToAndInc(rank, nbProc, iThread);
#else
                iterGridPoint->jumpToAndInc(0, 1, iThread);
#endif
                // iterates on points of the grid
                while (iterGridPoint->isValid())
                {
                    ArrayXd pointCoord = iterGridPoint->getCoordinate();
                    if ((m_gridCurrent->isStrictlyInside(pointCoord)) || (m_optimize->isNotNeedingBC(pointCoord)))
                    {
                        // get value current function value
                        Eigen::ArrayXd phiPointIn(nbRegimes);
                        for (int iReg = 0; iReg < nbRegimes; ++iReg)
                            phiPointIn(iReg) = vecInterpolator[iReg]->apply(pointCoord);
                        // optimize the current point and the set of regimes
                        pair< ArrayXd, ArrayXd>  solutionAndControl = m_optimize->stepOptimize(pointCoord, semilag, p_time, phiPointIn);
#ifdef USE_MPI
                        // copie solution
                        int iposArray = iterGridPoint->getRelativePosition();
                        ilocToGLobal(iposArray) = iterGridPoint->getCount();
                        // copie solution
                        for (int iReg = 0; iReg < nbRegimes; ++iReg)
                            phiOutLoc[iReg](iposArray) = solutionAndControl.first(iReg);
                        for (int iCont = 0; iCont < nbControl; ++iCont)
                            controlOutLoc[iCont](iposArray) = solutionAndControl.second(iCont);
#else
                        for (int iReg = 0; iReg < nbRegimes; ++iReg)
                            (*phiOut[iReg])(iterGridPoint->getCount()) = solutionAndControl.first(iReg);
                        for (int iCont = 0; iCont < nbControl; ++iCont)
                            (*controlOut[iCont])(iterGridPoint->getCount()) = solutionAndControl.second(iCont);
#endif
                    }
                    else
                    {

#ifdef USE_MPI
                        int iposArray = iterGridPoint->getRelativePosition();
                        ilocToGLobal(iposArray) = iterGridPoint->getCount();
                        // use boundary condition
                        for (int iReg = 0; iReg < nbRegimes; ++iReg)
                            phiOutLoc[iReg](iposArray) = p_boundaryFunc(iReg, pointCoord);
                        for (int iCont = 0; iCont < nbControl; ++iCont)
                            controlOutLoc[iCont](iposArray) =  0. ;
#else
                        for (int iReg = 0; iReg < nbRegimes; ++iReg)
                            (*phiOut[iReg])(iterGridPoint->getCount()) = p_boundaryFunc(iReg, pointCoord);
                        for (int iCont = 0; iCont < nbControl; ++iCont)
                            (*controlOut[iCont])(iterGridPoint->getCount()) = 0.;

#endif
                    }
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
        ArrayXd storeGlob(nbPointsCur);
        for (int iReg = 0; iReg < nbRegimes; ++iReg)
        {
            boost::mpi::all_gatherv<double>(m_world, phiOutLoc[iReg].data(), phiOutLoc[iReg].size(), storeGlob.data());
            for (int ipos = 0; ipos < ilocToGLobalGlob.size(); ++ipos)
                (*phiOut[iReg])(ilocToGLobalGlob(ipos)) = storeGlob(ipos);
        }
        for (int iCont = 0 ; iCont < nbControl; ++iCont)
        {
            boost::mpi::all_gatherv<double>(m_world, controlOutLoc[iCont].data(), controlOutLoc[iCont].size(), storeGlob.data());
            for (int ipos = 0; ipos <  ilocToGLobalGlob.size(); ++ipos)
                (*controlOut[iCont])(ilocToGLobalGlob(ipos)) = storeGlob(ipos);
        }
#endif
    }
    return make_pair(phiOut, controlOut);
}

/// \brief Permits to dump continuation values on archive
/// \param p_ar                   archive to dump in
/// \param p_name                 name used for object
/// \param p_iStep               Step number or identifier for time step
/// \param p_phiIn                for each regime the function value
void TransitionStepSemilagrang::dumpValues(std::shared_ptr<gs::BinaryFileArchive> p_ar, const string &p_name, const int &p_iStep, const vector< shared_ptr< ArrayXd > > &p_phiIn,
        const vector< shared_ptr< ArrayXd > > &p_control) const
{
    int rank = 0 ;
#ifdef USE_MPI
    rank = m_world.rank();
#endif
    if (rank == 0)
    {
        string stepString = boost::lexical_cast<string>(p_iStep) ;
        string valDump = p_name + "Val";
        *p_ar << gs::Record(p_phiIn, valDump.c_str(), stepString.c_str()) ;
        string valControl = p_name + "Control";
        *p_ar << gs::Record(p_control, valControl.c_str(), stepString.c_str()) ;
        p_ar->flush();
    }
}

