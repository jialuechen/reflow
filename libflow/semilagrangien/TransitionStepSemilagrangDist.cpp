
#ifdef USE_MPI
#include <memory>
#include <boost/mpi.hpp>
#include <boost/lexical_cast.hpp>
#ifdef _OPENMP
#include <omp.h>
#include "libflow/core/utils/OpenmpException.h"
#endif
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "geners/Record.hh"
#include "geners/vectorIO.hh"
#include "libflow/core/parallelism/ParallelComputeGridSplitting.h"
#include "libflow/core/utils/primeNumber.h"
#include "libflow/core/grids/GridIterator.h"
#include "libflow/semilagrangien/OptimizerSLBase.h"
#include "libflow/semilagrangien/TransitionStepSemilagrangDist.h"
#include "libflow/core/parallelism/GridReach.h"
#include "libflow/core/utils/eigenGeners.h"
#include "libflow/core/grids/FullGridGeners.h"
#include "libflow/core/grids/RegularLegendreGridDerivedGeners.h"
#include "libflow/core/grids/RegularSpaceGridDerivedGeners.h"
#include "libflow/core/grids/GeneralSpaceGridDerivedGeners.h"


using namespace  libflow;
using namespace  Eigen;
using namespace  std;



TransitionStepSemilagrangDist::TransitionStepSemilagrangDist(const  shared_ptr<FullGrid> &p_gridCurrent,
        const  shared_ptr<FullGrid> &p_gridPrevious,
        const  shared_ptr<OptimizerSLBase > &p_optimize,
        const boost::mpi::communicator &p_world):
    m_gridCurrent(p_gridCurrent), m_gridPrevious(p_gridPrevious), m_optimize(p_optimize), m_world(p_world)
{
    // initial and previous dimensions
    ArrayXi initialDimension   = p_gridCurrent->getDimensions();
    ArrayXi initialDimensionPrev  = p_gridPrevious->getDimensions();
    // organize the hypercube splitting for parallel
    ArrayXi splittingRatio = paraOptimalSplitting(initialDimension, p_optimize->getDimensionToSplit(), m_world);
    ArrayXi splittingRatioPrev = paraOptimalSplitting(initialDimensionPrev, p_optimize->getDimensionToSplit(), m_world);
    // cone value
    function < SubMeshIntCoord(const SubMeshIntCoord &) > fMesh = GridReach<OptimizerSLBase>(p_gridCurrent, p_gridPrevious, p_optimize);
    // ParallelComputeGridsSplitting objects
    m_paral = make_shared<ParallelComputeGridSplitting>(initialDimension, initialDimensionPrev, fMesh, splittingRatio, splittingRatioPrev, m_world);
    // get back grid treated by current processor
    Array<  array<int, 2 >, Dynamic, 1 > gridLocal = m_paral->getCurrentCalculationGrid();
    // Construct local sub grid
    m_gridCurrentProc = m_gridCurrent->getSubGrid(gridLocal);
    // only if the grid is not empty
    if (m_gridCurrentProc->getNbPoints() > 0)
    {
        // get back grid extended on previous step
        Array<  array<int, 2 >, Dynamic, 1 > gridLocalExtended = m_paral->getExtendedGridProcOldGrid();
        m_gridExtendPreviousStep =  p_gridPrevious->getSubGrid(gridLocalExtended);
    }
}

pair< vector< shared_ptr< ArrayXd > >, vector<  shared_ptr< ArrayXd > > >  TransitionStepSemilagrangDist::oneStep(const vector< shared_ptr< ArrayXd > > &p_phiIn,  const double &p_time,   const function<double(const int &, const Eigen::ArrayXd &)> &p_boundaryFunc) const
{
    // number of regimes at current time
    int nbRegimes = m_optimize->getNbRegime();
    vector< shared_ptr< ArrayXd > >  phiOut(nbRegimes);
    int nbControl =  m_optimize->getNbControl();
    vector< shared_ptr< ArrayXd > >  controlOut(nbControl);
    vector< shared_ptr< SemiLagrangEspCond > >  semilag(p_phiIn.size());
    vector < shared_ptr< ArrayXd > > phiInExtended(p_phiIn.size());
    for (size_t iReg  = 0; iReg < p_phiIn.size() ; ++iReg)
    {
        if (p_phiIn[iReg])
            phiInExtended[iReg] = make_shared< ArrayXd >(m_paral->runOneStep(*p_phiIn[iReg])) ;
        else
        {
            // utilitary
            ArrayXd emptyArray;
            phiInExtended[iReg] = make_shared< ArrayXd >(m_paral->runOneStep(emptyArray)) ;
        }
    }

    // only if the processor is working
    if (m_gridCurrentProc->getNbPoints() > 0)
    {
        // create spectral operateur
        std::vector<std::shared_ptr<InterpolatorSpectral> > vecInterpolator(p_phiIn.size());
        // Organize the data splitting : spread the incoming values on an extended grid
        for (size_t iReg  = 0; iReg < p_phiIn.size() ; ++iReg)
        {
            vecInterpolator[iReg] = m_gridExtendPreviousStep->createInterpolatorSpectral(*phiInExtended[iReg]);
            // create a semi lagrangian object
            semilag[iReg] = make_shared<SemiLagrangEspCond>(vecInterpolator[iReg], m_gridExtendPreviousStep->getExtremeValues(), m_optimize->getBModifVol());
        }

        //  allocate for solution
        for (int iReg = 0; iReg < nbRegimes; ++iReg)
            phiOut[iReg] = make_shared< ArrayXd >(m_gridCurrentProc->getNbPoints());
        for (int iCont = 0; iCont < nbControl; ++iCont)
            controlOut[iCont] = make_shared< ArrayXd >(m_gridCurrentProc->getNbPoints());


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
                shared_ptr< GridIterator > iterGridPoint = m_gridCurrentProc->getGridIterator();

                // account fo threads
                iterGridPoint->jumpToAndInc(0, 1, iThread);

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
                        std::pair< ArrayXd, ArrayXd>  solutionAndControl = m_optimize->stepOptimize(pointCoord, semilag, p_time, phiPointIn);
                        // copie solution
                        for (int  iReg = 0; iReg < nbRegimes; ++iReg)
                            (*phiOut[iReg])(iterGridPoint->getCount()) = solutionAndControl.first(iReg);
                        for (int iCont = 0; iCont < nbControl; ++iCont)
                            (*controlOut[iCont])(iterGridPoint->getCount()) = solutionAndControl.second(iCont);
                    }
                    else
                    {
                        // use boundary condition
                        for (int iReg = 0; iReg < nbRegimes; ++iReg)
                            (*phiOut[iReg])(iterGridPoint->getCount()) = p_boundaryFunc(iReg, pointCoord);
                        for (int iCont = 0; iCont < nbControl; ++iCont)
                            for (int iCont = 0; iCont < nbControl; ++iCont)
                                (*controlOut[iCont])(iterGridPoint->getCount()) = 0. ;

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
    }
    return make_pair(phiOut, controlOut);
}

void TransitionStepSemilagrangDist::dumpValues(shared_ptr<gs::BinaryFileArchive> p_ar, const string &p_name, const int &p_iStep,
        const vector< shared_ptr< ArrayXd > > &p_phiIn, const vector< shared_ptr< ArrayXd > > &p_control, const bool &p_bOneFile) const
{
    string stepString = boost::lexical_cast<string>(p_iStep) ;
    ArrayXi initialDimensionPrev  = m_gridPrevious->getDimensions();
    ArrayXi initialDimension  = m_gridCurrent->getDimensions();
    if (!p_bOneFile)
    {
        // dump caracteristics of the splitting
        // organize the hypercube splitting for parallel
        vector<int> vecPrev(initialDimensionPrev.data(), initialDimensionPrev.data() + initialDimensionPrev.size());
        *p_ar << gs::Record(vecPrev, "initialSizeOfMeshPrev", stepString.c_str()) ;
        vector<int> vec(initialDimension.data(), initialDimension.data() + initialDimension.size());
        *p_ar << gs::Record(vec, "initialSizeOfMesh", stepString.c_str()) ;
        string valDump = p_name + "Val";
        *p_ar << gs::Record(p_phiIn, valDump.c_str(), stepString.c_str()) ;
        string valDumpCont = p_name + "Control";
        *p_ar << gs::Record(p_control, valDumpCont.c_str(), stepString.c_str()) ;
    }
    else
    {
        // utilitary
        Array< array<int, 2 >, Dynamic, 1 >  gridOnProc0Prev(initialDimensionPrev.size());
        for (int id = 0; id < initialDimensionPrev.size(); ++id)
        {
            gridOnProc0Prev(id)[0] = 0 ;
            gridOnProc0Prev(id)[1] = initialDimensionPrev(id) ;
        }
        ArrayXi splittingRatioPrev = paraOptimalSplitting(initialDimensionPrev, m_optimize->getDimensionToSplit(), m_world);
        ParallelComputeGridSplitting  paralObjectPrev(initialDimensionPrev, splittingRatioPrev, m_world);
        vector< shared_ptr< ArrayXd > > vecArrayPrev(p_phiIn.size());
        for (size_t iReg = 0; iReg < p_phiIn.size(); ++iReg)
        {
            if (m_world.rank() < m_paral->getNbProcessorUsedPrev())
                vecArrayPrev[iReg] = make_shared<ArrayXd>(paralObjectPrev.reconstruct(*p_phiIn[iReg], gridOnProc0Prev));
        }
        if (m_world.rank() == 0)
        {
            string valDump =  p_name  + "Val"; ;
            *p_ar << gs::Record(vecArrayPrev, valDump.c_str(), stepString.c_str()) ;
        }
        Array< array<int, 2 >, Dynamic, 1 >  gridOnProc0(initialDimension.size());
        for (int id = 0; id < initialDimension.size(); ++id)
        {
            gridOnProc0(id)[0] = 0 ;
            gridOnProc0(id)[1] = initialDimension(id) ;
        }
        ArrayXi splittingRatio = paraOptimalSplitting(initialDimension, m_optimize->getDimensionToSplit(), m_world);
        ParallelComputeGridSplitting  paralObject(initialDimension, splittingRatio, m_world);
        vector< shared_ptr< ArrayXd > > vecArray(p_control.size());
        for (size_t iCont = 0; iCont < p_control.size(); ++iCont)
        {
            if (m_world.rank() < m_paral->getNbProcessorUsed())
                vecArray[iCont] = make_shared<ArrayXd>(paralObject.reconstruct(*p_control[iCont], gridOnProc0));
        }
        if (m_world.rank() == 0)
        {
            string valDump =  p_name  + "Control"; ;
            *p_ar << gs::Record(vecArray, valDump.c_str(), stepString.c_str()) ;
        }
    }
    if (m_world.rank() == 0)
        p_ar->flush();
    m_world.barrier() ;
}

void  TransitionStepSemilagrangDist::reconstructOnProc0(const vector< shared_ptr< Eigen::ArrayXd > > &p_phiIn, vector< shared_ptr< Eigen::ArrayXd > > &p_phiOut)
{
    p_phiOut.resize(p_phiIn.size());
    ArrayXi initialDimension   = m_gridCurrent->getDimensions();
    Array< array<int, 2 >, Dynamic, 1 >  gridOnProc0(initialDimension.size());
    for (int id = 0; id < initialDimension.size(); ++id)
    {
        gridOnProc0(id)[0] = 0 ;
        gridOnProc0(id)[1] = initialDimension(id) ;
    }
    for (size_t i = 0; i < p_phiIn.size(); ++i)
    {
        if (m_world.rank() < m_paral->getNbProcessorUsed())
            p_phiOut[i] = make_shared<Eigen::ArrayXd>(m_paral->reconstruct(*p_phiIn[i], gridOnProc0));
    }
}
#endif
