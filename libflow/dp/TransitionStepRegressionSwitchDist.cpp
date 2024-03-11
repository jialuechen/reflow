
#ifdef USE_MPI
#include <functional>
#include <memory>
#include <boost/mpi.hpp>
#ifdef _OPENMP
#include <omp.h>
#include "libflow/core/utils/OpenmpException.h"
#endif
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "geners/Record.hh"
#include "geners/vectorIO.hh"
#include "libflow/core/utils/types.h"
#include "libflow/core/parallelism/ParallelComputeGridSplitting.h"
#include "libflow/core/grids/FullRegularIntGridIterator.h"
#include "libflow/core/grids/RegularSpaceIntGrid.h"
#include "libflow/core/utils/eigenGeners.h"
#include "libflow/regression/BaseRegressionGeners.h"
#include "libflow/dp/OptimizerSwitchBase.h"
#include "libflow/dp/TransitionStepRegressionSwitchDist.h"


using namespace  libflow;
using namespace  Eigen;
using namespace  std;



TransitionStepRegressionSwitchDist::TransitionStepRegressionSwitchDist(const  vector< shared_ptr<RegularSpaceIntGrid> >  &p_pGridCurrent,
        const  vector< shared_ptr<RegularSpaceIntGrid> >  &p_pGridPrevious,
        const  shared_ptr<OptimizerSwitchBase > &p_pOptimize,
        const boost::mpi::communicator &p_world): m_pGridCurrent(p_pGridCurrent),
    m_pGridPrevious(p_pGridPrevious), m_pOptimize(p_pOptimize), m_paral(p_pGridCurrent.size()), m_gridCurrentProc(p_pGridCurrent.size()), m_gridExtendPreviousStep(p_pGridCurrent.size()), m_world(p_world)
{
    vector< Array< bool, Dynamic, 1> > dimToSplit = m_pOptimize->getDimensionToSplit();
    for (size_t iReg = 0; iReg <  p_pGridCurrent.size(); ++iReg)
    {
        // initial and previous dimensions
        ArrayXi initialDimension   = p_pGridCurrent[iReg]->getDimensions();
        ArrayXi initialDimensionPrev  = p_pGridPrevious[iReg]->getDimensions();
        // organize the hypercube splitting for parallel
        ArrayXi splittingRatio = paraOptimalSplitting(initialDimension, dimToSplit[iReg], m_world);
        ArrayXi splittingRatioPrev = paraOptimalSplitting(initialDimensionPrev, dimToSplit[iReg], m_world);
        // cone value
        auto fMesh = [iReg, p_pGridCurrent, p_pGridPrevious, p_pOptimize](const SubMeshIntCoord & p_intMesh)->SubMeshIntCoord
        {
            vector< array<int, 2 > >  intMeshUsed(p_intMesh.size());
            for (int i = 0; i < p_intMesh.size(); ++i)
            {
                // from grid starting at 0 to real grid position
                intMeshUsed[i][0]  = p_intMesh(i)[0] + p_pGridCurrent[iReg]->getLowValueDim(i);
                intMeshUsed[i][1]  = p_intMesh(i)[1] + p_pGridCurrent[iReg]->getLowValueDim(i) - 1; // last is outside the grid
            }
            vector< array<int, 2 > > retCone = p_pOptimize->getCone(iReg, intMeshUsed);
            // cap to max, min
            SubMeshIntCoord ret(retCone.size());
            for (int i = 0; i < p_intMesh.size(); ++i)
            {
                ret(i)[0] = max(retCone[i][0], p_pGridPrevious[iReg]->getLowValueDim(i))  ;
                ret(i)[1] = min(retCone[i][1], p_pGridPrevious[iReg]->getMaxValueDim(i));
            }
            for (int i = 0; i < p_intMesh.size(); ++i)
            {
                ret(i)[1] += 1; // border should be  outside
                // from real position to position starting at 0
                ret(i)[0]  -= p_pGridPrevious[iReg]->getLowValueDim(i);
                ret(i)[1]  -= p_pGridPrevious[iReg]->getLowValueDim(i);
            }
            return ret;
        };

        // ParallelComputeGridsSplitting objects
        m_paral[iReg] = make_shared<ParallelComputeGridSplitting>(initialDimension, initialDimensionPrev,
                        function < SubMeshIntCoord(const SubMeshIntCoord &) >(fMesh),
                        splittingRatio, splittingRatioPrev, m_world);
        // get back grid treated by current processor
        SubMeshIntCoord gridLocal = m_paral[iReg]->getCurrentCalculationGrid();
        // Construct local sub grid
        m_gridCurrentProc[iReg] = m_pGridCurrent[iReg]->getSubGrid(gridLocal);
        // only if the grid is not empty
        if (m_gridCurrentProc[iReg]->getNbPoints() > 0)
        {
            // get back grid extended on previous step
            SubMeshIntCoord  gridLocalExtended = m_paral[iReg]->getExtendedGridProcOldGrid();
            m_gridExtendPreviousStep [iReg] =  m_pGridPrevious[iReg]->getSubGrid(gridLocalExtended);
        }
    }
}


vector< shared_ptr< ArrayXXd >>  TransitionStepRegressionSwitchDist::oneStep(const vector< shared_ptr< ArrayXXd > > &p_phiIn,
                             const shared_ptr< BaseRegression>  &p_condExp) const
{
    // number of regimes at current time
    int nbRegimes = m_pOptimize->getNbRegime();
    vector< shared_ptr< ArrayXXd > >  phiOut(nbRegimes);
    // only if the processor is working
    vector < shared_ptr< ArrayXXd > > phiInExtended(p_phiIn.size());
    // Organize the data splitting : spread the incoming values on an extended grid
    for (int  iReg  = 0; iReg < nbRegimes ; ++iReg)
    {
        // utilitary
        ArrayXXd emptyArray;
        if (p_phiIn[iReg])
        {
            phiInExtended[iReg] = make_shared< ArrayXXd >(m_paral[iReg]->runOneStep(*p_phiIn[iReg])) ;
        }
        else
            phiInExtended[iReg] = make_shared< ArrayXXd >(m_paral[iReg]->runOneStep(emptyArray)) ;

        if (m_gridCurrentProc[iReg]->getNbPoints() > 0)
        {
            //  allocate for solution
            phiOut[iReg] = make_shared< ArrayXXd >(p_condExp->getNbSimul(), m_gridCurrentProc[iReg]->getNbPoints());
        }
    }

    for (int  iReg  = 0; iReg < nbRegimes ; ++iReg)
    {
        if (m_gridCurrentProc[iReg]->getNbPoints() > 0)
        {

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
                    FullRegularIntGridIterator  iterGridPoint = m_gridCurrentProc[iReg]->getGridIterator();

                    // account fo threads
                    iterGridPoint.jumpToAndInc(0, 1, iThread);

                    // iterates on points of the grid
                    while (iterGridPoint.isValid())
                    {
                        ArrayXi pointCoord = iterGridPoint.getIntCoordinate();

                        // optimize the current point and the set of regimes
                        ArrayXd  solution = m_pOptimize->stepOptimize(m_gridExtendPreviousStep, iReg, pointCoord, p_condExp, phiInExtended);
                        // copie solution
                        (*phiOut[iReg]).col(iterGridPoint.getCount()) = solution;
                        iterGridPoint.nextInc(nbThreads);
                    }
#ifdef _OPENMP
                });
#endif
            }
#ifdef _OPENMP
            excep.rethrow();
#endif
        }
    }
    return phiOut;
}

void TransitionStepRegressionSwitchDist::dumpContinuationValues(shared_ptr<gs::BinaryFileArchive> p_ar, const string &p_name, const int &p_iStep,
        const vector< shared_ptr< ArrayXXd > > &p_phiInPrev,
        const  shared_ptr<BaseRegression>    &p_condExp) const
{
    string stepString = boost::lexical_cast<string>(p_iStep) ;
    vector< Array< bool, Dynamic, 1> > dimToSplit = m_pOptimize->getDimensionToSplit();

    if (m_world.rank() == 0)
        // store regressor
        *p_ar <<  gs::Record(dynamic_cast<const BaseRegression &>(*p_condExp), "regressor", stepString.c_str()) ;

    for (size_t iReg = 0; iReg <  m_paral.size(); ++iReg)
    {
        ArrayXi initialDimensionPrev  = m_pGridPrevious[iReg]->getDimensions();
        ArrayXi initialDimension  =   m_pGridCurrent[iReg]->getDimensions();
        // utilitary
        SubMeshIntCoord  gridOnProc0Prev(initialDimensionPrev.size());
        for (int id = 0; id < initialDimensionPrev.size(); ++id)
        {
            gridOnProc0Prev(id)[0] = 0 ;
            gridOnProc0Prev(id)[1] = initialDimensionPrev(id) ;
        }
        ArrayXi splittingRatioPrev = paraOptimalSplitting(initialDimensionPrev, dimToSplit[iReg], m_world);
        ParallelComputeGridSplitting  paralObjectPrev(initialDimensionPrev, splittingRatioPrev, m_world);
        ArrayXXd reconstructedArray ;
        if (m_world.rank() < m_paral[iReg]->getNbProcessorUsedPrev())
            reconstructedArray = paralObjectPrev.reconstruct(*p_phiInPrev[iReg], gridOnProc0Prev);
        if (m_world.rank() == 0)
        {
            ArrayXXd transposeCont = reconstructedArray.transpose();
            ArrayXXd basisValues = p_condExp->getCoordBasisFunctionMultiple(transposeCont).transpose();
            *p_ar << gs::Record(basisValues, (p_name + "basisValues").c_str(), stepString.c_str()) ;
        }
    }
    if (m_world.rank() == 0)
        p_ar->flush() ; // necessary for python mapping
    m_world.barrier() ; // onlyt to prevent the reading in simualtion before the end of writting
}


#endif
