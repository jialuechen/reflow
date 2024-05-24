
#ifdef USE_MPI
#include <functional>
#include <memory>
#include <boost/mpi.hpp>
#ifdef _OPENMP
#include <omp.h>
#include "reflow/core/utils/OpenmpException.h"
#endif
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "geners/Record.hh"
#include "geners/vectorIO.hh"
#include "reflow/core/parallelism/ParallelComputeGridSplitting.h"
#include "reflow/core/grids/GridIterator.h"
#include "reflow/core/utils/eigenGeners.h"
#include "reflow/tree/ContinuationValueTree.h"
#include "reflow/tree/ContinuationValueTreeGeners.h"
#include "reflow/tree/GridTreeValue.h"
#include "reflow/tree/GridTreeValueGeners.h"
#include "reflow/dp/TransitionStepTreeDPDist.h"
#include "reflow/core/parallelism/GridReach.h"


using namespace  reflow;
using namespace  Eigen;
using namespace  std;



TransitionStepTreeDPDist::TransitionStepTreeDPDist(const  shared_ptr<FullGrid> &p_pGridCurrent,
        const  shared_ptr<FullGrid> &p_pGridPrevious,
        const  shared_ptr<OptimizerDPTreeBase > &p_pOptimize,
        const boost::mpi::communicator &p_world): TransitionStepBaseDist(p_pGridCurrent, p_pGridPrevious, p_pOptimize,  p_world) {}


std::pair< vector< shared_ptr< ArrayXXd >>, vector<  shared_ptr< ArrayXXd > > > TransitionStepTreeDPDist::oneStep(const vector< shared_ptr< ArrayXXd > > &p_phiIn,
                                        const shared_ptr< Tree>  &p_condExp) const
{
    // number of regimes at current time
    int nbRegimes = m_pOptimize->getNbRegime();
    vector< shared_ptr< ArrayXXd > >  phiOut(nbRegimes);
    int nbControl =  m_pOptimize->getNbControl();
    vector< shared_ptr< ArrayXXd > >  controlOut(nbControl);
    // only if the processor is working
    vector < shared_ptr< ArrayXXd > > phiInExtended(p_phiIn.size());
    // Organize the data splitting : spread the incoming values on an extended grid
    for (size_t iReg  = 0; iReg < p_phiIn.size() ; ++iReg)
    {
        // utilitary
        ArrayXXd emptyArray;
        if (p_phiIn[iReg])
        {
            phiInExtended[iReg] = make_shared< ArrayXXd >(m_paral->runOneStep(*p_phiIn[iReg])) ;
        }
        else
            phiInExtended[iReg] = make_shared< ArrayXXd >(m_paral->runOneStep(emptyArray)) ;
    }
    if (m_gridCurrentProc->getNbPoints() > 0)
    {
        //  allocate for solution
        for (int iReg = 0; iReg < nbRegimes; ++iReg)
            phiOut[iReg] = make_shared< ArrayXXd >(p_condExp->getNbNodes(), m_gridCurrentProc->getNbPoints());
        for (int iCont = 0; iCont < nbControl; ++iCont)
            controlOut[iCont] = make_shared< ArrayXXd >(p_condExp->getNbNodes(), m_gridCurrentProc->getNbPoints());

        //  create continuation values on extended grid
        vector< ContinuationValueTree > contVal(p_phiIn.size());
        for (size_t iReg = 0; iReg < p_phiIn.size(); ++iReg)
            contVal[iReg] = ContinuationValueTree(m_gridExtendPreviousStep, p_condExp, *phiInExtended[iReg]);


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

                    // optimize the current point and the set of regimes
                    std::pair< ArrayXXd, ArrayXXd>  solutionAndControl = static_pointer_cast<OptimizerDPTreeBase>(m_pOptimize)->stepOptimize(m_gridExtendPreviousStep, pointCoord, contVal);
                    // copie solution
                    for (int iReg = 0; iReg < nbRegimes; ++iReg)
                        (*phiOut[iReg]).col(iterGridPoint->getCount()) = solutionAndControl.first.col(iReg);
                    for (int iCont = 0; iCont < nbControl; ++iCont)
                        (*controlOut[iCont]).col(iterGridPoint->getCount()) = solutionAndControl.second.col(iCont);
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

void TransitionStepTreeDPDist::dumpContinuationValues(shared_ptr<gs::BinaryFileArchive> p_ar, const string &p_name, const int &p_iStep,
        const vector< shared_ptr< ArrayXXd > > &p_phiInPrev,  const vector< shared_ptr< ArrayXXd > > &p_control,
        const std::shared_ptr< Tree>      &p_tree, const bool &p_bOneFile) const
{
    string stepString = boost::lexical_cast<string>(p_iStep) ;
    ArrayXi initialDimensionPrev  = m_pGridPrevious->getDimensions();
    ArrayXi initialDimension  =   m_pGridCurrent->getDimensions();
    if (!p_bOneFile)
    {
        Array<  array<int, 2 >, Dynamic, 1 > gridLocalPrev =	 m_paral->getPreviousCalculationGrid();
        shared_ptr<FullGrid>  gridPrevious = m_pGridPrevious->getSubGrid(gridLocalPrev);
        Array<  array<int, 2 >, Dynamic, 1 > gridLocal =	 m_paral->getCurrentCalculationGrid();
        shared_ptr<FullGrid>  gridCurrent = m_pGridCurrent->getSubGrid(gridLocal);
        // dump caracteristics of the splitting
        // organize the hypercube splitting for parallel
        vector<int> vecPrev(initialDimensionPrev.data(), initialDimensionPrev.data() + initialDimensionPrev.size());
        *p_ar << gs::Record(vecPrev, "initialSizeOfMeshPrev", stepString.c_str()) ;
        vector<int> vecCurrent(initialDimension.data(), initialDimension.data() + initialDimension.size());
        *p_ar << gs::Record(vecCurrent, "initialSizeOfMesh", stepString.c_str()) ;
        vector<ArrayXXd  > valExp(p_phiInPrev.size());
        for (size_t it = 0; it < p_phiInPrev.size(); ++it)
        {
            valExp[it] = p_tree->expCondMultiple(p_phiInPrev[it]->transpose()).transpose();
        }
        *p_ar <<  gs::Record(valExp, (p_name + "Values").c_str(), stepString.c_str()) ;
        *p_ar <<  gs::Record(p_control, (p_name + "Control").c_str(), stepString.c_str()) ;
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
        ArrayXi splittingRatioPrev = paraOptimalSplitting(initialDimensionPrev, m_pOptimize->getDimensionToSplit(), m_world);
        ParallelComputeGridSplitting  paralObjectPrev(initialDimensionPrev, splittingRatioPrev, m_world);
        vector< GridTreeValue> contVal(p_phiInPrev.size());
        for (size_t iReg = 0; iReg < p_phiInPrev.size(); ++iReg)
        {
            ArrayXXd reconstructedArray ;
            if (m_world.rank() < m_paral->getNbProcessorUsedPrev())
                reconstructedArray = paralObjectPrev.reconstruct(*p_phiInPrev[iReg], gridOnProc0Prev);
            if (m_world.rank() == 0)
            {
                ArrayXXd reconstructedArrayExp = p_tree->expCondMultiple(reconstructedArray.transpose()).transpose();
                contVal[iReg] = GridTreeValue(m_pGridPrevious, reconstructedArrayExp);
            }
        }
        if (m_world.rank() == 0)
            *p_ar << gs::Record(contVal, (p_name + "Values").c_str(), stepString.c_str()) ;
        // now the control
        Array< array<int, 2 >, Dynamic, 1 >  gridOnProc0(initialDimension.size());
        for (int id = 0; id < initialDimension.size(); ++id)
        {
            gridOnProc0(id)[0] = 0 ;
            gridOnProc0(id)[1] = initialDimension(id) ;
        }
        ArrayXi splittingRatio = paraOptimalSplitting(initialDimension, m_pOptimize->getDimensionToSplit(), m_world);
        ParallelComputeGridSplitting  paralObject(initialDimension, splittingRatio, m_world);
        vector< GridTreeValue > control(p_control.size());
        for (size_t iCont = 0; iCont < p_control.size(); ++iCont)
        {
            ArrayXXd reconstructedArray ;
            if (m_world.rank() < m_paral->getNbProcessorUsed())
                reconstructedArray = paralObject.reconstruct(*p_control[iCont], gridOnProc0);
            if (m_world.rank() == 0)
            {
                control[iCont] = GridTreeValue(m_pGridCurrent,  reconstructedArray);
            }
        }
        if (m_world.rank() == 0)
            *p_ar << gs::Record(control, (p_name + "Control").c_str(), stepString.c_str()) ;
    }
    if (m_world.rank() == 0)
        p_ar->flush() ; // necessary for python mapping
    m_world.barrier() ; // onlyt to prevent the reading in simualtion before the end of writting

}

#endif
