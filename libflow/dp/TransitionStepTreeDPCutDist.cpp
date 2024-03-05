
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
#include "libflow/core/parallelism/ParallelComputeGridSplitting.h"
#include "libflow/core/grids/GridIterator.h"
#include "libflow/core/utils/eigenGeners.h"
#include "libflow/tree/ContinuationCutsTree.h"
#include "libflow/tree/ContinuationCutsTreeGeners.h"
#include "libflow/dp/TransitionStepTreeDPCutDist.h"
#include "libflow/core/parallelism/GridReach.h"


using namespace  libflow;
using namespace  Eigen;
using namespace  std;



TransitionStepTreeDPCutDist::TransitionStepTreeDPCutDist(const  shared_ptr<FullGrid> &p_pGridCurrent,
        const  shared_ptr<FullGrid> &p_pGridPrevious,
        const  shared_ptr<OptimizerDPCutTreeBase > &p_pOptimize,
        const boost::mpi::communicator &p_world): TransitionStepBaseDist(p_pGridCurrent, p_pGridPrevious, p_pOptimize, p_world) {}


vector< shared_ptr< ArrayXXd > >  TransitionStepTreeDPCutDist::oneStep(const vector< shared_ptr< ArrayXXd > > &p_phiIn,
        const shared_ptr< Tree>  &p_condExp) const
{
    // number of regimes at current time
    int nbRegimes = m_pOptimize->getNbRegime();
    vector< shared_ptr< ArrayXXd > >  phiOut(nbRegimes);
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
            phiOut[iReg] = make_shared< ArrayXXd >(p_condExp->getNbNodes() * (m_pGridCurrent->getDimension() + 1), m_gridCurrentProc->getNbPoints());

        //  create continuation values on extended grid
        vector< ContinuationCutsTree > contVal(p_phiIn.size());
        for (size_t iReg = 0; iReg < p_phiIn.size(); ++iReg)
            contVal[iReg] = ContinuationCutsTree(m_gridExtendPreviousStep, p_condExp, *phiInExtended[iReg]);


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

                    // optimize the current point and the set of regimes -> get back cuts per simulation and stock point
                    ArrayXXd  solution = static_pointer_cast<OptimizerDPCutTreeBase>(m_pOptimize)->stepOptimize(m_gridExtendPreviousStep, pointCoord, contVal);
                    // copie solution
                    for (int iReg = 0; iReg < nbRegimes; ++iReg)
                        (*phiOut[iReg]).col(iterGridPoint->getCount()) = solution.col(iReg);
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
    return phiOut;
}

void TransitionStepTreeDPCutDist::dumpContinuationCutsValues(shared_ptr<gs::BinaryFileArchive> p_ar, const string &p_name, const int &p_iStep,
        const vector< shared_ptr< ArrayXXd > > &p_phiInPrev,
        const shared_ptr< Tree>     &p_condExp,
        const bool &p_bOneFile) const
{
    string stepString = boost::lexical_cast<string>(p_iStep) ;
    ArrayXi initialDimensionPrev  = m_pGridPrevious->getDimensions();
    ArrayXi initialDimension  =   m_pGridCurrent->getDimensions();
    // calculate number of nodes in trre
    int nbNodes = p_phiInPrev[0]->rows() / (m_pGridCurrent->getDimension() + 1);

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
        int nbCuts = m_pGridCurrent->getDimension() + 1;
        vector< ArrayXXd >  expValues(nbCuts * p_phiInPrev.size());
        if (m_world.rank() < m_paral->getNbProcessorUsedPrev())
        {
            // regresse the values
            for (size_t iReg = 0; iReg < p_phiInPrev.size(); ++iReg)
            {
                for (int ic  = 0;  ic < nbCuts; ++ic)
                {
                    // size ( nbStock, (nb simul * nbCuts)
                    ArrayXXd transposeCont = p_phiInPrev[iReg]->block(ic * nbNodes, 0, nbNodes, p_phiInPrev[iReg]->cols()). transpose();
                    expValues[ic + nbCuts * iReg] = p_condExp->expCondMultiple(transposeCont).transpose();
                }
            }
            // for cut zero add stock components
            shared_ptr<GridIterator> iterRegGrid = gridPrevious->getGridIterator();
            while (iterRegGrid->isValid())
            {
                // coordinates
                ArrayXd pointCoordReg = iterRegGrid->getCoordinate();
                // point number
                int ipoint =  iterRegGrid->getCount();
                for (size_t iReg = 0; iReg < p_phiInPrev.size(); ++iReg)
                {
                    for (int id = 0 ; id < pointCoordReg.size(); ++id)
                        expValues[nbCuts * iReg].col(ipoint) -= expValues[id + 1 + nbCuts * iReg].col(ipoint) * pointCoordReg(id);
                }
                iterRegGrid->next();
            }
        }
        *p_ar <<  gs::Record(expValues, (p_name + "Values").c_str(), stepString.c_str()) ;
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
        vector< ContinuationCutsTree> contVal(p_phiInPrev.size());
        for (size_t iReg = 0; iReg < p_phiInPrev.size(); ++iReg)
        {
            ArrayXXd reconstructedArray ;
            if (m_world.rank() < m_paral->getNbProcessorUsedPrev())
                reconstructedArray = paralObjectPrev.reconstruct(*p_phiInPrev[iReg], gridOnProc0Prev);
            if (m_world.rank() == 0)
                contVal[iReg] = ContinuationCutsTree(m_pGridPrevious, p_condExp, reconstructedArray);
        }
        if (m_world.rank() == 0)
            *p_ar << gs::Record(contVal, (p_name + "Values").c_str(), stepString.c_str()) ;
    }
    if (m_world.rank() == 0)
        p_ar->flush() ; // necessary for python mapping
    m_world.barrier() ; // onlyt to prevent the reading in simualtion before the end of writting
}

#endif
