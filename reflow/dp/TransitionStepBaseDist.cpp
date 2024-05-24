
#ifdef USE_MPI
#include <functional>
#include <memory>
#include <boost/mpi.hpp>
#include <Eigen/Dense>
#include "reflow/core/parallelism/ParallelComputeGridSplitting.h"
#include "reflow/core/utils/primeNumber.h"
#include "reflow/dp/OptimizerBase.h"
#include "reflow/dp/TransitionStepBaseDist.h"
#include "reflow/core/parallelism/GridReach.h"


using namespace  reflow;
using namespace  Eigen;
using namespace  std;



TransitionStepBaseDist::TransitionStepBaseDist(const  shared_ptr<FullGrid> &p_pGridCurrent,
        const  shared_ptr<FullGrid> &p_pGridPrevious,
        const  shared_ptr<OptimizerBase > &p_pOptimize,
        const boost::mpi::communicator &p_world):
    m_pGridCurrent(p_pGridCurrent), m_pGridPrevious(p_pGridPrevious), m_pOptimize(p_pOptimize), m_world(p_world)
{
    // initial and previous dimensions
    ArrayXi initialDimension   = p_pGridCurrent->getDimensions();
    ArrayXi initialDimensionPrev  = p_pGridPrevious->getDimensions();
    // organize the hypercube splitting for parallel
    ArrayXi splittingRatio = paraOptimalSplitting(initialDimension, m_pOptimize->getDimensionToSplit(), p_world);
    ArrayXi splittingRatioPrev = paraOptimalSplitting(initialDimensionPrev, m_pOptimize->getDimensionToSplit(), p_world);
    // cone value
    function < SubMeshIntCoord(const SubMeshIntCoord &) > fMesh = GridReach<OptimizerBase>(p_pGridCurrent, p_pGridPrevious, p_pOptimize);
    // ParallelComputeGridsSplitting objects
    m_paral = make_shared<ParallelComputeGridSplitting>(initialDimension, initialDimensionPrev, fMesh, splittingRatio, splittingRatioPrev, p_world);
    // get back grid treated by current processor
    Array<  array<int, 2 >, Dynamic, 1 > gridLocal = m_paral->getCurrentCalculationGrid();
    // Construct local sub grid
    m_gridCurrentProc = m_pGridCurrent->getSubGrid(gridLocal);
    // only if the grid is not empty
    if (m_gridCurrentProc->getNbPoints() > 0)
    {
        // get back grid extended on previous step
        Array<  array<int, 2 >, Dynamic, 1 > gridLocalExtended = m_paral->getExtendedGridProcOldGrid();
        m_gridExtendPreviousStep =  m_pGridPrevious->getSubGrid(gridLocalExtended);
    }
}

void  TransitionStepBaseDist::reconstructOnProc0(const vector< shared_ptr< Eigen::ArrayXXd > > &p_phiIn, vector< shared_ptr< Eigen::ArrayXXd > > &p_phiOut)
{
    p_phiOut.resize(p_phiIn.size());
    ArrayXi initialDimension   = m_pGridCurrent->getDimensions();
    Array< array<int, 2 >, Dynamic, 1 >  gridOnProc0(initialDimension.size());
    for (int id = 0; id < initialDimension.size(); ++id)
    {
        gridOnProc0(id)[0] = 0 ;
        gridOnProc0(id)[1] = initialDimension(id) ;
    }
    for (size_t i = 0; i < p_phiIn.size(); ++i)
    {
        if (m_world.rank() < m_paral->getNbProcessorUsed())
            p_phiOut[i] = make_shared<Eigen::ArrayXXd>(m_paral->reconstruct(*p_phiIn[i], gridOnProc0));
    }
}
#endif
