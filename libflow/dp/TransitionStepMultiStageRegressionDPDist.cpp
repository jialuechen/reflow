

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
#include "libflow/regression/ContinuationValue.h"
#include "libflow/regression/ContinuationValueGeners.h"
#include "libflow/regression/GridAndRegressedValue.h"
#include "libflow/regression/GridAndRegressedValueGeners.h"
#include "libflow/dp/TransitionStepMultiStageRegressionDPDist.h"
#include "libflow/core/parallelism/GridReach.h"


using namespace  libflow;
using namespace  Eigen;
using namespace  std;



TransitionStepMultiStageRegressionDPDist::TransitionStepMultiStageRegressionDPDist(const  shared_ptr<FullGrid> &p_pGridCurrent,
        const  shared_ptr<FullGrid> &p_pGridPrevious,
        const  shared_ptr<OptimizerMultiStageDPBase > &p_pOptimize,
        const boost::mpi::communicator &p_world):
    TransitionStepBaseDist(p_pGridCurrent, p_pGridPrevious, p_pOptimize, p_world), m_bDump(false)
{
    calParalDet();
}
//std::static_pointer_cast<OptimizerBase>(
TransitionStepMultiStageRegressionDPDist::TransitionStepMultiStageRegressionDPDist(const  shared_ptr<FullGrid> &p_pGridCurrent,
        const  shared_ptr<FullGrid> &p_pGridPrevious,
        const  shared_ptr<OptimizerMultiStageDPBase > &p_pOptimize,
        const  std::shared_ptr<gs::BinaryFileArchive>   &p_arGen,
        const  std::string &p_nameDump,
        const  bool   &p_bOneFileDet,
        const boost::mpi::communicator &p_world):
    TransitionStepBaseDist(p_pGridCurrent, p_pGridPrevious, p_pOptimize, p_world),
    m_arGen(p_arGen), m_nameDump(p_nameDump), m_bOneFileDet(p_bOneFileDet), m_bDump(true)
{
    calParalDet();
}


void TransitionStepMultiStageRegressionDPDist::calParalDet()
{
    // initial and previous dimensions
    ArrayXi initialDimension   = m_pGridCurrent->getDimensions();
    // organize the hypercube splitting for parallel
    ArrayXi splittingRatio = paraOptimalSplitting(initialDimension, m_pOptimize->getDimensionToSplit(), m_world);
    // cone value
    function < SubMeshIntCoord(const SubMeshIntCoord &) > fMesh = GridReach<OptimizerBase>(m_pGridCurrent, m_pGridCurrent, m_pOptimize);
    // ParallelComputeGridsSplitting objects
    m_paralDet = make_shared<ParallelComputeGridSplitting>(initialDimension, initialDimension, fMesh, splittingRatio, splittingRatio, m_world);
    // only if the grid is not empty
    if (m_gridCurrentProc->getNbPoints() > 0)
    {
        // get back grid extended on previous step
        Array<  array<int, 2 >, Dynamic, 1 > gridLocalExtended = m_paralDet->getExtendedGridProcOldGrid();
        m_gridExtendCurrentStep =  m_pGridCurrent->getSubGrid(gridLocalExtended);
    }
}


void TransitionStepMultiStageRegressionDPDist::oneStageInStep(const vector< shared_ptr< ArrayXXd > > &p_phiIn,
        vector< shared_ptr< ArrayXXd > > &p_phiOut,
        const shared_ptr< BaseRegression>  &p_condExp,
        const std::shared_ptr<ParallelComputeGridSplitting> &p_paralStep,
        const shared_ptr<FullGrid> &p_pGridPrevTransExtended) const
{
    // number of deterministic regimes
    int nbDetRegimes =  std::static_pointer_cast<OptimizerMultiStageDPBase>(m_pOptimize)->getNbDetRegime();
    // only if the processor is working
    vector < shared_ptr< ArrayXXd > > phiInExtended(p_phiIn.size());
    // Organize the data splitting : spread the incoming values on an extended grid
    for (size_t iReg  = 0; iReg < p_phiIn.size() ; ++iReg)
    {
        // utilitary
        ArrayXXd emptyArray;
        if (p_phiIn[iReg])
        {
            phiInExtended[iReg] = make_shared< ArrayXXd >(p_paralStep->runOneStep(*p_phiIn[iReg])) ;
        }
        else
            phiInExtended[iReg] = make_shared< ArrayXXd >(p_paralStep->runOneStep(emptyArray)) ;
    }
    if (m_gridCurrentProc->getNbPoints() > 0)
    {
        //  create continuation values on extended grid
        vector<shared_ptr<ContinuationValue> > contVal(p_phiIn.size());
        // stochastic
        for (size_t iReg = 0; iReg < p_phiIn.size(); ++iReg)
            contVal[iReg] = make_shared<ContinuationValue>(p_pGridPrevTransExtended, p_condExp, *phiInExtended[iReg]);


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
                    ArrayXXd solution = std::static_pointer_cast<OptimizerMultiStageDPBase>(m_pOptimize)->stepOptimize(p_pGridPrevTransExtended, pointCoord, contVal, phiInExtended);
                    // copie solution
                    for (int iReg = 0; iReg < nbDetRegimes; ++iReg)
                        (*p_phiOut[iReg]).col(iterGridPoint->getCount()) = solution.col(iReg);
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

}

vector< shared_ptr< ArrayXXd > > TransitionStepMultiStageRegressionDPDist::oneStep(const vector< shared_ptr< ArrayXXd > > &p_phiIn,
        const shared_ptr< BaseRegression>  &p_condExp) const
{
    // number of regimes at current time
    int nbRegimes = m_pOptimize->getNbRegime();
    // number of deterministic regimes
    int nbDetRegimes =  std::static_pointer_cast<OptimizerMultiStageDPBase>(m_pOptimize)->getNbDetRegime();
    vector< shared_ptr< ArrayXXd > >  phiOut(nbDetRegimes);
    if (m_gridCurrentProc->getNbPoints() > 0)
    {
        //  allocate for solution
        for (int iReg = 0; iReg < nbDetRegimes; ++iReg)
            phiOut[iReg] = make_shared< ArrayXXd >(p_condExp->getNbSimul(), m_gridCurrentProc->getNbPoints());
    }
    //  check in debug the coherence
    assert(nbDetRegimes >= nbRegimes);

    shared_ptr< SimulatorMultiStageDPBase > simulator = std::static_pointer_cast<OptimizerMultiStageDPBase>(m_pOptimize)->getSimulator();

    int  nbPeriodsOfCurrentStep = simulator->getNbPeriodsInTransition();

    // set period number in simulator
    simulator->setPeriodInTransition(nbPeriodsOfCurrentStep - 1);

    // optimize for current step
    oneStageInStep(p_phiIn, phiOut, p_condExp, m_paral, m_gridExtendPreviousStep);

    // now iterate on deterministic period
    for (int iPeriod = nbPeriodsOfCurrentStep - 2; iPeriod >= 0; iPeriod--)
    {
        // dump if necessary
        if (m_bDump)
        {
            dumpContinuationDetValues(phiOut, p_condExp, iPeriod);
        }
        // local
        vector< shared_ptr< ArrayXXd > >  phiInLoc(nbDetRegimes);
        if (m_gridCurrentProc->getNbPoints() > 0)
        {
            for (int iReg = 0; iReg < nbDetRegimes; ++iReg)
                phiInLoc[iReg] = make_shared< ArrayXXd >(*phiOut[iReg]);
        }

        // set period number in simulator
        simulator->setPeriodInTransition(iPeriod);

        // optimize for current step
        oneStageInStep(phiInLoc, phiOut, p_condExp, m_paralDet, m_gridExtendCurrentStep);
    }
    // if the number of determinist regime strictly above , juste select the good number
    if (nbDetRegimes > nbRegimes)
    {
        vector< shared_ptr< ArrayXXd > >  phiOutRed(nbRegimes);
        if (m_gridCurrentProc->getNbPoints() > 0)
        {
            for (int iReg = 0; iReg < nbRegimes; ++iReg)
                phiOutRed[iReg] =  phiOut[iReg];
        }
        return phiOutRed;
    }
    else
    {
        return phiOut;
    }

}

void TransitionStepMultiStageRegressionDPDist::dumpContinuationValues(shared_ptr<gs::BinaryFileArchive> p_ar, const string &p_name, const int &p_iStep,
        const vector< shared_ptr< ArrayXXd > > &p_phiInPrev,
        const  shared_ptr<BaseRegression>    &p_condExp,
        const bool &p_bOneFile) const
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
        // store regressor
        *p_ar <<  gs::Record(dynamic_cast<const BaseRegression &>(*p_condExp), (p_name + "regressor").c_str(), stepString.c_str()) ;
        vector<ArrayXXd> regressedValues(p_phiInPrev.size());
        if (m_world.rank() < m_paral->getNbProcessorUsedPrev())
        {
            // regresse the values
            for (size_t iReg = 0; iReg < p_phiInPrev.size(); ++iReg)
            {
                ArrayXXd transposeCont = p_phiInPrev[iReg]->transpose();
                regressedValues[iReg] = p_condExp->getCoordBasisFunctionMultiple(transposeCont).transpose();
            }
        }
        *p_ar <<  gs::Record(regressedValues, (p_name + "Values").c_str(), stepString.c_str()) ;
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
        vector< GridAndRegressedValue> contVal(p_phiInPrev.size());
        for (size_t iReg = 0; iReg < p_phiInPrev.size(); ++iReg)
        {
            ArrayXXd reconstructedArray ;
            if (m_world.rank() < m_paral->getNbProcessorUsedPrev())
                reconstructedArray = paralObjectPrev.reconstruct(*p_phiInPrev[iReg], gridOnProc0Prev);
            if (m_world.rank() == 0)
                contVal[iReg] = GridAndRegressedValue(m_pGridPrevious, p_condExp, reconstructedArray);
        }
        if (m_world.rank() == 0)
        {
            *p_ar << gs::Record(contVal, (p_name + "Values").c_str(), stepString.c_str()) ;
        }
    }
    if (m_world.rank() == 0)
        p_ar->flush() ; // necessary for python mapping
    m_world.barrier() ; // onlyt to prevent the reading in simualtion before the end of writting
}

void TransitionStepMultiStageRegressionDPDist::dumpContinuationDetValues(const vector< shared_ptr< ArrayXXd > > &p_phiInPrev,
        const  shared_ptr<BaseRegression>    &p_condExp,
        const int &p_iPeriod) const
{
    string stepString = boost::lexical_cast<string>(p_iPeriod);
    ArrayXi initialDimension  =   m_pGridCurrent->getDimensions();
    if (!m_bOneFileDet)
    {
        Array<  array<int, 2 >, Dynamic, 1 > gridLocal =	 m_paralDet->getCurrentCalculationGrid();
        shared_ptr<FullGrid>  gridCurrent = m_pGridCurrent->getSubGrid(gridLocal);
        // dump caracteristics of the splitting
        // organize the hypercube splitting for parallel
        vector<int> vecCurrent(initialDimension.data(), initialDimension.data() + initialDimension.size());
        *m_arGen << gs::Record(vecCurrent, "initialSizeOfMeshPrev", stepString.c_str()) ;
        // store regressor
        *m_arGen <<  gs::Record(dynamic_cast<const BaseRegression &>(*p_condExp), (m_nameDump + "regressor").c_str(), stepString.c_str()) ;
        vector<ArrayXXd> regressedValues(p_phiInPrev.size());
        if (m_world.rank() < m_paralDet->getNbProcessorUsedPrev())
        {
            // regresse the values
            for (size_t iReg = 0; iReg < p_phiInPrev.size(); ++iReg)
            {
                ArrayXXd transposeCont = p_phiInPrev[iReg]->transpose();
                regressedValues[iReg] = p_condExp->getCoordBasisFunctionMultiple(transposeCont).transpose();
            }
        }
        *m_arGen <<  gs::Record(regressedValues, (m_nameDump + "Values").c_str(), stepString.c_str()) ;
    }
    else
    {
        // utilitary
        Array< array<int, 2 >, Dynamic, 1 >  gridOnProc0(initialDimension.size());
        for (int id = 0; id < initialDimension.size(); ++id)
        {
            gridOnProc0(id)[0] = 0 ;
            gridOnProc0(id)[1] = initialDimension(id) ;
        }
        ArrayXi splittingRatio = paraOptimalSplitting(initialDimension, m_pOptimize->getDimensionToSplit(), m_world);
        ParallelComputeGridSplitting  paralObject(initialDimension, splittingRatio, m_world);
        vector< GridAndRegressedValue> contVal(p_phiInPrev.size());
        for (size_t iReg = 0; iReg < p_phiInPrev.size(); ++iReg)
        {
            ArrayXXd reconstructedArray ;
            if (m_world.rank() < m_paralDet->getNbProcessorUsedPrev())
                reconstructedArray = paralObject.reconstruct(*p_phiInPrev[iReg], gridOnProc0);
            if (m_world.rank() == 0)
                contVal[iReg] = GridAndRegressedValue(m_pGridCurrent, p_condExp, reconstructedArray);
        }
        if (m_world.rank() == 0)
        {
            *m_arGen << gs::Record(contVal, (m_nameDump + "Values").c_str(), stepString.c_str()) ;
        }
    }
    if (m_world.rank() == 0)
        m_arGen->flush() ; // necessary for python mapping
    m_world.barrier() ; // onlyt to prevent the reading in simualtion before the end of writting
}


void TransitionStepMultiStageRegressionDPDist::dumpBellmanValues(shared_ptr<gs::BinaryFileArchive> p_ar, const string &p_name, const int &p_iStep,
        const vector< shared_ptr< ArrayXXd > > &p_phiIn,
        const  shared_ptr<BaseRegression>    &p_condExp,
        const bool &p_bOneFile) const
{
    string stepString = boost::lexical_cast<string>(p_iStep) ;
    ArrayXi initialDimension  =   m_pGridCurrent->getDimensions();
    if (!p_bOneFile)
    {
        Array<  array<int, 2 >, Dynamic, 1 > gridLocal =	 m_paral->getCurrentCalculationGrid();
        shared_ptr<FullGrid>  gridCurrent = m_pGridCurrent->getSubGrid(gridLocal);
        // dump caracteristics of the splitting
        // organize the hypercube splitting for parallel
        vector<int> vecCurrent(initialDimension.data(), initialDimension.data() + initialDimension.size());
        *p_ar << gs::Record(vecCurrent, "initialSizeOfMesh", stepString.c_str()) ;
        // store regressor
        *p_ar <<  gs::Record(dynamic_cast<const BaseRegression &>(*p_condExp), (p_name + "regressor").c_str(), stepString.c_str()) ;
        vector<ArrayXXd> regressedValues(p_phiIn.size());
        if (m_world.rank() < m_paral->getNbProcessorUsed())
        {
            // regresse the values
            for (size_t iReg = 0; iReg < p_phiIn.size(); ++iReg)
            {
                ArrayXXd transposeCont = p_phiIn[iReg]->transpose();
                regressedValues[iReg] = p_condExp->getCoordBasisFunctionMultiple(transposeCont).transpose();
            }
        }
        *p_ar <<  gs::Record(regressedValues, (p_name + "Values").c_str(), stepString.c_str()) ;
    }
    else
    {
        // utilitary
        Array< array<int, 2 >, Dynamic, 1 >  gridOnProc0(initialDimension.size());
        for (int id = 0; id < initialDimension.size(); ++id)
        {
            gridOnProc0(id)[0] = 0 ;
            gridOnProc0(id)[1] = initialDimension(id) ;
        }
        ArrayXi splittingRatio = paraOptimalSplitting(initialDimension, m_pOptimize->getDimensionToSplit(), m_world);
        ParallelComputeGridSplitting  paralObject(initialDimension, splittingRatio, m_world);
        vector< GridAndRegressedValue> bellVal(p_phiIn.size());
        for (size_t iReg = 0; iReg < p_phiIn.size(); ++iReg)
        {
            ArrayXXd reconstructedArray ;
            if (m_world.rank() < m_paral->getNbProcessorUsed())
                reconstructedArray = paralObject.reconstruct(*p_phiIn[iReg], gridOnProc0);
            if (m_world.rank() == 0)
                bellVal[iReg] = GridAndRegressedValue(m_pGridCurrent, p_condExp, reconstructedArray);
        }
        if (m_world.rank() == 0)
        {
            *p_ar << gs::Record(bellVal, (p_name + "Values").c_str(), stepString.c_str()) ;
        }
    }
    if (m_world.rank() == 0)
        p_ar->flush() ; // necessary for python mapping
    m_world.barrier() ; // onlyt to prevent the reading in simualtion before the end of writting
}

#endif
