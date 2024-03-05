// Copyright (C) 2016 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#ifdef USE_MPI
#include <functional>
#include <memory>
#ifdef _OPENMP
#include <omp.h>
#include "libflow/core/utils/OpenmpException.h"
#endif
#include <boost/mpi.hpp>
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "geners/Record.hh"
#include "geners/vectorIO.hh"
#include "libflow/core/parallelism/ParallelComputeGridSplitting.h"
#include "libflow/core/utils/primeNumber.h"
#include "libflow/core/grids/GridIterator.h"
#include "libflow/core/utils/eigenGeners.h"
#include "libflow/regression/GridAndRegressedValue.h"
#include "libflow/regression/GridAndRegressedValueGeners.h"
#include "libflow/dp/TransitionStepDPDist.h"
#include "libflow/core/parallelism/GridReach.h"


using namespace  libflow;
using namespace  Eigen;
using namespace  std;



TransitionStepDPDist::TransitionStepDPDist(const  shared_ptr<FullGrid> &p_pGridCurrent,
        const  shared_ptr<FullGrid> &p_pGridPrevious,
        const  std::shared_ptr<BaseRegression> &p_regressorCurrent,
        const  std::shared_ptr<BaseRegression> &p_regressorPrevious,
        const  shared_ptr<OptimizerNoRegressionDPBase > &p_pOptimize,
        const boost::mpi::communicator &p_world):
    TransitionStepBaseDist(p_pGridCurrent, p_pGridPrevious, p_pOptimize, p_world), m_regressorPrevious(p_regressorPrevious), m_regressorCurrent(p_regressorCurrent)
{}

std::pair< shared_ptr< vector<  Eigen::ArrayXXd > >, shared_ptr< vector<  Eigen::ArrayXXd > >  > TransitionStepDPDist::oneStep(const vector<  Eigen::ArrayXXd > &p_phiIn) const
{
    // number of regimes at current time
    int nbRegimes = m_pOptimize->getNbRegime();
    shared_ptr< vector< ArrayXXd > >  phiOut = make_shared< vector< ArrayXXd > >(nbRegimes);
    int nbControl =  m_pOptimize->getNbControl();
    shared_ptr< vector< ArrayXXd  > >  controlOut = make_shared< vector< ArrayXXd > >(nbControl);
    // only if the processor is working
    vector < ArrayXXd >  phiInExtended(p_phiIn.size());
    // Organize the data splitting : spread the incoming values on an extended grid
    for (size_t iReg  = 0; iReg < p_phiIn.size() ; ++iReg)
    {
        // utilitary
        ArrayXXd emptyArray;
        if (p_phiIn[iReg].size() > 0)
            phiInExtended[iReg] = m_paral->runOneStep(p_phiIn[iReg]) ;
        else
            phiInExtended[iReg] = m_paral->runOneStep(emptyArray) ;
    }
    if (m_gridCurrentProc->getNbPoints() > 0)
    {
        //  allocate for solution
        for (int iReg = 0; iReg < nbRegimes; ++iReg)
            (*phiOut)[iReg] = ArrayXXd(m_regressorCurrent->getNumberOfFunction(), m_gridCurrentProc->getNbPoints());
        for (int iCont = 0; iCont < nbControl; ++iCont)
            (*controlOut)[iCont] = ArrayXXd(m_regressorCurrent->getNumberOfFunction(), m_gridCurrentProc->getNbPoints());

        //  create continuation values on extended grid
        vector< GridAndRegressedValue > valGridReg(p_phiIn.size());
        for (size_t iReg = 0; iReg < p_phiIn.size(); ++iReg)
        {
            valGridReg[iReg] = GridAndRegressedValue(m_gridExtendPreviousStep, m_regressorPrevious);
            valGridReg[iReg].setRegressedValues(phiInExtended[iReg]);
        }


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
                    std::pair< ArrayXXd, ArrayXXd>  solutionAndControl = static_pointer_cast<OptimizerNoRegressionDPBase>(m_pOptimize)->stepOptimize(pointCoord, valGridReg, m_regressorCurrent);
                    // copie solution
                    for (int iReg = 0; iReg < nbRegimes; ++iReg)
                        (*phiOut)[iReg].col(iterGridPoint->getCount()) = m_regressorCurrent->getCoordBasisFunction(solutionAndControl.first.col(iReg));
                    for (int iCont = 0; iCont < nbControl; ++iCont)
                        (*controlOut)[iCont].col(iterGridPoint->getCount()) = m_regressorCurrent->getCoordBasisFunction(solutionAndControl.second.col(iCont));
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

void TransitionStepDPDist::dumpValues(std::shared_ptr<gs::BinaryFileArchive> p_ar, const string &p_name, const int &p_iStep,
                                      const vector< ArrayXXd   > &p_control,
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
        *p_ar <<  gs::Record(dynamic_cast<const BaseRegression &>(*m_regressorCurrent), "regressor", stepString.c_str()) ;
        // store control
        *p_ar <<  gs::Record(p_control, (p_name + "Control").c_str(), stepString.c_str()) ;
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
        vector< GridAndRegressedValue > control(p_control.size());
        for (size_t iCont = 0; iCont < p_control.size(); ++iCont)
        {
            ArrayXXd reconstructedArray ;
            if (m_world.rank() < m_paral->getNbProcessorUsed())
                reconstructedArray = paralObject.reconstruct(p_control[iCont], gridOnProc0);
            if (m_world.rank() == 0)
            {
                control[iCont] = GridAndRegressedValue(m_pGridCurrent, m_regressorCurrent);
                control[iCont].setRegressedValues(reconstructedArray);
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
