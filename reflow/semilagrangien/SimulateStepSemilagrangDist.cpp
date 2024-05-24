
#ifdef USE_MPI
#include <memory>
#include "geners/Reference.hh"
#include "geners/vectorIO.hh"
#include "reflow/semilagrangien/SemiLagrangEspCond.h"
#include "reflow/semilagrangien/SimulateStepSemilagrangDist.h"
#include "reflow/core/utils/eigenGeners.h"
#include "reflow/core/utils/types.h"
#include "reflow/core/grids/FullGridGeners.h"
#include "reflow/core/grids/RegularSpaceGridGeners.h"
#include "reflow/core/grids/GeneralSpaceGridGeners.h"
#include "reflow/core/grids/SparseSpaceGridNoBoundGeners.h"
#include "reflow/core/grids/SparseSpaceGridBoundGeners.h"
#include "reflow/core/grids/InterpolatorSpectral.h"
#include "reflow/core/utils/primeNumber.h"
#include "reflow/core/utils/NodeParticleSplitting.h"
#include "reflow/core/parallelism/all_gatherv.hpp"

using namespace std;
using namespace reflow;
using namespace Eigen;

SimulateStepSemilagrangDist::SimulateStepSemilagrangDist(gs::BinaryFileArchive &p_ar,  const int &p_iStep,
        const string &p_name,
        const shared_ptr<FullGrid> &p_gridNext,
        const  shared_ptr<reflow::OptimizerSLBase > &p_pOptimize,
        const bool &p_bOneFile,
        const boost::mpi::communicator &p_world):
    m_gridNext(p_gridNext), m_pOptimize(p_pOptimize),
    m_bOneFile(p_bOneFile), m_world(p_world)
{
    string valDump = p_name + "Val";
    gs::Reference<decltype(m_vecFunctionNext)>(p_ar, valDump.c_str(), boost::lexical_cast<string>(p_iStep).c_str()).restore(0, &m_vecFunctionNext);
    if (!m_bOneFile)
    {
        vector<int> initialVecDimensionNext;
        gs::Reference< 	vector<int> >(p_ar, "initialSizeOfMeshPrev", boost::lexical_cast<string>(p_iStep).c_str()).restore(0, &initialVecDimensionNext);
        Map<const ArrayXi > initialDimensionNext(initialVecDimensionNext.data(), initialVecDimensionNext.size());
        ArrayXi splittingRatio = paraOptimalSplitting(initialDimensionNext, m_pOptimize->getDimensionToSplit(), m_world);
        m_parall =  make_shared<ParallelComputeGridSplitting>(initialDimensionNext, splittingRatio, m_world);
    }
}

void SimulateStepSemilagrangDist::oneStep(const ArrayXXd   &p_gaussian, ArrayXXd &p_statevector, ArrayXi &p_iReg, ArrayXXd  &p_phiInOut) const
{
    // spread simulations on processors
    unique_ptr<ArrayXXd >  particles(new ArrayXXd(p_statevector.transpose()));
    ArrayXi splittingRatio = ArrayXi::Constant(m_gridNext->getDimension(), 1);
    vector<int> prime = primeNumber(m_world.size());
    int idim = 0; // roll the dimensions
    for (size_t i = 0; i < prime.size(); ++i)
    {
        splittingRatio(idim % m_gridNext->getDimension()) *= prime[i];
        idim += 1;
    }
    // create object to split particules on processor
    NodeParticleSplitting splitparticle(particles, splittingRatio);
    // each simulation to a cell
    ArrayXi nCell(p_statevector.cols());
    Array<  array<double, 2 >, Dynamic, Dynamic > meshToCoord(m_gridNext->getDimension(), m_world.size());
    splitparticle.simToCell(nCell, meshToCoord);
    // simulation for current processor
    vector< int > simCurrentProc;
    simCurrentProc.reserve(2 * p_statevector.cols() / m_world.size()) ; // use a margin
    for (int is = 0; is <  p_statevector.cols(); ++is)
        if (nCell(is) == m_world.rank())
            simCurrentProc.push_back(is);
    // to store states calculated by current procesor
    ArrayXd statePerProc(m_gridNext->getDimension()*simCurrentProc.size());
    // to store regime and value per proc...
    ArrayXi regPerProc(simCurrentProc.size());
    ArrayXXd phiPerProc(m_pOptimize->getSimuFuncSize(), simCurrentProc.size());
    if (m_bOneFile)
    {
        // create interpolator and semi lagrangian
        vector<std::shared_ptr<InterpolatorSpectral> > specInterp(m_vecFunctionNext.size());
        vector<shared_ptr<SemiLagrangEspCond> > semiLag(m_vecFunctionNext.size()) ;
        for (size_t ireg = 0; ireg <  m_vecFunctionNext.size(); ++ireg)
        {
            specInterp[ireg] = m_gridNext->createInterpolatorSpectral(*m_vecFunctionNext[ireg]);
            semiLag[ireg] = make_shared<SemiLagrangEspCond>(specInterp[ireg], m_gridNext->getExtremeValues(), m_pOptimize->getBModifVol());
        }
        // store value function
        int is ;
#ifdef _OPENMP
        #pragma omp parallel for  private(is)
#endif
        for (is = 0; is <  static_cast<int>(simCurrentProc.size()); ++is)
        {
            int simuNumber = simCurrentProc[is];
            ArrayXd phiInPt(semiLag.size());
            for (size_t iReg = 0; iReg < semiLag.size(); ++iReg)
                phiInPt[iReg] = specInterp[iReg]->apply(p_statevector.col(simuNumber));
            m_pOptimize->stepSimulate(*m_gridNext, semiLag, p_statevector.col(simuNumber), p_iReg(simuNumber), p_gaussian.col(simuNumber), phiInPt, p_phiInOut.col(simuNumber));
            // copy result per proc for broadcast
            statePerProc.segment(is * m_gridNext->getDimension(), m_gridNext->getDimension()) = p_statevector.col(simuNumber);
            regPerProc(is) = p_iReg(simuNumber);
            phiPerProc.col(is) = p_phiInOut.col(simuNumber);
        }
    }
    else
    {
        // calculate extended grids
        std::vector<  std::array< double, 2>  >  regionByProcessor(splittingRatio.size());
        for (int id = 0; id < splittingRatio.size() ; ++id)
            regionByProcessor[id] = meshToCoord(id, m_world.rank());
        std::vector<  std::array< double, 2>  > cone = m_pOptimize->getCone(regionByProcessor);
        // now get subgrid correspond to the cone
        SubMeshIntCoord retGrid(m_gridNext->getDimension());
        std::vector <std::array< double, 2>  > extremVal =  m_gridNext->getExtremeValues();
        ArrayXd xCapMin(m_gridNext->getDimension()), xCapMax(m_gridNext->getDimension());
        for (int id = 0; id <  m_gridNext->getDimension(); ++id)
        {
            xCapMin(id)   = std::max(cone[id][0], extremVal[id][0]);
            xCapMax(id)  = std::min(cone[id][1], extremVal[id][1]);
        }
        ArrayXi  iCapMin =  m_gridNext->lowerPositionCoord(xCapMin);
        ArrayXi  iCapMax =  m_gridNext->upperPositionCoord(xCapMax) + 1; // last is excluded
        for (int id = 0; id <  m_gridNext->getDimension(); ++id)
        {
            retGrid(id)[0] = iCapMin(id);
            retGrid(id)[1] = iCapMax(id);
        }
        // extend continuation values
        shared_ptr<FullGrid> gridExtended = m_gridNext->getSubGrid(retGrid);
        std::vector< shared_ptr<Eigen::ArrayXd>  > vecFuncNextExtended(m_vecFunctionNext.size());
        for (size_t iReg = 0; iReg < m_vecFunctionNext.size(); ++iReg)
        {
            vecFuncNextExtended[iReg] = make_shared<Eigen::ArrayXd>(m_parall->reconstructAll<double>(*m_vecFunctionNext[iReg], retGrid));
        }
        // create interpolator and semi lagrangian
        vector<std::shared_ptr<InterpolatorSpectral> > specInterp(m_vecFunctionNext.size());
        vector<shared_ptr<SemiLagrangEspCond> > semiLag(m_vecFunctionNext.size()) ;
        for (size_t ireg = 0; ireg <   m_vecFunctionNext.size(); ++ireg)
        {
            specInterp[ireg] = gridExtended->createInterpolatorSpectral(*vecFuncNextExtended[ireg]);
            semiLag[ireg] = make_shared<SemiLagrangEspCond>(specInterp[ireg], gridExtended->getExtremeValues(), m_pOptimize->getBModifVol());
        }
        // store value function
        int  is ;
#ifdef _OPENMP
        #pragma omp parallel for  private(is)
#endif
        for (is = 0; is <  static_cast<int>(simCurrentProc.size()); ++is)
        {
            int simuNumber = simCurrentProc[is];
            ArrayXd phiInPt(semiLag.size());
            for (size_t iReg = 0; iReg < semiLag.size(); ++iReg)
                phiInPt[iReg] = specInterp[iReg]->apply(p_statevector.col(simuNumber));
            m_pOptimize->stepSimulate(*m_gridNext, semiLag, p_statevector.col(simuNumber), p_iReg(simuNumber), p_gaussian.col(simuNumber), phiInPt, p_phiInOut.col(simuNumber));
            // copy result per proc for broadcast
            statePerProc.segment(is * m_gridNext->getDimension(), m_gridNext->getDimension()) = p_statevector.col(simuNumber);
            regPerProc(is) = p_iReg(simuNumber);
            phiPerProc.col(is) = p_phiInOut.col(simuNumber);
        }
    }
    // broadcast
    vector<double> stateAllSim;
    boost::mpi::all_gatherv<double>(m_world, statePerProc.data(), statePerProc.size(), stateAllSim);
    vector<int> regAllSim;
    boost::mpi::all_gatherv<int>(m_world, regPerProc.data(), regPerProc.size(), regAllSim);
    vector<double> phiAllSim;
    boost::mpi::all_gatherv<double>(m_world, phiPerProc.data(), phiPerProc.size(), phiAllSim);
    vector<int> simAllProc;
    boost::mpi::all_gatherv<int>(m_world, simCurrentProc.data(), simCurrentProc.size(), simAllProc);

    // update results
    int iis = 0;
    for (size_t is = 0; is < simAllProc.size(); ++is)
    {
        for (int iid = 0; iid < m_pOptimize->getSimuFuncSize(); ++iid)
            p_phiInOut(iid, simAllProc[is]) = phiAllSim[iis++];
        p_iReg(simAllProc[is]) = regAllSim[is];
        Map<const ArrayXd >  ptState(&stateAllSim[is * m_gridNext->getDimension()], m_gridNext->getDimension());
        p_statevector.col(simAllProc[is]) = ptState ;
    }
}
#endif
