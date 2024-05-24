
#ifdef USE_MPI
#include <memory>
#include "geners/Reference.hh"
#include "geners/vectorIO.hh"
#include "reflow/semilagrangien/SemiLagrangEspCond.h"
#include "reflow/semilagrangien/SimulateStepSemilagrangControlDist.h"
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

SimulateStepSemilagrangControlDist::SimulateStepSemilagrangControlDist(gs::BinaryFileArchive &p_ar,  const int &p_iStep,
        const string &p_name,
        const shared_ptr<FullGrid> &p_gridCur,
        const shared_ptr<FullGrid> &p_gridNext,
        const shared_ptr<reflow::OptimizerSLBase > &p_pOptimize,
        const bool &p_bOneFile,
        const boost::mpi::communicator &p_world):
    m_gridCur(p_gridCur), m_gridNext(p_gridNext), m_pOptimize(p_pOptimize),
    m_bOneFile(p_bOneFile), m_world(p_world)
{
    string valDump = p_name + "Control";
    gs::Reference<decltype(m_control)>(p_ar, valDump.c_str(), boost::lexical_cast<string>(p_iStep).c_str()).restore(0, &m_control);
    if (!m_bOneFile)
    {
        vector<int> initialVecDimension;
        gs::Reference< 	vector<int> >(p_ar, "initialSizeOfMesh", boost::lexical_cast<string>(p_iStep).c_str()).restore(0, &initialVecDimension);
        Map<const ArrayXi > initialDimension(initialVecDimension.data(), initialVecDimension.size());
        ArrayXi splittingRatio = paraOptimalSplitting(initialDimension, p_pOptimize->getDimensionToSplit(), m_world);
        m_parall =  make_shared<ParallelComputeGridSplitting>(initialDimension, splittingRatio, m_world);
    }
}

void SimulateStepSemilagrangControlDist::oneStep(const ArrayXXd   &p_gaussian, ArrayXXd &p_statevector, ArrayXi &p_iReg, ArrayXXd  &p_phiInOut) const
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
        vector<std::shared_ptr<InterpolatorSpectral> > controlInterp(m_control.size());
        for (size_t iCont = 0; iCont < m_control.size(); ++iCont)
            controlInterp[iCont] = m_gridCur->createInterpolatorSpectral(*m_control[iCont]);

        // store value function
        int  is ;
#ifdef _OPENMP
        #pragma omp parallel for  private(is)
#endif
        for (is = 0; is <  static_cast<int>(simCurrentProc.size()); ++is)
        {
            int simuNumber = simCurrentProc[is];
            m_pOptimize->stepSimulateControl(*m_gridNext, controlInterp, p_statevector.col(simuNumber), p_iReg(simuNumber), p_gaussian.col(simuNumber),  p_phiInOut.col(simuNumber));
            // copy result per proc for broadcast
            statePerProc.segment(is * m_gridNext->getDimension(), m_gridNext->getDimension()) = p_statevector.col(simuNumber);
            regPerProc(is) = p_iReg(simuNumber);
            phiPerProc.col(is) = p_phiInOut.col(simuNumber);
        }
    }
    else
    {
        // calculate  grids for current processor
        ArrayXd xCapMin(m_gridCur->getDimension()), xCapMax(m_gridCur->getDimension());
        std::vector <std::array< double, 2>  > extrem = m_gridCur->getExtremeValues();
        for (int id = 0; id < m_gridCur->getDimension(); ++id)
        {
            xCapMin(id) = std::max(extrem[0][0], meshToCoord(id, m_world.rank())[0] - small);
            xCapMax(id) =  std::min(extrem[0][1], meshToCoord(id, m_world.rank())[1] + small);
        }
        ArrayXi  iCapMin =  m_gridCur->lowerPositionCoord(xCapMin);
        ArrayXi  iCapMax =  m_gridCur->upperPositionCoord(xCapMax) + 1; // last is excluded        // calculate extended grids
        SubMeshIntCoord retGrid(m_gridCur->getDimension());
        for (int id = 0; id <  m_gridCur->getDimension(); ++id)
        {
            retGrid(id)[0] = iCapMin(id);
            retGrid(id)[1] = iCapMax(id);
        }
        // extend continuation values
        shared_ptr<FullGrid> gridExtended = m_gridCur->getSubGrid(retGrid);
        std::vector< shared_ptr<Eigen::ArrayXd>  > controlExtended(m_control.size());
        for (size_t iCont = 0; iCont < m_control.size(); ++iCont)
        {
            controlExtended[iCont] = make_shared<Eigen::ArrayXd>(m_parall->reconstructAll<double>(*m_control[iCont], retGrid));
        }
        vector<std::shared_ptr<InterpolatorSpectral> > controlInterp(m_control.size());
        for (size_t iCont = 0; iCont < m_control.size(); ++iCont)
            controlInterp[iCont] = gridExtended->createInterpolatorSpectral(*controlExtended[iCont]);

        // store value function
        int is ;
#ifdef _OPENMP
        #pragma omp parallel for  private(is)
#endif
        for (is = 0; is <  static_cast<int>(simCurrentProc.size()); ++is)
        {
            int simuNumber = simCurrentProc[is];
            m_pOptimize->stepSimulateControl(*m_gridNext, controlInterp, p_statevector.col(simuNumber), p_iReg(simuNumber), p_gaussian.col(simuNumber), p_phiInOut.col(simuNumber));
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
