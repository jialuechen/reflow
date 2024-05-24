
#ifdef USE_MPI
#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include "reflow/core/parallelism/all_gatherv.hpp"
#endif
#include "reflow/semilagrangien/SimulateStepSemilagrang.h"
#include "reflow/core/utils/eigenGeners.h"
#include "reflow/core/grids/FullGridGeners.h"
#include "reflow/core/grids/RegularSpaceGridGeners.h"
#include "reflow/core/grids/GeneralSpaceGridGeners.h"
#include "reflow/core/grids/SparseSpaceGridNoBoundGeners.h"
#include "reflow/core/grids/SparseSpaceGridBoundGeners.h"

using namespace std;
using namespace reflow;
using namespace Eigen;

SimulateStepSemilagrang::SimulateStepSemilagrang(gs::BinaryFileArchive &p_ar,  const int &p_iStep,  const string &p_name, const   shared_ptr<SpaceGrid>   &p_gridNext,
        const  shared_ptr<reflow::OptimizerSLBase > &p_pOptimize
#ifdef USE_MPI
        , const boost::mpi::communicator &p_world
#endif
                                                ):
    m_gridNext(p_gridNext), m_specInterp(p_pOptimize->getNbRegime()), m_semiLag(p_pOptimize->getNbRegime()), m_pOptimize(p_pOptimize)
#ifdef USE_MPI
    , m_world(p_world)
#endif
{
    vector< shared_ptr<ArrayXd>  > vecFunctionNext;
    string valDump = p_name + "Val";
    gs::Reference<decltype(vecFunctionNext)>(p_ar, valDump.c_str(), boost::lexical_cast<string>(p_iStep).c_str()).restore(0, &vecFunctionNext);
    // create interpolator and semi lagrangian
    for (size_t ireg = 0; ireg <  vecFunctionNext.size(); ++ireg)
    {
        m_specInterp[ireg] = m_gridNext->createInterpolatorSpectral(*vecFunctionNext[ireg]);
        m_semiLag[ireg] = make_shared<SemiLagrangEspCond>(m_specInterp[ireg], m_gridNext->getExtremeValues(), m_pOptimize->getBModifVol());
    }

}

void SimulateStepSemilagrang::oneStep(const ArrayXXd   &p_gaussian, ArrayXXd &p_statevector, ArrayXi &p_iReg, ArrayXXd  &p_phiInOut) const
{
#ifdef USE_MPI
    int rank = m_world.rank();
    int nbProc = m_world.size();    // parallelism
    int nsimPProc = (int)(p_statevector.cols() / nbProc);
    int nRestSim = p_statevector.cols() % nbProc;
    int iFirstSim = rank * nsimPProc + (rank < nRestSim ? rank : nRestSim);
    int iLastSim  = iFirstSim + nsimPProc + (rank < nRestSim ? 1 : 0);
    ArrayXXd statePerSim(p_statevector.rows(), iLastSim - iFirstSim);
    ArrayXXd valueFunctionPerSim(p_phiInOut.rows(), iLastSim - iFirstSim);
    // spread calculations on processors
    int  is = 0 ;
#ifdef _OPENMP
    #pragma omp parallel for  private(is)
#endif
    for (is = iFirstSim; is <  iLastSim; ++is)
    {
        ArrayXd phiInPt(m_semiLag.size());
        for (size_t iReg = 0; iReg < m_semiLag.size(); ++iReg)
            phiInPt[iReg] = m_specInterp[iReg]->apply(p_statevector.col(is));
        m_pOptimize->stepSimulate(*m_gridNext, m_semiLag, p_statevector.col(is), p_iReg(is), p_gaussian.col(is), phiInPt, p_phiInOut.col(is));
        statePerSim.col(is - iFirstSim) = p_statevector.col(is);
        valueFunctionPerSim.col(is - iFirstSim) = p_phiInOut.col(is);
    }
    boost::mpi::all_gatherv<double>(m_world, statePerSim.data(), statePerSim.size(), p_statevector.data());
    boost::mpi::all_gatherv<double>(m_world, valueFunctionPerSim.data(), valueFunctionPerSim.size(), p_phiInOut.data());
#else
    int is ;
#ifdef _OPENMP
    #pragma omp parallel for  private(is)
#endif
    for (is = 0; is <  p_statevector.cols(); ++is)
    {
        ArrayXd phiInPt(m_semiLag.size());
        for (size_t iReg = 0; iReg < m_semiLag.size(); ++iReg)
            phiInPt[iReg] = m_specInterp[iReg]->apply(p_statevector.col(is));
        m_pOptimize->stepSimulate(*m_gridNext, m_semiLag, p_statevector.col(is), p_iReg(is), p_gaussian.col(is), phiInPt, p_phiInOut.col(is));
    }
#endif
}
