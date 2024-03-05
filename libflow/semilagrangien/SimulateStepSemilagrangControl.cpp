// Copyright (C) 2016 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#ifdef USE_MPI
#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include "libflow/core/parallelism/all_gatherv.hpp"
#endif
#include "libflow/semilagrangien/SimulateStepSemilagrangControl.h"
#include "libflow/core/utils/eigenGeners.h"
#include "libflow/core/grids/FullGridGeners.h"
#include "libflow/core/grids/RegularSpaceGridGeners.h"
#include "libflow/core/grids/GeneralSpaceGridGeners.h"
#include "libflow/core/grids/SparseSpaceGridNoBoundGeners.h"
#include "libflow/core/grids/SparseSpaceGridBoundGeners.h"

using namespace std;
using namespace libflow;
using namespace Eigen;

SimulateStepSemilagrangControl::SimulateStepSemilagrangControl(gs::BinaryFileArchive &p_ar,  const int &p_iStep,  const string &p_name,
        const  shared_ptr<SpaceGrid>   &p_gridCur, const shared_ptr<SpaceGrid>   &p_gridNext,
        const  shared_ptr<libflow::OptimizerSLBase > &p_pOptimize
#ifdef USE_MPI
        , const boost::mpi::communicator &p_world
#endif
                                                              ):
    m_gridNext(p_gridNext), m_controlInterp(p_pOptimize->getNbControl()),  m_pOptimize(p_pOptimize)
#ifdef USE_MPI
    , m_world(p_world)
#endif
{
    string valDump = p_name + "Control";
    vector< shared_ptr<ArrayXd > > control;
    gs::Reference<decltype(control)>(p_ar, valDump.c_str(), boost::lexical_cast<string>(p_iStep).c_str()).restore(0, &control);
    for (int iCont = 0; iCont < p_pOptimize->getNbControl(); ++iCont)
        m_controlInterp[iCont] = p_gridCur->createInterpolatorSpectral(*control[iCont]);

}

void SimulateStepSemilagrangControl::oneStep(const ArrayXXd   &p_gaussian, ArrayXXd &p_statevector, ArrayXi &p_iReg, ArrayXXd  &p_phiInOut) const
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
        m_pOptimize->stepSimulateControl(*m_gridNext, m_controlInterp, p_statevector.col(is), p_iReg(is), p_gaussian.col(is), p_phiInOut.col(is));
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
        m_pOptimize->stepSimulateControl(*m_gridNext, m_controlInterp, p_statevector.col(is), p_iReg(is), p_gaussian.col(is), p_phiInOut.col(is));
    }
#endif
}
