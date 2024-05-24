
#ifdef USE_MPI
#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include "reflow/core/parallelism/all_gatherv.hpp"
#endif
#include "geners/Reference.hh"
#include "geners/vectorIO.hh"
#include "reflow/core/utils/eigenGeners.h"
#include "reflow/core/utils/types.h"
#include "reflow/regression/BaseRegressionGeners.h"
#include "reflow/dp/SimulateStepSwitch.h"

using namespace std;
using namespace reflow;
using namespace Eigen;


SimulateStepSwitch::SimulateStepSwitch(gs::BinaryFileArchive &p_ar,  const int &p_iStep,  const string &p_nameCont,
                                       const   vector< shared_ptr<RegularSpaceIntGrid> > &p_pGridFollowing,
                                       const  shared_ptr<OptimizerSwitchBase > &p_pOptimize
#ifdef USE_MPI
                                       , const boost::mpi::communicator &p_world
#endif
                                      ):
    m_pGridFollowing(p_pGridFollowing),
    m_pOptimize(p_pOptimize),
    m_basisFunc(p_pOptimize->getNbRegime())
#ifdef USE_MPI
    , m_world(p_world)
#endif
{
    string stepString = boost::lexical_cast<string>(p_iStep);
    m_regressor = gs::Reference< BaseRegression >(p_ar, "regressor", stepString.c_str()).get(0);
    for (size_t iReg = 0; iReg <  p_pGridFollowing.size(); ++iReg)
    {
        gs::Reference< ArrayXXd  >(p_ar, (p_nameCont + "basisValues").c_str(), stepString.c_str()).restore(iReg, & m_basisFunc[iReg]);
    }
}

void SimulateStepSwitch::oneStep(vector<StateWithIntState > &p_statevector, ArrayXXd  &p_phiInOut) const
{
#ifdef USE_MPI
    int rank = m_world.rank();
    int nbProc = m_world.size();    // parallelism
    int nsimPProc = (int)(p_statevector.size() / nbProc);
    int nRestSim = p_statevector.size() % nbProc;
    int iFirstSim = rank * nsimPProc + (rank < nRestSim ? rank : nRestSim);
    int iLastSim  = iFirstSim + nsimPProc + (rank < nRestSim ? 1 : 0);
    // maximal  integr state size
    ArrayXi dimSizePerReg(m_pGridFollowing.size());
    int dimMax = 0;
    for (size_t iReg = 0; iReg < m_pGridFollowing.size(); ++iReg)
    {
        dimSizePerReg(iReg) = m_pGridFollowing[iReg]->getDimension();
        dimMax = std::max(dimMax, dimSizePerReg(iReg));
    }
    ArrayXXi statePerSim(dimMax, iLastSim - iFirstSim);
    ArrayXXd valueFunctionPerSim(p_phiInOut.rows(), iLastSim - iFirstSim);
    // spread calculations on processors
    int  is = 0 ;
#ifdef _OPENMP
    #pragma omp parallel for  private(is)
#endif
    for (is = iFirstSim; is <  iLastSim; ++is)
    {
        // get regime for particle
        int iReg = p_statevector[is].getRegime();
        m_pOptimize->stepSimulate(m_pGridFollowing, m_regressor, m_basisFunc, p_statevector[is], p_phiInOut.col(is));
        // store for broadcast
        statePerSim.col(is - iFirstSim).head(dimSizePerReg(iReg)) = p_statevector[is].getPtState();
        if (valueFunctionPerSim.size() > 0)
            valueFunctionPerSim.col(is - iFirstSim) = p_phiInOut.col(is);
    }
    // broadcast
    ArrayXXi stateAllSim(dimMax, p_statevector.size());
    boost::mpi::all_gatherv<int>(m_world, statePerSim.data(), statePerSim.size(), stateAllSim.data());
    boost::mpi::all_gatherv<double>(m_world, valueFunctionPerSim.data(), valueFunctionPerSim.size(), p_phiInOut.data());
    // update results
    for (size_t iis = 0; iis < p_statevector.size(); ++iis)
        p_statevector[iis].setPtState(stateAllSim.col(iis).head(dimSizePerReg(p_statevector[iis].getRegime())));
#else
    int  is = 0 ;
#ifdef _OPENMP
    #pragma omp parallel for  private(is)
#endif
    for (is = 0; is <  static_cast<int>(p_statevector.size()); ++is)
    {
        m_pOptimize->stepSimulate(m_pGridFollowing, m_regressor, m_basisFunc, p_statevector[is], p_phiInOut.col(is));
    }

#endif

}

