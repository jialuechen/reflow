
#ifdef USE_MPI
#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include "reflow/core/parallelism/all_gatherv.hpp"
#endif
#include "reflow/dp/SimulateStepRegressionControl.h"

using namespace std;
using namespace reflow;
using namespace Eigen;

SimulateStepRegressionControl::SimulateStepRegressionControl(gs::BinaryFileArchive &p_ar,  const int &p_iStep,  const string &p_nameCont,
        const   shared_ptr<SpaceGrid> &p_pGridFollowing,
        const  shared_ptr<reflow::OptimizerBaseInterp > &p_pOptimize
#ifdef USE_MPI
        ,  const boost::mpi::communicator &p_world
#endif
                                                            ):
    m_pGridFollowing(p_pGridFollowing),
    m_pOptimize(p_pOptimize)
#ifdef USE_MPI
    , m_world(p_world)
#endif
{
    gs::Reference< vector<GridAndRegressedValue> > (p_ar, (p_nameCont + "Control").c_str(), boost::lexical_cast<string>(p_iStep).c_str()).restore(0, &m_control);
}

void SimulateStepRegressionControl::oneStep(vector<StateWithStocks > &p_statevector, ArrayXXd  &p_phiInOut) const
{
#ifdef USE_MPI
    int rank = m_world.rank();
    int nbProc = m_world.size();    // parallelism
    int nsimPProc = (int)(p_statevector.size() / nbProc);
    int nRestSim = p_statevector.size() % nbProc;
    int iFirstSim = rank * nsimPProc + (rank < nRestSim ? rank : nRestSim);
    int iLastSim  = iFirstSim + nsimPProc + (rank < nRestSim ? 1 : 0);
    ArrayXXd stockPerSim(m_pGridFollowing->getDimension(), iLastSim - iFirstSim);
    // nows store regimes
    ArrayXi regimePerSim(iLastSim - iFirstSim);
    ArrayXXd valueFunctionPerSim(p_phiInOut.rows(), iLastSim - iFirstSim);
    // spread calculations on processors
    int  is = 0 ;
#ifdef _OPENMP
    #pragma omp parallel for  private(is)
#endif
    for (is = iFirstSim; is <  iLastSim; ++is)
    {
        m_pOptimize->stepSimulateControl(m_pGridFollowing, m_control, p_statevector[is], p_phiInOut.col(is));
        // store for broadcast
        stockPerSim.col(is - iFirstSim) = p_statevector[is].getPtStock();
        regimePerSim(is - iFirstSim) = p_statevector[is].getRegime();
        if (valueFunctionPerSim.size() > 0)
            valueFunctionPerSim.col(is - iFirstSim) = p_phiInOut.col(is);
    }
    // broadcast
    ArrayXXd stockAllSim(m_pGridFollowing->getDimension(), p_statevector.size());
    boost::mpi::all_gatherv<double>(m_world, stockPerSim.data(), stockPerSim.size(), stockAllSim.data());
    ArrayXi regimeAllSim(p_statevector.size());
    boost::mpi::all_gatherv<int>(m_world, regimePerSim.data(), regimePerSim.size(), regimeAllSim);
    boost::mpi::all_gatherv<double>(m_world, valueFunctionPerSim.data(), valueFunctionPerSim.size(), p_phiInOut.data());
    // update results
    for (size_t iis = 0; iis < p_statevector.size(); ++iis)
    {
        p_statevector[iis].setPtStock(stockAllSim.col(iis));
        p_statevector[iis].setRegime(regimeAllSim(iis));
    }
#else
    int is = 0 ;
#ifdef _OPENMP
    #pragma omp parallel for  private(is)
#endif
    for (is = 0; is <  static_cast<int>(p_statevector.size()); ++is)
    {
        m_pOptimize->stepSimulateControl(m_pGridFollowing, m_control, p_statevector[is], p_phiInOut.col(is));
    }

#endif

}
