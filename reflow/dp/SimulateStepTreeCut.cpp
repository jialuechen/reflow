
#include "reflow/dp/SimulateStepTreeCut.h"
#ifdef USE_MPI
#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include "reflow/core/parallelism/all_gatherv.hpp"
#endif
#include "reflow/tree/ContinuationCutsTreeGeners.h"

using namespace std;
using namespace reflow;
using namespace Eigen;


SimulateStepTreeCut::SimulateStepTreeCut(gs::BinaryFileArchive &p_ar,  const int &p_iStep,  const string &p_nameCont,
        const   shared_ptr<SpaceGrid> &p_pGridFollowing, const  shared_ptr<reflow::OptimizerDPCutTreeBase > &p_pOptimize
#ifdef USE_MPI
        , const boost::mpi::communicator &p_world
#endif
                                        ): m_pGridFollowing(p_pGridFollowing),
    m_pOptimize(p_pOptimize)
#ifdef USE_MPI
    , m_world(p_world)
#endif
{
    gs::Reference< vector< ContinuationCutsTree > > (p_ar, (p_nameCont + "Values").c_str(), boost::lexical_cast<string>(p_iStep).c_str()).restore(0, &m_continuationObj);
}

void SimulateStepTreeCut::oneStep(vector<StateTreeStocks > &p_statevector, ArrayXXd  &p_phiInOut) const
{


#ifdef USE_MPI
    int rank = m_world.rank();
    int nbProc = m_world.size();    // parallelism
    int nsimPProc = (int)(p_statevector.size() / nbProc);
    int nRestSim = p_statevector.size() % nbProc;
    int iFirstSim = rank * nsimPProc + (rank < nRestSim ? rank : nRestSim);
    int iLastSim  = iFirstSim + nsimPProc + (rank < nRestSim ? 1 : 0);
    ArrayXXd stockPerSim(m_pGridFollowing->getDimension(), iLastSim - iFirstSim);
    ArrayXXd valueFunctionPerSim(p_phiInOut.rows(), iLastSim - iFirstSim);
    // spread calculations on processors
    int  is = 0 ;
#ifdef _OPENMP
    #pragma omp parallel for  private(is)
#endif
    for (is = iFirstSim; is <  iLastSim; ++is)
    {
        m_pOptimize->stepSimulate(m_pGridFollowing, m_continuationObj, p_statevector[is], p_phiInOut.col(is));
        // store for broadcast
        stockPerSim.col(is - iFirstSim) = p_statevector[is].getPtStock();
        if (valueFunctionPerSim.size() > 0)
            valueFunctionPerSim.col(is - iFirstSim) = p_phiInOut.col(is);
    }
    // broadcast
    ArrayXXd stockAllSim(m_pGridFollowing->getDimension(), p_statevector.size());
    boost::mpi::all_gatherv<double>(m_world, stockPerSim.data(), stockPerSim.size(), stockAllSim.data());
    boost::mpi::all_gatherv<double>(m_world, valueFunctionPerSim.data(), valueFunctionPerSim.size(), p_phiInOut.data());
    // update results
    for (size_t iis = 0; iis < p_statevector.size(); ++iis)
        p_statevector[iis].setPtStock(stockAllSim.col(iis));
#else
    int  is = 0 ;
#ifdef _OPENMP
    #pragma omp parallel for  private(is)
#endif
    for (is = 0; is <  static_cast<int>(p_statevector.size()); ++is)
    {
        m_pOptimize->stepSimulate(m_pGridFollowing, m_continuationObj, p_statevector[is], p_phiInOut.col(is));
    }

#endif
}
