

#include "libflow/dp/SimulateStepMultiStageRegression.h"
#ifdef USE_MPI
#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include "libflow/core/parallelism/all_gatherv.hpp"
#endif

using namespace std;
using namespace libflow;
using namespace Eigen;


SimulateStepMultiStageRegression::SimulateStepMultiStageRegression(std::shared_ptr<gs::BinaryFileArchive> &p_ar,  const int &p_iStep,  const string &p_nameCont, const std::string &p_nameDetCont,
        const   shared_ptr<SpaceGrid> &p_pGridFollowing, const  shared_ptr<libflow::OptimizerMultiStageDPBase > &p_pOptimize
#ifdef USE_MPI
        , const boost::mpi::communicator &p_world
#endif
                                                                  ): m_pGridFollowing(p_pGridFollowing),
    m_pOptimize(p_pOptimize), m_ar(p_ar), m_iStep(p_iStep), m_nameCont(p_nameCont), m_nameDetCont(p_nameDetCont)
#ifdef USE_MPI
    , m_world(p_world)
#endif
{}

void SimulateStepMultiStageRegression::oneStep(vector<StateWithStocks > &p_statevector, std::vector<ArrayXXd>  &p_phiInOut) const
{
    shared_ptr< SimulatorMultiStageDPBase > simulator = m_pOptimize->getSimulator();
    int  nbPeriodsOfCurrentStep = simulator->getNbPeriodsInTransition();
    for (int iPeriod = 0; iPeriod < nbPeriodsOfCurrentStep ; iPeriod++)
    {
        // set period number in simulator
        simulator->setPeriodInTransition(iPeriod);

        // to store the next grid
        vector< GridAndRegressedValue > contVal;
        shared_ptr< SpaceGrid> gridFollLoc;
        if (iPeriod == (nbPeriodsOfCurrentStep - 1))
        {
            gs::Reference< vector< GridAndRegressedValue > >(*m_ar, (m_nameCont + "Values").c_str(), boost::lexical_cast<string>(m_iStep).c_str()).restore(0, &contVal);
            gridFollLoc = m_pGridFollowing;
        }
        else
        {
            gs::Reference< vector< GridAndRegressedValue > > (*m_ar, m_nameDetCont.c_str(), boost::lexical_cast<string>(iPeriod).c_str()).restore(0, &contVal);
            gridFollLoc = contVal[0].getGrid();
        }

#ifdef USE_MPI
        int rank = m_world.rank();
        int nbProc = m_world.size();    // parallelism
        int nsimPProc = (int)(p_statevector.size() / nbProc);
        int nRestSim = p_statevector.size() % nbProc;
        int iFirstSim = rank * nsimPProc + (rank < nRestSim ? rank : nRestSim);
        int iLastSim  = iFirstSim + nsimPProc + (rank < nRestSim ? 1 : 0);
        ArrayXXd stockPerSim(p_statevector[0].getStockSize(), iLastSim - iFirstSim);
        ArrayXXd valueFunctionPerSim(p_phiInOut[0].rows(), iLastSim - iFirstSim);
        // spread calculations on processors
        int  is = 0 ;
#ifdef _OPENMP
        #pragma omp parallel for  private(is)
#endif
        for (is = iFirstSim; is <  iLastSim; ++is)
        {
            m_pOptimize->stepSimulate(gridFollLoc, contVal, p_statevector[is], p_phiInOut[iPeriod].col(is));
            // store for broadcast
            stockPerSim.col(is - iFirstSim) = p_statevector[is].getPtStock();
            if (valueFunctionPerSim.size() > 0)
                valueFunctionPerSim.col(is - iFirstSim) = p_phiInOut[iPeriod].col(is);
        }
        // broadcast
        ArrayXXd stockAllSim(p_statevector[0].getStockSize(), p_statevector.size());
        boost::mpi::all_gatherv<double>(m_world, stockPerSim.data(), stockPerSim.size(), stockAllSim.data());
        boost::mpi::all_gatherv<double>(m_world, valueFunctionPerSim.data(), valueFunctionPerSim.size(), p_phiInOut[iPeriod].data());
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
            m_pOptimize->stepSimulate(gridFollLoc, contVal, p_statevector[is], p_phiInOut[iPeriod].col(is));
        }
#endif
        //prepare next period
        if (iPeriod < nbPeriodsOfCurrentStep - 1)
        {
            p_phiInOut[iPeriod + 1] = p_phiInOut[iPeriod];
        }
    }
}
