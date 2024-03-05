
#ifdef USE_MPI
#include "geners/Reference.hh"
#include "geners/vectorIO.hh"
#include "libflow/dp/SimulateStepRegressionCutDist.h"
#include "libflow/core/utils/eigenGeners.h"
#include "libflow/core/utils/primeNumber.h"
#include "libflow/core/utils/NodeParticleSplitting.h"
#include "libflow/core/utils/types.h"
#include "libflow/core/parallelism/all_gatherv.hpp"

using namespace std;
using namespace libflow;
using namespace Eigen;


SimulateStepRegressionCutDist::SimulateStepRegressionCutDist(gs::BinaryFileArchive &p_ar,  const int &p_iStep,  const string &p_nameCont,
        const   shared_ptr<FullGrid> &p_pGridFollowing, const  shared_ptr<OptimizerDPCutBase > &p_pOptimize,
        const bool &p_bOneFile, const boost::mpi::communicator &p_world): m_pGridFollowing(p_pGridFollowing),
    m_pOptimize(p_pOptimize), m_contValue((p_pGridFollowing->getDimension() + 1) * p_pOptimize->getNbRegime()),
    m_bOneFile(p_bOneFile)	, m_world(p_world)
{
    string stepString = boost::lexical_cast<string>(p_iStep);
    if (m_bOneFile)
    {
        gs::Reference< vector< ContinuationCuts > >(p_ar, (p_nameCont + "Values").c_str(), stepString.c_str()).restore(0, &m_continuationObj);
    }
    else
    {
        vector<int> initialVecDimensionFollow;
        gs::Reference< 	vector<int> >(p_ar, "initialSizeOfMeshPrev", stepString.c_str()).restore(0, &initialVecDimensionFollow);
        Map<const ArrayXi > initialDimensionFollow(initialVecDimensionFollow.data(), initialVecDimensionFollow.size());
        ArrayXi splittingRatio = paraOptimalSplitting(initialDimensionFollow, m_pOptimize->getDimensionToSplit(), p_world);
        m_parall =  make_shared<ParallelComputeGridSplitting>(initialDimensionFollow, splittingRatio, p_world);
        gs::Reference< vector< ArrayXXd > >(p_ar, (p_nameCont + "Values").c_str(), stepString.c_str()).restore(0, & m_contValue);
        m_regressor = gs::Reference< BaseRegression >(p_ar, "regressor", stepString.c_str()).get(0);

    }
}

void SimulateStepRegressionCutDist::oneStep(vector<StateWithStocks > &p_statevector, ArrayXXd  &p_phiInOut) const
{
    unique_ptr<ArrayXXd >  particles(new ArrayXXd(p_statevector.size(), m_pGridFollowing->getDimension()));
    for (size_t is = 0; is < p_statevector.size(); ++is)
        for (int isto = 0; isto < m_pGridFollowing->getDimension(); ++isto)
            (*particles)(is, isto) = p_statevector[is].getPtStock()(isto);
    ArrayXi splittingRatio = ArrayXi::Constant(m_pGridFollowing->getDimension(), 1);
    vector<int> prime = primeNumber(m_world.size());
    int idim = 0; // roll the dimensions
    for (size_t i = 0; i < prime.size(); ++i)
    {
        splittingRatio(idim % m_pGridFollowing->getDimension()) *= prime[i];
        idim += 1;
    }
    // create object to split particules on processor
    NodeParticleSplitting splitparticle(particles, splittingRatio);
    // each simulation to a cell
    ArrayXi nCell(p_statevector.size());
    Array<  array<double, 2 >, Dynamic, Dynamic > meshToCoord(m_pGridFollowing->getDimension(), m_world.size());
    splitparticle.simToCell(nCell, meshToCoord);
    // simulation for current processor
    vector< int > simCurrentProc;
    simCurrentProc.reserve(2 * p_statevector.size() / m_world.size()) ; // use a margin
    for (size_t is = 0; is <  p_statevector.size(); ++is)
        if (nCell(is) == m_world.rank())
            simCurrentProc.push_back(is);
    // nows store stocks
    ArrayXd stockPerSim(m_pGridFollowing->getDimension()*simCurrentProc.size());
    // nows store regimes
    ArrayXi regimePerSim(simCurrentProc.size());
    // store value functions
    ArrayXXd valueFunctionPerSim(m_pOptimize->getSimuFuncSize(), simCurrentProc.size());
    if (m_bOneFile)
    {
        // spread calculations on processors
        for (size_t is = 0; is <  simCurrentProc.size(); ++is)
        {
            int simuNumber = simCurrentProc[is];
            m_pOptimize->stepSimulate(m_pGridFollowing, m_continuationObj, p_statevector[simuNumber], p_phiInOut.col(simuNumber));
            // store for broadcast
            stockPerSim.segment(is * m_pGridFollowing->getDimension(), m_pGridFollowing->getDimension()) = p_statevector[simuNumber].getPtStock();
            regimePerSim(is) = p_statevector[simuNumber].getRegime();
            if (valueFunctionPerSim.size() > 0)
                valueFunctionPerSim.col(is) = p_phiInOut.col(simuNumber);
        }
    }
    else
    {
        // calculate extended grids
        vector<  array< double, 2>  >  regionByProcessor(splittingRatio.size());
        for (int id = 0; id < splittingRatio.size() ; ++id)
            regionByProcessor[id] = meshToCoord(id, m_world.rank());
        vector<  array< double, 2>  > cone = m_pOptimize->getCone(regionByProcessor);
        // now get subgrid correspond to the cone
        SubMeshIntCoord retGrid(m_pGridFollowing->getDimension());
        vector <array< double, 2>  > extremVal =  m_pGridFollowing->getExtremeValues();
        ArrayXd xCapMin(m_pGridFollowing->getDimension()), xCapMax(m_pGridFollowing->getDimension());
        for (int id = 0; id <  m_pGridFollowing->getDimension(); ++id)
        {
            xCapMin(id)   = max(cone[id][0], extremVal[id][0]);
            xCapMax(id)  = min(cone[id][1], extremVal[id][1]);
        }
        ArrayXi  iCapMin =  m_pGridFollowing->lowerPositionCoord(xCapMin);
        ArrayXi  iCapMax =  m_pGridFollowing->upperPositionCoord(xCapMax) + 1; // last is excluded
        for (int id = 0; id <  m_pGridFollowing->getDimension(); ++id)
        {
            retGrid(id)[0] = iCapMin(id);
            retGrid(id)[1] = iCapMax(id);
        }
        // extend continuation values
        shared_ptr<FullGrid> gridExtended = m_pGridFollowing->getSubGrid(retGrid);
        vector< ContinuationCuts >  continuationExtended(m_pOptimize->getNbRegime());
        int nbCuts = m_pGridFollowing->getDimension() + 1;
        for (int iReg = 0; iReg < m_pOptimize->getNbRegime(); ++iReg)
        {
            // extended cuts
            Array< ArrayXXd, Dynamic, 1>   valuesExtended(nbCuts) ;
            for (int ic = 0; ic < nbCuts; ++ic)
                valuesExtended(ic) = m_parall->reconstructAll<double>(m_contValue[ic + nbCuts * iReg], retGrid);
            continuationExtended[iReg] = ContinuationCuts() ;
            // affect
            continuationExtended[iReg].loadForSimulation(gridExtended, m_regressor, valuesExtended);
        }
        // spread calculations on processors
        for (size_t is = 0; is <  simCurrentProc.size(); ++is)
        {
            int simuNumber = simCurrentProc[is];
            m_pOptimize->stepSimulate(m_pGridFollowing, continuationExtended,  p_statevector[simuNumber], p_phiInOut.col(simuNumber));
            // store for broadcast
            stockPerSim.segment(is * m_pGridFollowing->getDimension(), m_pGridFollowing->getDimension()) = p_statevector[simuNumber].getPtStock();
            regimePerSim(is) = p_statevector[simuNumber].getRegime();
            if (valueFunctionPerSim.size() > 0)
                valueFunctionPerSim.col(is) = p_phiInOut.col(simuNumber);
        }
    }
    // broadcast
    vector<double> stockAllSim;
    boost::mpi::all_gatherv<double>(m_world, stockPerSim.data(), stockPerSim.size(), stockAllSim);
    vector<int> regimeAllSim;
    boost::mpi::all_gatherv<int>(m_world, regimePerSim.data(), regimePerSim.size(), regimeAllSim);
    vector<double> valueFunctionAllSim;
    boost::mpi::all_gatherv<double>(m_world, valueFunctionPerSim.data(), valueFunctionPerSim.size(), valueFunctionAllSim);
    vector<int> simAllProc;
    boost::mpi::all_gatherv<int>(m_world, simCurrentProc.data(), simCurrentProc.size(), simAllProc);
    // update results
    int iis = 0;
    for (size_t is = 0; is < simAllProc.size(); ++is)
    {
        for (int iid = 0; iid < m_pOptimize->getSimuFuncSize(); ++iid)
            p_phiInOut(iid, simAllProc[is]) = valueFunctionAllSim[iis++];
        Map<const ArrayXd >  ptStock(&stockAllSim[is * m_pGridFollowing->getDimension()], m_pGridFollowing->getDimension());
        p_statevector[simAllProc[is]].setPtStock(ptStock);
        p_statevector[simAllProc[is]].setRegime(regimeAllSim[is]);
    }

}
#endif
