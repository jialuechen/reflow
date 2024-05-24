
#ifdef USE_MPI
#include "geners/Reference.hh"
#include "geners/vectorIO.hh"
#include "reflow/dp/SimulateStepRegressionControlDist.h"
#include "reflow/regression/GridAndRegressedValue.h"
#include "reflow/regression/GridAndRegressedValueGeners.h"
#include "reflow/core/utils/eigenGeners.h"
#include "reflow/core/utils/primeNumber.h"
#include "reflow/core/utils/NodeParticleSplitting.h"
#include "reflow/core/utils/types.h"
#include "reflow/core/utils/constant.h"
#include "reflow/core/utils/comparisonUtils.h"
#include "reflow/core/parallelism/all_gatherv.hpp"

using namespace std;
using namespace reflow;
using namespace Eigen;


SimulateStepRegressionControlDist::SimulateStepRegressionControlDist(gs::BinaryFileArchive &p_ar,  const int &p_iStep,  const string &p_nameCont,
        const   shared_ptr<FullGrid> &p_pGridCurrent,  const   shared_ptr<FullGrid> &p_pGridFollowing, const  shared_ptr<OptimizerBaseInterp > &p_pOptimize,
        const bool &p_bOneFile, const boost::mpi::communicator &p_world): m_pGridCurrent(p_pGridCurrent), m_pGridFollowing(p_pGridFollowing),
    m_pOptimize(p_pOptimize), m_bOneFile(p_bOneFile)	, m_world(p_world)
{
    string stepString = boost::lexical_cast<string>(p_iStep);
    if (m_bOneFile)
    {
        gs::Reference< vector< GridAndRegressedValue > >(p_ar, (p_nameCont + "Control").c_str(), stepString.c_str()).restore(0, &m_control);
    }
    else
    {
        vector<int> initialVecDimension;
        gs::Reference< 	vector<int> >(p_ar, "initialSizeOfMesh", stepString.c_str()).restore(0, &initialVecDimension);
        Map<const ArrayXi > initialDimension(initialVecDimension.data(), initialVecDimension.size());
        ArrayXi splittingRatio = paraOptimalSplitting(initialDimension, m_pOptimize->getDimensionToSplit(), p_world);
        m_parall =  make_shared<ParallelComputeGridSplitting>(initialDimension, splittingRatio, p_world);
        gs::Reference< vector< ArrayXXd > >(p_ar, (p_nameCont + "Control").c_str(), stepString.c_str()).restore(0, & m_contValue);
        m_regressor = gs::Reference< BaseRegression >(p_ar, "regressor", stepString.c_str()).get(0);

    }
}

void SimulateStepRegressionControlDist::oneStep(vector<StateWithStocks > &p_statevector, ArrayXXd  &p_phiInOut) const
{
    unique_ptr<ArrayXXd >  particles(new ArrayXXd(p_statevector.size(), m_pGridCurrent->getDimension()));
    for (size_t is = 0; is < p_statevector.size(); ++is)
        for (int isto = 0; isto < m_pGridCurrent->getDimension(); ++isto)
            (*particles)(is, isto) = p_statevector[is].getPtStock()(isto);

    ArrayXi splittingRatio = ArrayXi::Constant(m_pGridCurrent->getDimension(), 1);
    vector<int> prime = primeNumber(m_world.size());
    int idim = 0; // roll the dimensions
    for (size_t i = 0; i < prime.size(); ++i)
    {
        splittingRatio(idim % m_pGridCurrent->getDimension()) *= prime[i];
        idim += 1;
    }
    // create object to split particules on processor
    NodeParticleSplitting splitparticle(particles, splittingRatio);
    // each simulation to a cell
    ArrayXi nCell(p_statevector.size());
    Array<  array<double, 2 >, Dynamic, Dynamic > meshToCoord(m_pGridCurrent->getDimension(), m_world.size());
    splitparticle.simToCell(nCell, meshToCoord);
    // simulation for current processor
    vector< int > simCurrentProc;
    simCurrentProc.reserve(2 * p_statevector.size() / m_world.size()) ; // use a margin
    for (size_t is = 0; is <  p_statevector.size(); ++is)
        if (nCell(is) == m_world.rank())
            simCurrentProc.push_back(is);
    // nows store stocks
    ArrayXd stockPerSim(m_pGridCurrent->getDimension()*simCurrentProc.size());
    // store value functions
    ArrayXXd valueFunctionPerSim(m_pOptimize->getSimuFuncSize(), simCurrentProc.size());
    if (m_bOneFile)
    {
        // spread calculations on processors
        for (size_t is = 0; is <  simCurrentProc.size(); ++is)
        {
            int simuNumber = simCurrentProc[is];
            m_pOptimize->stepSimulateControl(m_pGridFollowing, m_control, p_statevector[simuNumber], p_phiInOut.col(simuNumber));
            // store for broadcast
            stockPerSim.segment(is * m_pGridCurrent->getDimension(), m_pGridCurrent->getDimension()) = p_statevector[simuNumber].getPtStock();
            if (valueFunctionPerSim.size() > 0)
                valueFunctionPerSim.col(is) = p_phiInOut.col(simuNumber);
        }
    }
    else
    {
        // calculate  grids for each processor
        ArrayXd xCapMin(m_pGridCurrent->getDimension()), xCapMax(m_pGridCurrent->getDimension());
        std::vector <std::array< double, 2>  > extrem = m_pGridCurrent->getExtremeValues();
        for (int id = 0; id < m_pGridCurrent->getDimension(); ++id)
        {
            xCapMin(id) = std::max(extrem[id][0], meshToCoord(id, m_world.rank())[0] - small);
            xCapMax(id) =  std::min(extrem[id][1], meshToCoord(id, m_world.rank())[1] + small);
        }
        ArrayXi  iCapMin =  m_pGridCurrent->lowerPositionCoord(xCapMin);
        ArrayXi  iCapMax =  m_pGridCurrent->upperPositionCoord(xCapMax) + 1; // last is excluded
        SubMeshIntCoord retGrid(m_pGridCurrent->getDimension());
        for (int id = 0; id <  m_pGridCurrent->getDimension(); ++id)
        {
            retGrid(id)[0] = iCapMin(id);
            retGrid(id)[1] = iCapMax(id);
        }
        // extend  values
        shared_ptr<FullGrid> gridExtended = m_pGridCurrent->getSubGrid(retGrid);
        vector< GridAndRegressedValue >  controlExtended(m_pOptimize->getNbControl());
        for (int iCont = 0; iCont < m_pOptimize->getNbControl(); ++iCont)
        {
            ArrayXXd valuesExtended = m_parall->reconstructAll<double>(m_contValue[iCont], retGrid);
            controlExtended[iCont] = GridAndRegressedValue(gridExtended, m_regressor);
            // affect
            controlExtended[iCont].setRegressedValues(valuesExtended);
        }
        // spread calculations on processors
        for (size_t is = 0; is <  simCurrentProc.size(); ++is)
        {
            int simuNumber = simCurrentProc[is];
            m_pOptimize->stepSimulateControl(m_pGridFollowing, controlExtended,  p_statevector[simuNumber], p_phiInOut.col(simuNumber));
            // store for broadcast
            stockPerSim.segment(is * m_pGridCurrent->getDimension(), m_pGridCurrent->getDimension()) = p_statevector[simuNumber].getPtStock();
            if (valueFunctionPerSim.size() > 0)
                valueFunctionPerSim.col(is) = p_phiInOut.col(simuNumber);
        }
    }
    // broadcast
    vector<double> stockAllSim;
    boost::mpi::all_gatherv<double>(m_world, stockPerSim.data(), stockPerSim.size(), stockAllSim);
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
        Map<const ArrayXd >  ptStock(&stockAllSim[is * m_pGridCurrent->getDimension()], m_pGridCurrent->getDimension());
        p_statevector[simAllProc[is]].setPtStock(ptStock);
    }

}
#endif
