// Copyright (C) 2023 EDF

// This code is published under the GNU Lesser General Public License (GNU LGPL)
#ifdef USE_MPI
#include "geners/Reference.hh"
#include "geners/vectorIO.hh"
#include "libflow/dp/SimulateStepMultiStageRegressionDist.h"
#include "libflow/core/utils/eigenGeners.h"
#include "libflow/core/utils/primeNumber.h"
#include "libflow/core/utils/NodeParticleSplitting.h"
#include "libflow/core/utils/types.h"
#include "libflow/core/parallelism/all_gatherv.hpp"

using namespace std;
using namespace libflow;
using namespace Eigen;


SimulateStepMultiStageRegressionDist::SimulateStepMultiStageRegressionDist(const shared_ptr<gs::BinaryFileArchive> &p_ar,
        const int &p_iStep,  const string &p_nameCont,
        const string &p_nameDetCont,
        const  shared_ptr<FullGrid> &p_pGridCurrent,
        const  shared_ptr<FullGrid> &p_pGridFollowing,
        const  shared_ptr<OptimizerMultiStageDPBase > &p_pOptimize,
        const bool &p_bOneFile,
        const boost::mpi::communicator &p_world):
    m_pGridCurrent(p_pGridCurrent),
    m_pGridFollowing(p_pGridFollowing),
    m_pOptimize(p_pOptimize),
    m_ar(p_ar), m_iStep(p_iStep), m_nameCont(p_nameCont), m_nameDetCont(p_nameDetCont),
    m_bOneFile(p_bOneFile), m_world(p_world)
{
}

vector< GridAndRegressedValue> SimulateStepMultiStageRegressionDist::readContinuationInArchive(const string &p_name, const string &p_stepString)
{
    vector< GridAndRegressedValue>  continuationObj;
    gs::Reference< vector< GridAndRegressedValue> >(*m_ar, (p_name + "Values").c_str(), p_stepString.c_str()).restore(0, &continuationObj);
    return continuationObj;
}

pair< shared_ptr<BaseRegression>, vector< ArrayXXd > >  SimulateStepMultiStageRegressionDist::readRegressedValues(const string &p_name, const string &p_stepString)
{
    vector<int> initialVecDimensionFollow;
    gs::Reference< 	vector<int> >(*m_ar, "initialSizeOfMeshPrev", p_stepString.c_str()).restore(0, &initialVecDimensionFollow);
    Map<const ArrayXi > initialDimensionFollow(initialVecDimensionFollow.data(), initialVecDimensionFollow.size());
    ArrayXi splittingRatio = paraOptimalSplitting(initialDimensionFollow, m_pOptimize->getDimensionToSplit(), m_world);
    m_parall =  make_shared<ParallelComputeGridSplitting>(initialDimensionFollow, splittingRatio, m_world);
    vector< ArrayXXd > contValue;
    gs::Reference< vector< ArrayXXd > >(*m_ar, (p_name + "Values").c_str(), p_stepString.c_str()).restore(0, &contValue);
    shared_ptr<BaseRegression>  regressor = gs::Reference< BaseRegression >(*m_ar, (p_name + "regressor").c_str(), p_stepString.c_str()).get(0);
    return make_pair(regressor, contValue);
}

pair<vector<int>, vector<  array< double, 2>  >  > SimulateStepMultiStageRegressionDist::splitParticleOnProcessor(const vector<StateWithStocks >    &p_statevector, const shared_ptr<FullGrid> &p_gridFollow)const
{
    unique_ptr<ArrayXXd >  particles(new ArrayXXd(p_statevector.size(), p_gridFollow->getDimension()));
    for (size_t is = 0; is < p_statevector.size(); ++is)
        for (int isto = 0; isto < p_gridFollow->getDimension(); ++isto)
            (*particles)(is, isto) = p_statevector[is].getPtStock()(isto);
    ArrayXi splittingRatio = ArrayXi::Constant(p_gridFollow->getDimension(), 1);
    vector<int> prime = primeNumber(m_world.size());
    int idim = 0; // roll the dimensions
    for (size_t i = 0; i < prime.size(); ++i)
    {
        splittingRatio(idim % p_gridFollow->getDimension()) *= prime[i];
        idim += 1;
    }
    // create object to split particules on processor
    NodeParticleSplitting splitparticle(particles, splittingRatio);
    // each simulation to a cell
    ArrayXi nCell(p_statevector.size());
    Array<  array<double, 2 >, Dynamic, Dynamic > meshToCoord(p_gridFollow->getDimension(), m_world.size());
    splitparticle.simToCell(nCell, meshToCoord);
    // simulation for current processor
    vector< int > simCurrentProc;
    simCurrentProc.reserve(2 * p_statevector.size() / m_world.size()) ; // use a margin
    for (size_t is = 0; is <  p_statevector.size(); ++is)
        if (nCell(is) == m_world.rank())
            simCurrentProc.push_back(is);
    vector<  array< double, 2>  >  regionByProcessor(splittingRatio.size());
    for (int id = 0; id < splittingRatio.size() ; ++id)
        regionByProcessor[id] = meshToCoord(id, m_world.rank());
    return make_pair(simCurrentProc, regionByProcessor);
}

SubMeshIntCoord  SimulateStepMultiStageRegressionDist::calculateSubMeshExtended(const shared_ptr<FullGrid> &p_gridFollow, const vector<  array< double, 2>  >   &p_regionByProcessor) const
{
    vector<  array< double, 2>  > cone = m_pOptimize->getCone(p_regionByProcessor);
    // now get subgrid correspond to the cone
    SubMeshIntCoord retGrid(p_gridFollow->getDimension());
    vector <array< double, 2>  > extremVal =  p_gridFollow->getExtremeValues();
    ArrayXd xCapMin(p_gridFollow->getDimension()), xCapMax(p_gridFollow->getDimension());
    for (int id = 0; id <  p_gridFollow->getDimension(); ++id)
    {
        xCapMin(id)   = max(cone[id][0], extremVal[id][0]);
        xCapMax(id)  = min(cone[id][1], extremVal[id][1]);
    }
    ArrayXi  iCapMin =  p_gridFollow->lowerPositionCoord(xCapMin);
    ArrayXi  iCapMax =  p_gridFollow->upperPositionCoord(xCapMax) + 1; // last is excluded
    for (int id = 0; id <  p_gridFollow->getDimension(); ++id)
    {
        retGrid(id)[0] = iCapMin(id);
        retGrid(id)[1] = iCapMax(id);
    }
    return retGrid;
}

void SimulateStepMultiStageRegressionDist::oneStep(vector<StateWithStocks > &p_statevector, vector<ArrayXXd>  &p_phiInOut)
{

    shared_ptr< SimulatorMultiStageDPBase > simulator = m_pOptimize->getSimulator();
    int  nbPeriodsOfCurrentStep = simulator->getNbPeriodsInTransition();

    if (m_bOneFile)
    {
        pair< vector<int>, vector<  array< double, 2>  > > simToProcAndRegion =  splitParticleOnProcessor(p_statevector, m_pGridFollowing);
        size_t nbSimCurProc = simToProcAndRegion.first.size();
        // nows store stocks
        ArrayXd stockPerSim(m_pGridFollowing->getDimension()*nbSimCurProc);
        // nows store regimes
        ArrayXi regimePerSim(nbSimCurProc);
        // store value functions
        ArrayXXd valueFunctionPerSim(m_pOptimize->getSimuFuncSize(), nbSimCurProc);
        // store all the simulation results
        vector<int> simAllProc;

        for (int iPeriod = 0; iPeriod < nbPeriodsOfCurrentStep ; iPeriod++)
        {
            // set period number in simulator
            simulator->setPeriodInTransition(iPeriod);
            // to store the next grid
            vector< GridAndRegressedValue > contVal;
            shared_ptr< FullGrid> gridFollLoc;
            if (iPeriod == (nbPeriodsOfCurrentStep - 1))
            {
                contVal = readContinuationInArchive(m_nameCont, boost::lexical_cast<string>(m_iStep));
                gridFollLoc = m_pGridFollowing;
            }
            else
            {
                contVal = readContinuationInArchive(m_nameDetCont, boost::lexical_cast<string>(iPeriod));
                gridFollLoc = m_pGridCurrent;
            }
            // spread calculations on processors
            for (size_t is = 0; is <  nbSimCurProc; ++is)
            {
                int simuNumber = simToProcAndRegion.first[is];
                m_pOptimize->stepSimulate(gridFollLoc, contVal, p_statevector[simuNumber], p_phiInOut[iPeriod].col(simuNumber));
                // store for broadcast
                stockPerSim.segment(is * gridFollLoc->getDimension(), gridFollLoc->getDimension()) = p_statevector[simuNumber].getPtStock();
                regimePerSim(is) = p_statevector[simuNumber].getRegime();
                if (valueFunctionPerSim.size() > 0)
                    valueFunctionPerSim.col(is) = p_phiInOut[iPeriod].col(simuNumber);
            }
            vector<double> valueFunctionAllSim;
            boost::mpi::all_gatherv<double>(m_world, valueFunctionPerSim.data(), valueFunctionPerSim.size(), valueFunctionAllSim);
            boost::mpi::all_gatherv<int>(m_world, simToProcAndRegion.first.data(), nbSimCurProc, simAllProc);
            int iis = 0;
            for (size_t is = 0; is < simAllProc.size(); ++is)
            {
                for (int iid = 0; iid < m_pOptimize->getSimuFuncSize(); ++iid)
                    p_phiInOut[iPeriod](iid, simAllProc[is]) = valueFunctionAllSim[iis++];
            }
            //prepare next period
            if (iPeriod < nbPeriodsOfCurrentStep - 1)
            {
                p_phiInOut[iPeriod + 1] = p_phiInOut[iPeriod];
            }
        }
        // broadcast
        vector<double> stockAllSim;
        boost::mpi::all_gatherv<double>(m_world, stockPerSim.data(), stockPerSim.size(), stockAllSim);
        vector<int> regimeAllSim;
        boost::mpi::all_gatherv<int>(m_world, regimePerSim.data(), regimePerSim.size(), regimeAllSim);
        // update results
        for (size_t is = 0; is < simAllProc.size(); ++is)
        {
            Map<const ArrayXd >  ptStock(&stockAllSim[is * m_pGridFollowing->getDimension()], m_pGridFollowing->getDimension());
            p_statevector[simAllProc[is]].setPtStock(ptStock);
            p_statevector[simAllProc[is]].setRegime(regimeAllSim[is]);
        }
    }
    else
    {
        for (int iPeriod = 0; iPeriod < nbPeriodsOfCurrentStep ; iPeriod++)
        {
            // set period number in simulator
            simulator->setPeriodInTransition(iPeriod);

            // to store the next grid
            shared_ptr< FullGrid> gridFollLoc;
            int nbRegime = -1;
            pair< shared_ptr<BaseRegression>, vector< ArrayXXd > >  regAndVal;
            if (iPeriod == (nbPeriodsOfCurrentStep - 1))
            {
                gridFollLoc = m_pGridFollowing;
                nbRegime = m_pOptimize->getNbRegime();
                regAndVal  = readRegressedValues(m_nameCont, boost::lexical_cast<string>(m_iStep));
            }
            else
            {
                gridFollLoc = m_pGridCurrent ;
                nbRegime = m_pOptimize->getNbDetRegime();
                regAndVal  = readRegressedValues(m_nameDetCont, boost::lexical_cast<string>(iPeriod));
            }
            pair< vector<int>, vector< array< double, 2>  > > simToProcAndRegion =  splitParticleOnProcessor(p_statevector, gridFollLoc);
            // number of particle for the current processor
            size_t nbSimCurProc = simToProcAndRegion.first.size();
            // nows store stocks
            ArrayXd stockPerSim(m_pGridFollowing->getDimension()*nbSimCurProc);
            // nows store regimes
            ArrayXi regimePerSim(nbSimCurProc);
            // store value functions
            ArrayXXd valueFunctionPerSim(m_pOptimize->getSimuFuncSize(), nbSimCurProc);
            // extend continuation values
            SubMeshIntCoord retGrid = calculateSubMeshExtended(gridFollLoc, simToProcAndRegion.second)  ;
            shared_ptr<FullGrid> gridExtended = gridFollLoc->getSubGrid(retGrid);
            vector< GridAndRegressedValue >  continuationExtended;
            // read regressed values on archive
            continuationExtended.reserve(nbRegime);
            for (int iReg = 0; iReg < nbRegime; ++iReg)
            {
                ArrayXXd valuesExtended = m_parall->reconstructAll<double>(regAndVal.second[iReg], retGrid);
                continuationExtended.push_back(GridAndRegressedValue(gridExtended, regAndVal.first));
                // affect
                continuationExtended[iReg].setRegressedValues(valuesExtended);
            }
            // spread calculations on processors
            for (size_t is = 0; is <  nbSimCurProc; ++is)
            {
                int simuNumber = simToProcAndRegion.first[is];;
                m_pOptimize->stepSimulate(gridFollLoc, continuationExtended,  p_statevector[simuNumber], p_phiInOut[iPeriod].col(simuNumber));
                // store for broadcast
                stockPerSim.segment(is * gridFollLoc->getDimension(), gridFollLoc->getDimension()) = p_statevector[simuNumber].getPtStock();
                regimePerSim(is) = p_statevector[simuNumber].getRegime();
                if (valueFunctionPerSim.size() > 0)
                    valueFunctionPerSim.col(is) = p_phiInOut[iPeriod].col(simuNumber);
            }
            // broadcast
            vector<double> stockAllSim;
            boost::mpi::all_gatherv<double>(m_world, stockPerSim.data(), stockPerSim.size(), stockAllSim);
            vector<int> regimeAllSim;
            boost::mpi::all_gatherv<int>(m_world, regimePerSim.data(), regimePerSim.size(), regimeAllSim);
            vector<double> valueFunctionAllSim;
            boost::mpi::all_gatherv<double>(m_world, valueFunctionPerSim.data(), valueFunctionPerSim.size(), valueFunctionAllSim);
            vector<int> simAllProc;
            boost::mpi::all_gatherv<int>(m_world, simToProcAndRegion.first.data(), nbSimCurProc, simAllProc);
            // update results
            int iis = 0;
            for (size_t is = 0; is < simAllProc.size(); ++is)
            {
                for (int iid = 0; iid < m_pOptimize->getSimuFuncSize(); ++iid)
                    p_phiInOut[iPeriod](iid, simAllProc[is]) = valueFunctionAllSim[iis++];
                Map<const ArrayXd >  ptStock(&stockAllSim[is * m_pGridFollowing->getDimension()], m_pGridFollowing->getDimension());
                p_statevector[simAllProc[is]].setPtStock(ptStock);
                p_statevector[simAllProc[is]].setRegime(regimeAllSim[is]);
            }
            //prepare next period
            if (iPeriod < nbPeriodsOfCurrentStep - 1)
            {
                p_phiInOut[iPeriod + 1] = p_phiInOut[iPeriod];
            }
        }
    }
}
#endif
