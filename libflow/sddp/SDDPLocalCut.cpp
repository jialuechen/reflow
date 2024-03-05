
#ifdef USE_MPI
#include <boost/mpi.hpp>
#include <boost/version.hpp>
#endif
#include <vector>
#include <tuple>
#include <boost/serialization/vector.hpp>
#include "boost/lexical_cast.hpp"
#include "geners/BinaryFileArchive.hh"
#include "geners/Record.hh"
#include "geners/Reference.hh"
#include "libflow/sddp/SDDPLocalCut.h"
#include "libflow/sddp/SDDPACutGeners.h"

using namespace Eigen ;
using namespace std;
using namespace gs;

namespace libflow
{

SDDPLocalCut::SDDPLocalCut() {}

SDDPLocalCut::SDDPLocalCut(const int &p_date, const int &p_sample, shared_ptr< LocalRegression >  p_regressor): m_date(p_date), m_regressor(p_regressor),
    m_cuts(p_regressor->getNbMeshTotal()), m_sample(p_sample) {}

SDDPLocalCut::SDDPLocalCut(const int &p_date, shared_ptr< LocalRegression >  p_regressor): m_date(p_date), m_regressor(p_regressor),
    m_cuts(p_regressor->getNbMeshTotal()), m_sample(1) {}


vector< tuple< shared_ptr<ArrayXd>, int, int >  > SDDPLocalCut::createVectorStatesParticle(const SDDPVisitedStates &p_states) const
{
    vector< tuple< shared_ptr<ArrayXd>, int, int >  >  vectorOfLp;
    if (m_regressor->getNbSimul() == 0)
    {
        // just copy all stocks
        vectorOfLp.reserve(p_states.getStateSize());
        for (int istate  = 0; istate < p_states.getStateSize(); ++istate)
            vectorOfLp.push_back(make_tuple(p_states.getAState(istate), 0, 0));
    }
    else
    {
        // a cut is only affected to some particles
        int nbLP = 0;
        for (int istate  = 0; istate < p_states.getStateSize(); ++istate)
            nbLP += m_regressor->getSimulBelongingToCell()[p_states.getMeshAssociatedToState(istate)]->size();
        vectorOfLp.reserve(nbLP);
        for (int istate  = 0; istate < p_states.getStateSize(); ++istate)
        {
            int imesh = p_states.getMeshAssociatedToState(istate);
            for (size_t ipart  = 0; ipart < m_regressor->getSimulBelongingToCell()[imesh]->size() ; ++ipart)
                vectorOfLp.push_back(make_tuple(p_states.getAState(istate), (*m_regressor->getSimulBelongingToCell()[imesh])[ipart], imesh));
        }
    }
    return vectorOfLp ;
}

#ifdef USE_MPI
void SDDPLocalCut::createAndStoreCuts(const ArrayXXd &p_cutPerSim, const SDDPVisitedStates &p_states, const vector< tuple< shared_ptr<ArrayXd>, int, int >  > &p_vectorOfLp,
                                      const shared_ptr<BinaryFileArchive> &p_ar, const boost::mpi::communicator &p_world)
#else
void SDDPLocalCut::createAndStoreCuts(const ArrayXXd &p_cutPerSim, const SDDPVisitedStates &p_states, const vector< tuple< shared_ptr<ArrayXd>, int, int >  > &,
                                      const shared_ptr<BinaryFileArchive> &p_ar)
#endif
{
#ifdef USE_MPI
    // Tab to store to send to processor
    vector< array<int, 2> > tabSendToProcessor;
    // Tab to store what is to receive from other processor
    vector< array<int, 2> > tabRecFromProcessor;

    // nsample per sample per mesh
    int nbState = p_states.getStateSize();
    vector<int> nbPartPerState(nbState);
    for (int ist = 0 ; ist < nbState; ++ist)
        nbPartPerState[ist] =  m_regressor-> getSimulBelongingToCell()[ p_states.getMeshAssociatedToState(ist)]->size() * m_sample;
    // first and last stock by current  processor
    array<int, 2>   stockByProc;
    routingSchedule(p_vectorOfLp.size()*m_sample, nbPartPerState, tabSendToProcessor, tabRecFromProcessor, stockByProc, p_world);
    int iFirstState = stockByProc[0];
    int iLastState = stockByProc[1] ;
    ArrayXXd cutPerSimProc(p_cutPerSim.rows(), p_vectorOfLp.size()*m_sample);
    mpiExec(p_cutPerSim, tabSendToProcessor, tabRecFromProcessor, cutPerSimProc, p_world);
#else
    int iFirstState = 0;
    int iLastState =   p_states.getStateSize() ;
    Map< const ArrayXXd > cutPerSimProc(p_cutPerSim.data(), p_cutPerSim.rows(), p_cutPerSim.cols());
#endif

    vector< shared_ptr<SDDPACut> > localCut;
    localCut.reserve(iLastState - iFirstState);
    vector<int> meshCut;
    meshCut.reserve(iLastState - iFirstState);
    int iposCut = 0;
    for (int isto = iFirstState; isto < iLastState; ++isto)
    {
        int imesh = p_states.getMeshAssociatedToState(isto); // mesh used for conditional expectation
        ArrayXXd cutExpectancy = ArrayXXd::Zero(p_cutPerSim.rows(), m_regressor->getSimulBelongingToCell()[imesh]->size());
        for (size_t is = 0; is < m_regressor->getSimulBelongingToCell()[imesh]->size(); ++is)
        {
            // conditional expectation with respect to state
            for (int isample = 0; isample < m_sample; ++isample)
            {
                cutExpectancy.col(is) += cutPerSimProc.col(iposCut++);

            }
            cutExpectancy.col(is) /= m_sample;
        }
        // now conditional expectation with respect to external  : create the cut
        shared_ptr<ArrayXXd> cutArray = make_shared<ArrayXXd>(m_regressor->getCoordBasisFunctionMultipleOneCell(imesh, cutExpectancy));
        shared_ptr<SDDPACut> cutToAdd = make_shared<SDDPACut>(cutArray);
        localCut.push_back(cutToAdd);
        meshCut.push_back(imesh);
    }
#ifdef USE_MPI
    gatherAndStoreCuts(localCut,  meshCut,  "CutMesh", m_cuts,  p_ar, m_regressor->getNbMeshTotal(), m_date, p_world);
#else
    gatherAndStoreCuts(localCut,  meshCut,  "CutMesh", m_cuts,  p_ar, m_regressor->getNbMeshTotal(), m_date);
#endif
}


ArrayXXd    SDDPLocalCut::getCutsAssociatedToTheParticle(int p_isim) const
{
    // cell associated
    int ncell = m_regressor->getCellAssociatedToSim(p_isim);
    if (m_cuts[ncell].size() == 0)
        return ArrayXXd();
    ArrayXd aParticule = ((m_regressor->getNbSimul() > 0) ? m_regressor->getParticle(p_isim) : ArrayXd());
    int iStateSize = m_cuts[ncell][0]->getStateSize();
    ArrayXXd retCut(iStateSize + 1, m_cuts[ncell].size());
    for (size_t icut = 0; icut < m_cuts[ncell].size(); ++icut)
    {
        retCut.col(icut) = m_regressor->getValuesOneCell(aParticule, ncell, *m_cuts[ncell][icut]->getCut());
    }
    return retCut;
}

ArrayXXd    SDDPLocalCut::getCutsAssociatedToAParticle(const ArrayXd &p_aParticle) const
{
    // cell associated
    int ncell = m_regressor->getMeshNumberAssociatedTo(p_aParticle);
    if (m_cuts[ncell].size() == 0)
        return ArrayXXd();
    int iStateSize = m_cuts[ncell][0]->getStateSize();
    ArrayXXd retCut(iStateSize + 1, m_cuts[ncell].size());
    for (size_t icut = 0; icut < m_cuts[ncell].size(); ++icut)
    {
        retCut.col(icut) = m_regressor->getValuesOneCell(p_aParticle, ncell, *m_cuts[ncell][icut]->getCut());
    }
    return retCut;
}


}
