
#ifdef USE_MPI
#include <boost/mpi.hpp>
#include <boost/version.hpp>
#include "reflow/core/parallelism/all_gatherv.hpp"
#endif
#include <vector>
#include <tuple>
#include <boost/serialization/vector.hpp>
#include "boost/lexical_cast.hpp"
#include "geners/BinaryFileArchive.hh"
#include "geners/Record.hh"
#include "geners/Reference.hh"
#include "reflow/sddp/SDDPCutTree.h"
#include "reflow/core/utils/constant.h"

using namespace Eigen ;
using namespace std;
using namespace gs;

namespace reflow
{

SDDPCutTree::SDDPCutTree() {}

SDDPCutTree::SDDPCutTree(const int &p_date, const int &p_sample,  const std::vector<double>  &p_proba, const std::vector< std::vector< std::array<int, 2> > > &p_connected, const ArrayXXd &p_nodes): m_date(p_date), m_cuts(p_nodes.cols()), m_tree(p_proba, p_connected), m_nodes(p_nodes), m_sample(p_sample) {}

SDDPCutTree::SDDPCutTree(const int &p_date, const Eigen::ArrayXXd &p_nodes): m_date(p_date), m_nodes(p_nodes), m_sample(1) {}

vector< tuple< shared_ptr<ArrayXd>, int, int >  > SDDPCutTree::createVectorStatesParticle(const SDDPVisitedStatesTree &p_states) const
{
    vector< tuple< shared_ptr<ArrayXd>, int, int >  >  vectorOfLp;
    // a cut is only affected to some particles
    int nbLP = 0;
    for (int istate  = 0; istate < p_states.getStateSize(); ++istate)
        nbLP += m_tree.getNbConnected(p_states.getMeshAssociatedToState(istate));
    vectorOfLp.reserve(nbLP);
    for (int istate  = 0; istate < p_states.getStateSize(); ++istate)
    {
        int imesh = p_states.getMeshAssociatedToState(istate);
        for (int ipart  = 0; ipart < m_tree.getNbConnected(imesh) ; ++ipart)
            vectorOfLp.push_back(make_tuple(p_states.getAState(istate), m_tree.getArrivalNode(imesh, ipart), imesh));
    }
    return vectorOfLp ;
}


#ifdef USE_MPI
void SDDPCutTree::createAndStoreCuts(const ArrayXXd &p_cutPerSim, const SDDPVisitedStatesTree &p_states, const vector< tuple< shared_ptr<ArrayXd>, int, int >  > &p_vectorOfLp,
                                     const shared_ptr<BinaryFileArchive> &p_ar, const boost::mpi::communicator &p_world)
#else
void SDDPCutTree::createAndStoreCuts(const ArrayXXd &p_cutPerSim, const SDDPVisitedStatesTree &p_states, const vector< tuple< shared_ptr<ArrayXd>, int, int >  > &,
                                     const shared_ptr<BinaryFileArchive> &p_ar)
#endif
{
#ifdef USE_MPI
    // Tab to store to send to processor
    vector< array<int, 2> > tabSendToProcessor;
    // Tab to store what is to receive from other processor
    vector< array<int, 2> > tabRecFromProcessor;
    int nbState = p_states.getStateSize();
    vector<int> nbPartPerState(nbState);
    for (int ist = 0 ; ist < nbState; ++ist)
        nbPartPerState[ist] =  m_tree.getNbConnected(p_states.getMeshAssociatedToState(ist)) * m_sample;

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
    vector<int> nodeCut;
    nodeCut.reserve(iLastState - iFirstState);
    int iposCut = 0;
    for (int isto = iFirstState; isto < iLastState; ++isto)
    {
        int inode = p_states.getMeshAssociatedToState(isto); // node use for expectation
        shared_ptr<ArrayXXd> cutArray = make_shared<ArrayXXd>(ArrayXXd::Zero(p_cutPerSim.rows(), 1));
        ArrayXd cutExpectancy(p_cutPerSim.rows());
        for (int is = 0; is < m_tree.getNbConnected(inode); ++is)
        {
            cutExpectancy.setZero();
            // conditional expectation with respect to state
            for (int isample = 0; isample < m_sample; ++isample)
            {
                cutExpectancy += cutPerSimProc.col(iposCut++);
            }
            cutExpectancy /= m_sample;
            cutArray->col(0)  += m_tree.getProba(inode, is) * cutExpectancy;
        }
        shared_ptr<SDDPACut> cutToAdd = make_shared<SDDPACut>(cutArray);
        localCut.push_back(cutToAdd);
        nodeCut.push_back(inode);
    }
#ifdef USE_MPI
    gatherAndStoreCuts(localCut,  nodeCut,  "CutNode", m_cuts,  p_ar, m_nodes.cols(), m_date, p_world);
#else
    gatherAndStoreCuts(localCut,  nodeCut,  "CutNode", m_cuts,  p_ar, m_nodes.cols(), m_date);
#endif
}


ArrayXXd    SDDPCutTree::getCutsAssociatedToTheParticle(int p_node) const
{
    if (m_cuts[p_node].size() == 0)
        return ArrayXXd();
    int iStateSize = m_cuts[p_node][0]->getStateSize();
    ArrayXXd retCut(iStateSize + 1, m_cuts[p_node].size());
    for (size_t icut = 0; icut < m_cuts[p_node].size(); ++icut)
    {
        retCut.col(icut) =  m_cuts[p_node][icut]->getCut()->col(0);
    }
    return retCut;
}


ArrayXXd  SDDPCutTree::getCutsAssociatedToAParticle(const ArrayXd &p_aParticle) const
{
    int inode = 0;
    if (m_date > 0)
    {
        for (int i = 0; i < m_nodes.cols(); ++i)
        {
            bool bKeep = true;
            for (int id = 0 ; id < p_aParticle.size(); ++id)
            {
                if (std::fabs(p_aParticle(id) - m_nodes(id, i)) > tiny)
                {
                    bKeep = false;
                    break;
                }
            }
            if (bKeep)
            {
                inode = i;
                break;
            }
        }
    }
    int iStateSize = m_cuts[inode][0]->getStateSize();
    ArrayXXd retCut(iStateSize + 1, m_cuts[inode].size());
    for (size_t icut = 0; icut < m_cuts[inode].size(); ++icut)
    {
        retCut.col(icut) = m_cuts[inode][icut]->getCut()->col(0);
    }
    return retCut;
}
}
