
#ifdef USE_MPI
#include <boost/version.hpp>
#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include "libflow/core/parallelism/gathervExtension.hpp"
#include <boost/serialization/vector.hpp>
#include "libflow/core/utils/eigenSerialization.h"
#include "libflow/core/utils/stdSharedPtrSerialization.h"
#endif
#include <memory>
#include <iostream>
#include <Eigen/Dense>
#include "libflow/core/utils/constant.h"
#include "libflow/sddp/SDDPVisitedStatesBase.h"

using namespace Eigen;
using namespace std;

namespace libflow
{
SDDPVisitedStatesBase:: SDDPVisitedStatesBase() {}


SDDPVisitedStatesBase:: SDDPVisitedStatesBase(const int &p_nbNode): m_meshToState(p_nbNode) {}

SDDPVisitedStatesBase:: SDDPVisitedStatesBase(const vector< vector< int> >   &p_meshToState,  const vector< shared_ptr< Eigen::ArrayXd >  > &p_stateVisited, const vector< int > &p_associatedMesh)   : m_stateVisited(p_stateVisited),
    m_associatedMesh(p_associatedMesh), m_meshToState(p_meshToState) {}

bool SDDPVisitedStatesBase:: isStateNotAlreadyVisited(const shared_ptr< ArrayXd > &p_state,  const int  &p_point) const
{
    for (size_t i = 0; i < m_meshToState[p_point].size(); ++i)
    {
        int iState = m_meshToState[p_point][i];
        for (int j = 0; j < m_stateVisited[iState]->size(); ++j)
        {
            if (std::fabs((*m_stateVisited[iState])(j) - (*p_state)(j)) > tiny)
            {
                break;
            }
            if (j ==   m_stateVisited[iState]->size() - 1)
            {
                return false;
            }
        }
    }
    return true;
}


void  SDDPVisitedStatesBase::recalculateVisitedState()
{
    std::vector< std::shared_ptr< Eigen::ArrayXd > > stateVisited  = m_stateVisited ;
    std::vector<int> associatedMesh = m_associatedMesh;
    m_stateVisited.clear();
    m_associatedMesh.clear();
    for (size_t i = 0; i < m_meshToState.size(); ++i)
    {
        m_meshToState[i].clear();
    }
    for (size_t i = 0; i < associatedMesh.size(); ++i)
        if (isStateNotAlreadyVisited(stateVisited[i], associatedMesh[i]))
        {
            m_meshToState[associatedMesh[i] ].push_back(m_stateVisited.size());
            m_stateVisited.push_back(stateVisited[i]);
            m_associatedMesh.push_back(associatedMesh[i]);
        }
}

void SDDPVisitedStatesBase::print() const
{
    cout << "States visited " << endl ;
    for (size_t i = 0; i < m_stateVisited.size() ; ++i)
        cout << "For  mesh " << m_associatedMesh[i] << " visited state " << *m_stateVisited[i] << endl ;
}

#ifdef USE_MPI
void SDDPVisitedStatesBase::sendToRoot(const boost::mpi::communicator &p_world)
{
    vector<int>  meshVisited(m_associatedMesh);
    gatherv(p_world, meshVisited.data(), meshVisited.size(), m_associatedMesh, 0);
    ArrayXd  state;
    int isizeState = ((m_stateVisited.size() > 0) ? m_stateVisited[0]->size() : 0);
    state.resize(m_stateVisited.size()*isizeState);
    for (size_t i = 0; i <  m_stateVisited.size(); ++i)
        state.segment(i * isizeState, isizeState) = *(m_stateVisited[i]);
    ArrayXd stateGather;
    gatherv(p_world, state.data(), state.size(), stateGather, 0);
    if (p_world.rank() == 0)
    {
        int isizeState = m_stateVisited[0]->size();
        m_stateVisited.resize(m_associatedMesh.size());
        for (size_t i = 0; i < m_associatedMesh.size(); ++i)
        {
            m_stateVisited[i] = make_shared<ArrayXd>(stateGather.segment(isizeState * i, isizeState));
        }
        // eliminate double
        recalculateVisitedState();
    }
}

void SDDPVisitedStatesBase::sendFromRoot(const boost::mpi::communicator &p_world)
{
    // Add this because Bug on mac (nullify existing  vectors)
    if (p_world.rank() != 0)
    {
        m_stateVisited.clear();
        m_associatedMesh.clear();
        for (size_t i = 0; i < m_meshToState.size(); ++i)
        {
            m_meshToState[i].clear();
        }
    }
    boost::mpi:: broadcast(p_world, m_associatedMesh, 0);
    // boost BUG
#if BOOST_VERSION <  105600
    boost::mpi:: broadcast(p_world, m_stateVisited, 0);
#else
    ArrayXd  state;
    if (p_world.rank() == 0)
    {
        int isizeState = ((m_stateVisited.size() > 0) ? m_stateVisited[0]->size() : 0);
        state.resize(m_stateVisited.size()*isizeState);
        for (size_t i = 0; i <  m_stateVisited.size(); ++i)
            state.segment(i * isizeState, isizeState) = *(m_stateVisited[i]);
    }
    boost::mpi:: broadcast(p_world, state, 0);
    if (p_world.rank() > 0)
    {
        if (state.size() > 0)
        {
            int isizeState = state.size() / m_associatedMesh.size();
            m_stateVisited.resize(m_associatedMesh.size());
            for (size_t i = 0; i < m_associatedMesh.size(); ++i)
            {
                m_stateVisited[i] = make_shared<ArrayXd>(state.segment(isizeState * i, isizeState));
            }
        }
    }
#endif
    for (size_t i = 0; i < m_meshToState.size(); ++i)
        boost::mpi:: broadcast(p_world,  m_meshToState[i], 0);
}
#endif
}








