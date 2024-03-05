
#ifdef USE_MPI
#include <boost/version.hpp>
#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include "libflow/core/parallelism/gathervExtension.hpp"
#include <boost/serialization/vector.hpp>
#endif
#include <memory>
#include <iostream>
#include "libflow/sddp/SDDPVisitedStatesTree.h"

using namespace Eigen;
using namespace std;

namespace libflow
{
SDDPVisitedStatesTree:: SDDPVisitedStatesTree()  : SDDPVisitedStatesBase() {}

SDDPVisitedStatesTree:: SDDPVisitedStatesTree(const int &p_nbNode)  : SDDPVisitedStatesBase(p_nbNode) {}

SDDPVisitedStatesTree:: SDDPVisitedStatesTree(const vector< vector< int> >   &p_meshToState, const vector< shared_ptr< Eigen::ArrayXd >  > &p_stateVisited, const vector< int > &p_associatedMesh)   : SDDPVisitedStatesBase(p_meshToState, p_stateVisited, p_associatedMesh) {}

void SDDPVisitedStatesTree::addVisitedState(const shared_ptr< ArrayXd > &p_state,  const int  &p_point)
{
#ifdef _OPENMP
    #pragma omp critical (visited)
#endif
    {
        if (isStateNotAlreadyVisited(p_state, p_point))
        {
            m_meshToState[p_point].push_back(m_stateVisited.size());
            m_stateVisited.push_back(p_state);
            m_associatedMesh.push_back(p_point);
        }
    }
}

void SDDPVisitedStatesTree::addVisitedStateForAll(const shared_ptr< ArrayXd > &p_state,  const int &p_nbNode)
{
    for (int icell = 0; icell < p_nbNode; ++icell)
    {
        m_meshToState[icell].push_back(m_stateVisited.size());
        m_stateVisited.push_back(p_state);
        m_associatedMesh.push_back(icell);
    }
}

}








