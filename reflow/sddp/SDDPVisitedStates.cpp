
#ifdef USE_MPI
#include <boost/version.hpp>
#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include "reflow/core/parallelism/gathervExtension.hpp"
#include <boost/serialization/vector.hpp>
#include "reflow/core/utils/eigenSerialization.h"
#include "reflow/core/utils/stdSharedPtrSerialization.h"
#endif
#include <memory>
#include <iostream>
#include "reflow/sddp/SDDPVisitedStates.h"

using namespace Eigen;
using namespace std;

namespace reflow
{
SDDPVisitedStates:: SDDPVisitedStates() : SDDPVisitedStatesBase()  {}

SDDPVisitedStates:: SDDPVisitedStates(const int &p_nbNode) : SDDPVisitedStatesBase(p_nbNode) {}

SDDPVisitedStates:: SDDPVisitedStates(const vector< vector< int> >   &p_meshToState, const vector< shared_ptr< Eigen::ArrayXd >  > &p_stateVisited, const vector< int > &p_associatedMesh)   : SDDPVisitedStatesBase(p_meshToState, p_stateVisited, p_associatedMesh) {}

void SDDPVisitedStates::addVisitedState(const shared_ptr< ArrayXd > &p_state, const ArrayXd &p_particle, const LocalRegression &p_regressor)
{
    int ncell = p_regressor.getMeshNumberAssociatedTo(p_particle);
#ifdef _OPENMP
    #pragma omp critical (visited)
#endif
    {
        if (isStateNotAlreadyVisited(p_state, ncell))
        {
            m_meshToState[ncell].push_back(m_stateVisited.size());
            m_stateVisited.push_back(p_state);
            m_associatedMesh.push_back(ncell);
        }
    }
}

void SDDPVisitedStates::addVisitedStateForAll(const shared_ptr< ArrayXd > &p_state, const LocalRegression &p_regressor)
{
    int nbCell = p_regressor.getNbMeshTotal();
    for (int icell = 0; icell < nbCell; ++icell)
    {
        m_meshToState[icell].push_back(m_stateVisited.size());
        m_stateVisited.push_back(p_state);
        m_associatedMesh.push_back(icell);
    }
}
}








