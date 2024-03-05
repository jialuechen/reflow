
#include <iostream>
#include <map>
#include <Eigen/Dense>
#include "libflow/core/utils/Node.h"

using namespace std;
using namespace Eigen;

namespace libflow
{

/// \brief to order particle
/// \param p_pt1 first point for comparison
/// \param p_pt2 second point for comparison
bool OrderParticle(const std::pair<double, int> &p_pt1, const std::pair<double, int> &p_pt2)
{
    return (p_pt1.first < p_pt2.first) ;
}



void Node::partitionDomain(const ArrayXXd &p_globalSetOfParticles, const int &p_nbPartition,
                           const ArrayXi &p_partToSplit)
{
    int inpart = p_partToSplit.size();
    // number of particles per mesh
    int iRest = inpart % p_nbPartition;
    ArrayXi nbPartPerMesh = ArrayXi::Constant(p_nbPartition, inpart / p_nbPartition);
    nbPartPerMesh.head(iRest) += 1;
    m_partNumberBySon.resize(p_nbPartition);
    for (int id = 0; id < p_nbPartition; ++id)
    {
        m_partNumberBySon[id].resize(nbPartPerMesh(id));
    }
    vector<std::pair< double, int > > part(inpart);
    for (int i = 0 ; i < inpart ; ++i)
    {
        part[i] = std::make_pair(p_globalSetOfParticles(p_partToSplit(i), m_iDim), p_partToSplit(i));
    }

    vector< std::pair< double, int > >::iterator startD = part.begin();
    vector< std::pair< double, int > >::iterator endD = part.end();
    nth_element(startD, startD, endD, OrderParticle);
    m_coordVertex.resize(p_nbPartition + 1);
    int ipos = 0 ;
    m_coordVertex(0) = part[0].first;
    nth_element(startD + 1, startD + nbPartPerMesh(0) - 1, endD, OrderParticle);
    m_coordVertex(1) =  part[nbPartPerMesh(0) - 1].first;
    for (int i = 1  ; i < p_nbPartition ; ++i)
    {
        if (nbPartPerMesh(i) > 0)
        {
            ipos += nbPartPerMesh(i - 1);
            nth_element(startD + ipos, startD + ipos + nbPartPerMesh(i) - 1, endD, OrderParticle);
            m_coordVertex(i + 1) = part[ipos + nbPartPerMesh(i) - 1].first;
        }
    }
    int iposLoc = 0;
    for (int id = 0; id < p_nbPartition; ++id)
    {
        for (int i = 0; i < nbPartPerMesh(id); ++i)
            m_partNumberBySon[id](i) = part[iposLoc++].second ;
    }
}
}
