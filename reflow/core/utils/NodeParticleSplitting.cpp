#include<iostream>
#include "reflow/core/utils/NodeParticleSplitting.h"

using namespace std;
using namespace Eigen;

namespace reflow
{

NodeParticleSplitting::NodeParticleSplitting(const unique_ptr< ArrayXXd > &p_particles, const ArrayXi &p_nbMeshPerDim): m_root(new Node(p_particles->cols() - 1)), m_particles(p_particles), m_nbMeshPerDim(p_nbMeshPerDim)
{
    ArrayXi ipartToSplit(p_particles->rows());
    for (int is = 0; is < p_particles->rows(); ++is)
        ipartToSplit(is) = is;
    // sort particles in first dimension
    m_root->partitionDomain(* m_particles, m_nbMeshPerDim(p_particles->cols() - 1), ipartToSplit);
    // create son
    m_root->createSon();
    for (int j = 0; j <  m_root->getSonNumber(); ++j)
        BuildTree(m_root->getSon(j), m_root->getParticleSon(j));

}

void NodeParticleSplitting::BuildTree(const std::unique_ptr< Node> &p_node,  const Eigen::ArrayXi &p_iPart)
{
    p_node->partitionDomain(* m_particles, m_nbMeshPerDim(p_node-> getIDim()), p_iPart);
    p_node->createSon();
    for (int j = 0; j <  p_node->getSonNumber(); ++j)
        BuildTree(p_node->getSon(j), p_node->getParticleSon(j));

}

void NodeParticleSplitting::simToCellRecursive(unique_ptr<Node> &p_node,
        int &p_ipCell,
        int p_ipDim,
        int p_iProdMesh,
        int p_iDecMesh,
        ArrayXi &p_nCell,
        Array<  array<double, 2 >, Dynamic, Dynamic >   &p_meshCoord)
{
    const std::vector< Eigen::ArrayXi  >   &partNumberBySon = p_node->getPartNumberBySon();
    if (p_node->isItLeaf())
    {
        for (size_t  i = 0;  i <  static_cast<size_t>(partNumberBySon.size()); ++i)
        {
            for (size_t  j = 0 ; j <  static_cast<size_t>(partNumberBySon[i].size())	; ++j)
                p_nCell(partNumberBySon[i](j)) = p_ipCell;
            p_meshCoord(0, p_ipCell)[0] =  p_node->getCoordVertex()(i);
            if (i > 0)
                p_meshCoord(0, p_ipCell - 1)[1] =  p_node->getCoordVertex()(i);
            p_ipCell += 1 ;
        }
        // last
        p_meshCoord(0, p_ipCell - 1)[1] = p_node->getCoordVertex()(p_node->getCoordVertex().size() - 1);
    }
    else
    {
        int iDecMeshLoc = p_iDecMesh;
        p_iProdMesh /=  m_nbMeshPerDim(p_ipDim);
        for (size_t i = 0;  i <  static_cast<size_t>(partNumberBySon.size()); ++i)
        {
            for (size_t iloc = 0 ; iloc < static_cast<size_t>(p_iProdMesh); ++iloc)
                p_meshCoord(p_ipDim, iDecMeshLoc + iloc)[0] =  p_node->getCoordVertex()(i);
            if (i > 0)
            {
                for (int iloc = 0 ; iloc < 	p_iProdMesh; ++iloc)
                    p_meshCoord(p_ipDim, iDecMeshLoc + iloc - p_iProdMesh)[1] =  p_node->getCoordVertex()(i);
            }
            iDecMeshLoc += 	p_iProdMesh ;
        }
        for (int iloc = 0 ; iloc < 	p_iProdMesh; ++iloc)
            p_meshCoord(p_ipDim, iDecMeshLoc + iloc - p_iProdMesh)[1] =  p_node->getCoordVertex()(p_node->getCoordVertex().size() - 1);
        for (int i = 0; i < p_node->getSonNumber(); ++i)
        {
            simToCellRecursive(p_node->getSon(i), p_ipCell, p_ipDim - 1, p_iProdMesh, p_iDecMesh, p_nCell, p_meshCoord);
            p_iDecMesh += p_iProdMesh;
        }
    }
}

void NodeParticleSplitting::simToCell(ArrayXi &p_nCell,
                                      Array<  array<double, 2 >, Dynamic, Dynamic >   &p_meshCoord)
{
    int ipCell = 0 ;
    int ipDim = m_particles->cols() - 1; // start working in last dimension
    int iProdMesh = m_nbMeshPerDim.prod();
    int iDecMesh = 0;
    // recursive call
    simToCellRecursive(m_root, ipCell, ipDim, iProdMesh, iDecMesh, p_nCell, p_meshCoord);

}
}
