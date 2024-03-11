
#ifndef NODEPARTICLESPLITTING_H
#define NODEPARTICLESPLITTING_H
#include <memory>
#include <array>
#include <Eigen/Dense>
#include "libflow/core/utils/Node.h"


namespace libflow
{

/// \class NodeParticleSplitting NodeParticleSplitting.h
/// Permits to split a Cartesian mesh  with the same number of particles inside each mesh. The new mesh is not conform
class NodeParticleSplitting
{
private :

    std::unique_ptr<Node> m_root; ///< root node for tree
    const std::unique_ptr< Eigen::ArrayXXd > &m_particles ;  ///< particles to spread towards meshes
    Eigen::ArrayXi  m_nbMeshPerDim ; ///< number of meshes per dimension

    /// \brief function use recursively to build the tree
    /// \param p_node  current node
    /// \param p_iPart  list of the particles number to tree
    void BuildTree(const std::unique_ptr< Node >   &p_node,  const Eigen::ArrayXi &p_iPart);

    /// \brief  to each point, give its mesh number after sorting (recursive version)
    //// \param p_node    Current node
    ///  \param  p_ipCell  Current cell number
    ///  \param p_ipDim    Current dimension
    /// \param  p_iProdMesh Utilitarian to increment mesh number
    /// \param p_iDecMesh Offset for mesh
    /// \param p_nCell   Array simulation mesh number
    /// \param m_meshCoord  for each mesh give it min and max coordinates
    void simToCellRecursive(std::unique_ptr<Node> &p_node,
                            int &p_ipCell, int p_ipDim,
                            int p_iProdMesh,
                            int p_iDecMesh,
                            Eigen::ArrayXi &p_nCell,
                            Eigen::Array<  std::array<double, 2 >, Eigen::Dynamic, Eigen::Dynamic >   &m_meshCoord);
public :

    /// \brief Tree creation
    /// \param p_particles    N dimensional particles to treat
    /// \param p_nbMeshPerDim  number of mesh per direction
    NodeParticleSplitting(const std::unique_ptr<Eigen::ArrayXXd > &p_particles, const Eigen::ArrayXi &p_nbMeshPerDim);


    /// \brief to each point, give its mesh number after sorting
    /// \param p_nCell   Array simulation mesh number
    /// \param p_meshCoord  for each mesh give it min and max coordinates
    void simToCell(Eigen::ArrayXi &p_nCell,
                   Eigen::Array<  std::array<double, 2 >, Eigen::Dynamic, Eigen::Dynamic >   &p_meshCoord);

};
}
#endif /* NODEPARTICLESPLITTING_H */
