
#ifndef NODE_H
#define NODE_H
#include <Eigen/Dense>
#include <vector>
#include <memory>


/** \file Node.h
 * \brief Permit to store  some son nodes, each son node contraining the some number of particules
 *     \author Xavier Warin
*/
namespace libflow
{

/// \class Node Node.h
/// Utilitary to split meshes
class Node
{
private :

    std::vector< Eigen::ArrayXi  > m_partNumberBySon ; ///< for each son , defines the particles number belonging to this node
    std::vector<  std::unique_ptr< Node> > m_sonNodes ; ///< nodes son of current one
    Eigen::ArrayXd m_coordVertex ; ///< coordinate of the vertex of the meshes
    int m_iDim ; ///< current dimension treated

public :

    /// \brief Default node constructor
    Node(int p_iDim): m_iDim(p_iDim) {}

    /// \brief get son
    /// \param p_iSon  son number
    std::unique_ptr< Node> &getSon(int p_iSon)
    {
        assert(p_iSon <  m_sonNodes.size());
        return m_sonNodes[p_iSon];
    }

    /// \brief accessor
    ///@{
    inline int getSonNumber() const
    {
        return  m_sonNodes.size() ;
    }
    inline  const std::vector< Eigen::ArrayXi  >   &getPartNumberBySon() const
    {
        return  m_partNumberBySon;
    }
    inline  int getIDim() const
    {
        return m_iDim ;
    }
    inline const Eigen::ArrayXd &getCoordVertex() const
    {
        return m_coordVertex ;
    }
    ///@}

    /// \brief  sons creation
    void createSon()
    {
        if (m_iDim > 0)
        {
            int m_sonNumber = m_partNumberBySon.size();
            /* m_sonNodes.reserve(m_sonNumber); */
            m_sonNodes.resize(m_sonNumber);
            for (int i = 0; i < m_sonNumber; ++i)
                /* not implemented yet in compilers */
                /* m_sonNodes.push_back(std::make_unique<Node>(m_iDim-1); */
            {
                std::unique_ptr<Node> ptr(new Node(m_iDim - 1));
                m_sonNodes[i] = std::unique_ptr<Node>(std::move(ptr));
            }
        }
    }

    ///\brief Get particules number associated to one son
    /// \param p_iSon  son number
    inline const  Eigen::ArrayXi   &getParticleSon(const int &p_iSon) const
    {
        return  m_partNumberBySon[p_iSon];
    }

    ///\brief Organize a partition of the particles in current direction
    /// \param p_globalSetOfParticles  global set of particles (nb particiles , dimension)
    /// \param p_nbPartition    number of partition to achive in this direction
    /// \param p_partToSplit    define the set of the numbers of the particles to split
    void partitionDomain(const Eigen::ArrayXXd &p_globalSetOfParticles, const int &p_nbPartition,
                         const Eigen::ArrayXi &p_partToSplit);

    /// \brief test if a node is a leaf
    bool isItLeaf() const
    {
        return (m_sonNodes.size() == 0);
    }

} ;
}
#endif /* NODE_H */
