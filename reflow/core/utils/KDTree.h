#ifndef KDTREE_H
#define KDTREE_H
#include <algorithm>
#include <functional>
#include <memory>
#include <Eigen/Dense>


namespace reflow
{

/// \brief A node
class KDNode
{
private:
    Eigen::ArrayXd  m_point ; ///< current point coodinate
    size_t  m_index ; ///< point index
    std::shared_ptr< KDNode > m_leftNode; /// < left node
    std::shared_ptr< KDNode > m_rightNode ; /// < right node

    /// \brief create node recursively
    std::shared_ptr< KDNode > createTree(const std::vector<std::pair< Eigen::ArrayXd, size_t>>::iterator &p_beg,
                                         const std::vector<std::pair< Eigen::ArrayXd, size_t>>::iterator &p_end,
                                         const size_t &p_nbPoints,
                                         const size_t &p_idim);

public :

    /// \brief Constructors
    KDNode() {};

    KDNode(const Eigen::ArrayXd &p_point, const size_t &p_index,
           const std::shared_ptr< KDNode > &p_leftNode,
           const std::shared_ptr< KDNode > &p_rightNode): m_point(p_point),
        m_index(p_index), m_leftNode(p_leftNode), m_rightNode(p_rightNode) {}

    KDNode(const std::pair< Eigen::ArrayXd, size_t >  &p_pointIndex,
           const std::shared_ptr< KDNode > &p_leftNode,
           const std::shared_ptr< KDNode > &p_rightNode): m_point(p_pointIndex.first),
        m_index(p_pointIndex.second), m_leftNode(p_leftNode), m_rightNode(p_rightNode) {}


    // get back left node
    inline std::shared_ptr< KDNode > getLeft() const
    {
        return m_leftNode;
    }

    // get back right node
    inline std::shared_ptr<KDNode > getRight() const
    {
        return m_rightNode;
    }

    /// \brief get coordinates
    inline  double getCoord(const int &idim) const
    {
        return m_point(idim);
    }

    inline Eigen::ArrayXd getPoint() const
    {
        return m_point ;
    }

    inline size_t getIndex() const
    {
        return m_index;
    }

    inline  std::pair< Eigen::ArrayXd, size_t> getPointIndex() const
    {
        return std::make_pair(m_point, m_index);
    }

    inline bool isEmpty() const
    {
        return (m_point.size() == 0);
    }

};

/// \brief a KDTree
class KDTree
{
private :

    std::shared_ptr< KDNode > m_root ; ///< Root tree
    std::shared_ptr< KDNode > m_leaf ; ///< Leaf tree


    /// \brief Create the tree
    std::shared_ptr<KDNode> createTree(const std::vector< std::pair< Eigen::ArrayXd, size_t>>::iterator &p_beg,
                                       const std::vector< std::pair< Eigen::ArrayXd, size_t>>::iterator   &p_end,
                                       const size_t &p_nbPoints,
                                       const size_t &p_level);

    /// \brief Recursive search of the nearest point
    /// \param p_branch    current  branch
    /// \param p_pt        point to evaluate
    /// \param p_level     level
    /// \param p_best      best node
    /// \param p_bestDist  smallest distance so far
    std::shared_ptr< KDNode > nearest(const std::shared_ptr< KDNode > &p_branch,
                                      const Eigen::ArrayXd &p_pt,
                                      const size_t &p_level,
                                      const std::shared_ptr< KDNode > &p_best,
                                      const double &p_bestDist) const ;
    ///  Nearest resolution
    ///  p_pt  the point which is tested
    std::shared_ptr< KDNode > nearestNode(const Eigen::ArrayXd   &p_pt) const ;

public:

    KDTree() {}

    /// \brief constructor
    /// \param    p_pt  Array of point (N,nb points)
    KDTree(const Eigen::ArrayXXd &p_pt);

    /// get back nearest point index
    /// p_pt   point where we are interested in
    size_t inline nearestIndex(const Eigen::ArrayXd   &p_pt) const
    {
        return nearestNode(p_pt)->getIndex();
    }
};
}
#endif
