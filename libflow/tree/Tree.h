
#ifndef TREE_H
#define TREE_H
#include <memory>
#include <vector>
#include <array>
#include <Eigen/Dense>

/** \file Tree.h
 * \brief Class to store a scenario tree
 * \author Xavier warin
 */

namespace libflow
{
/// \class Tree Tree.h
/// Tree class to calculate conditional expectation
class Tree
{
private :
    std::vector<double> m_proba ;///< probality array
    std::vector< std::vector< std::array<int, 2> > > m_connected ; ///< connection matrix between points (nodes) on tree between 2 dates :  m_connected[i][j][0]  gives for node i at current date,  the number of the node at next date  and the index in the probability array is m_connected[i][j][1]. The number of connection of node i is m_connected[i].size()
    int m_nbNodeNextDate;


public :

    /// \brief Default constructor
    Tree();

    /// \brief Default destructor
    virtual ~Tree() {}

    /// \brief Constructor storing nodes values at curretndates and probability matrix
    /// \param p_proba     probability between nodes on two dates
    /// \param p_connected  connection between nodes
    Tree(const std::vector<double> &p_proba, const std::vector< std::vector< std::array<int, 2> >  > &p_connected);



    /// \brief update the data in existing Tree
    /// \param p_probab     probability between nodes on two dates
    /// \param p_connected  connection between nodes
    void update(const std::vector<double> &p_proba,
                const std::vector< std::vector< std::array<int, 2> > > &p_connected);

    /// \brief Get some local accessors
    ///@{
    inline std::vector<double> getProba() const
    {
        return m_proba;
    }

    inline std::vector< std::vector< std::array<int, 2> > >  getConnected() const
    {
        return  m_connected;
    }

    ///@}

    /// \brief Calculated conditional expectation
    /// \param p_values at the tree node at the next date
    Eigen::ArrayXd  expCond(const Eigen::ArrayXd &p_values) const ;

    /// \brief Calculated conditional expectation
    /// \param p_values at the tree node at the next date (size :  number of function to regress  \times number of nodes)
    Eigen::ArrayXXd  expCondMultiple(const Eigen::ArrayXXd &p_values) const ;

    /// \brief Number of nodes at current date
    inline int getNbNodes() const
    {
        return m_connected.size();
    }

    /// \brief Number of nodes at next date
    inline int getNbNodesNextDate() const
    {
        return m_nbNodeNextDate;
    }

    /// \brief connected node
    /// \param p_iStart starting node
    /// \param p_num    number of the branch
    /// \return arrival node
    inline int getArrivalNode(const int &p_iStart, const int &p_num)
    const
    {
        return m_connected[p_iStart][p_num][0];
    }

    /// \brief number of nodes connected to a node
    inline int getNbConnected(const int &p_node) const
    {
        return  m_connected[p_node].size();
    }

    /// \brief get probability
    /// \param p_iStart starting node
    /// \param p_num    number of the branch
    /// \return probability
    inline double getProba(const int &p_iStart, const int &p_num) const
    {
        return m_proba[m_connected[p_iStart][p_num][1]];
    }
};
}
#endif

