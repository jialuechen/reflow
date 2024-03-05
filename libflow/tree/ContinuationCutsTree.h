
#ifndef CONTINUATIONCUTSTREE_H
#define CONTINUATIONCUTSTREE_H
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "libflow/tree/Tree.h"
#include "libflow/core/grids/SpaceGrid.h"

/** \file ContinuationCutsTree.h
 *  \brief In DP, when transition problem are solved using LP some final cuts are given to the LP solver as ending conditions
 *        This permits to store cuts   coefficients with some tree.
 *        For each point of the grid, the cuts are stored.
 * \author  Xavier Warin
 */

namespace libflow
{

/// \class ContinuationCutsTree ContinuationCutsTree.h
/// Permits to store  some cuts at each point of a space grid
class ContinuationCutsTree
{

private :
    std::shared_ptr< SpaceGrid >    m_grid ; ///< grid used to define stock points
    std::vector< Eigen::ArrayXXd > m_cutCoeff ; ///< for each  cut  \f$ \bar a_0 + \sum_i^d a_i x_i  \f$ store the  coefficients coefficient. Size of m_cutCoeff : (d+1). For each component of the cut, store its value for a given node and a given stock point  (size (nb node , nb stock points)). Notice that  \f$ \bar a_0 =  a_0 - \sum_i^d a_i \bar x_i \f$

public :
    /// \brief Default constructor
    ContinuationCutsTree() {}

    /// \brief Constructor
    /// \param p_grid   grid for stocks
    /// \param p_condExp tree for conditional expectation
    /// \param p_values   functions to store  ((number of nodes at next date  by number of cuts) by (nb of stocks)). The pValues are given as
    ///                   \f$ a_0 + \sum_i^d a_i (x_i -\bar x_i) \f$ at a point \f$ \bar x \f$.
    ContinuationCutsTree(const  std::shared_ptr< SpaceGrid >   &p_grid,
                         const  std::shared_ptr< Tree >   &p_condExp,
                         const  Eigen::ArrayXXd &p_values);



    /// \brief Load another Continuation value object
    ///  Only a partial load of the objects is achieved
    /// \param p_grid   Grid to load
    /// \param p_values coefficient polynomials for regression
    virtual void loadForSimulation(const  std::shared_ptr< SpaceGrid > &p_grid,
                                   const std::vector< Eigen::ArrayXXd  >  &p_values)
    {
        m_grid = p_grid;
        m_cutCoeff = p_values ;
    }

    /// \brief Get a list of all cuts for all nodes \f$ (\bar a_0, a_1, ...a_d) \f$
    ///        for  stock points in a set
    /// \param  p_hypStock list of points  defining an hypercube :
    ///          - (i,0)  coordinate corresponds to min value in dimension i
    ///          - (i,1)  coordinate corresponds to max value in dimension i
    ///          .
    /// Return an array of cuts for all points in the hypercube  and nodes used
    /// shape  :  first dimension :   (nb nodes by number of cuts)
    ///           second dimension : nb of points  in the  hypercube
    Eigen::ArrayXXd  getCutsAllNodes(const Eigen::ArrayXXd &p_hypStock) const;


    /// \brief get list of cuts associated to an hypercube
    /// \param  p_hypStock      list of points  defining an hypercube
    /// \param  p_node           node number
    /// \return  list of cuts (nb cut coeff, nb stock points)
    Eigen::ArrayXXd getCutsANode(const Eigen::ArrayXXd &p_hypStock, const int &p_node) const;


    /// \brief get Regressed values stored
    const std::vector< Eigen::ArrayXXd>   getValues() const
    {
        return m_cutCoeff;
    }

    //// \brief Get back
    ///@{
    std::shared_ptr< SpaceGrid > getGrid() const
    {
        return m_grid;
    }
    inline int getNbNodes() const
    {
        return  m_cutCoeff[0].rows();
    }
    ///@}

};
}
#endif
