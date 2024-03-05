
#ifndef CONTINUATIONVALUETREE_H
#define CONTINUATIONVALUETREE_H
#include <iostream>
#include <memory>
#include <Eigen/Dense>
#include "libflow/tree/Tree.h"
#include "libflow/core/grids/SpaceGrid.h"

/** \file ContinuatonValueTree.h
 *  \brief Calculate and store continuation value functon for tree method
 * \author Xavier warin
 */

namespace libflow
{

/// \class ContinuationValueTree ContinuationValueTree.h
/// Permits to store, store  continuation values on a grid for tree methods
class ContinuationValueTree
{

private :

    std::shared_ptr< SpaceGrid >    m_grid ; ///< grid used to define stock points
    Eigen::ArrayXXd m_values ; ///< store node values  for each stock  (nb node, nb stock points)  at current date

public :
    /// \brief Default constructor
    ContinuationValueTree() {}

    /// \brief Constructor
    /// \param p_grid            grid for stocks
    /// \param p_condExp         tree for conditional expectation
    /// \param p_valuesNextDate  cash     matrix to store cash  (number of simulations by nb of stocks)
    ContinuationValueTree(const  std::shared_ptr< SpaceGrid >   &p_grid,
                          const std::shared_ptr< Tree >   &p_condExp,
                          const Eigen::ArrayXXd &p_valuesNextDate) :
        m_grid(p_grid),  m_values(p_condExp->expCondMultiple(p_valuesNextDate.transpose()).transpose())
    {
    }

    /// \brief Load another Continuation value object
    ///  Only a partial load of the objects is achieved
    /// \param p_grid   Grid to load
    /// \param p_values  continuation values for all nodes and stocks
    virtual void loadForSimulation(const  std::shared_ptr< SpaceGrid > &p_grid,
                                   const Eigen::ArrayXXd &p_values)
    {
        m_grid = p_grid;
        m_values = p_values ;
    }


    /// \brief Get continuation value for one stock point
    /// \param p_ptOfStock   grid point for interpolation
    /// \return the continuation value associated to each node used in optimization
    Eigen::ArrayXd getValueAtNodes(const Eigen::ArrayXd &p_ptOfStock) const
    {
        return m_grid->createInterpolator(p_ptOfStock)->applyVec(m_values);
    }

    /// \brief Same as before but use an interpolator
    Eigen::ArrayXd getValueAtNodes(const Interpolator   &p_interpol) const
    {
        return p_interpol.applyVec(m_values);
    }


    /// \brief Get a conditional expectation for a given
    /// \param p_node    node number
    /// \param  p_ptOfStock     stock points
    /// \return the continuation value associated to the given node used in optimization
    double  getValueAtANode(const int &p_node, const Eigen::ArrayXd &p_ptOfStock) const
    {
        return m_grid->createInterpolator(p_ptOfStock)->apply(m_values.row(p_node).transpose());
    }


    /// \brief Same as before but use an interpolator
    /// \param p_node    node number
    /// \param  p_ptOfStock     stock points
    /// \return the continuation value associated to the given node used in optimization
    double  getValueAtANode(const int &p_node, const Interpolator   &p_interpol) const
    {
        return p_interpol.apply(m_values.row(p_node).transpose());
    }

    //// \brief Get back
    ///@{
    const Eigen::ArrayXXd &getValues()  const
    {
        return m_values;
    }
    std::shared_ptr< SpaceGrid > getGrid() const
    {
        return m_grid;
    }
    ///@}

};

}


#endif
