
#ifndef GRIDTREEVALUE_H
#define GRIDTREEVALUE_H


/** \file GridTreeValue.h
 * \brief Permits du store some values on a grid and to interpolate on grid
 *  \author Xavier Warin
 */

namespace libflow
{
/// \class GridTreeValue GridTreeValue.h
/// Permits to store  values  defined on a grid : at each point on the grid a set of/// values is defined
class GridTreeValue
{

private :
    std::shared_ptr< SpaceGrid >    m_grid ; ///< grid used to define stock points
    std::vector< std::shared_ptr<InterpolatorSpectral> > m_interpFuncBasis; /// vector of spectral operator associated to the grid used for each point

public :

    /// \brief Default constructor
    GridTreeValue() {}

    /// \brief Constructor
    /// \param p_grid     grid for stocks
    /// \param p_values   functions to store  (number of node in the tree  by nb of stocks)
    GridTreeValue(const  std::shared_ptr< SpaceGrid >   &p_grid,
                  const Eigen::ArrayXXd &p_values) : m_grid(p_grid), m_interpFuncBasis(p_values.rows())
    {
        for (int i = 0; i < p_values.rows(); ++i)
            m_interpFuncBasis[i] = p_grid->createInterpolatorSpectral(p_values.row(i).transpose());
    }

    /// \brief Constructor used to store the grid and the regressor
    /// \param p_grid     grid for stocks
    GridTreeValue(const  std::shared_ptr< SpaceGrid >   &p_grid) : m_grid(p_grid) {}


    /// \brief Constructor used for deserialization
    /// \param  p_grid                grid for stocks
    /// \param  p_interpFuncBasis     spectral interpolator associated to each regression function basis
    GridTreeValue(const  std::shared_ptr< SpaceGrid >   &p_grid,
                  const std::vector< std::shared_ptr<InterpolatorSpectral> > &p_interpFuncBasis):  m_grid(p_grid),  m_interpFuncBasis(p_interpFuncBasis) {}



    /// \brief Get value function for one stock and one node in tree
    /// \param  p_ptOfStock     stock points
    /// \param  p_node          node number
    inline double getValue(const Eigen::ArrayXd &p_ptOfStock, const int   &p_node) const
    {
        return  m_interpFuncBasis[p_node]->apply(p_ptOfStock);
    }

    /// \brief Get value function for one stock and all nodes
    /// \param  p_ptOfStock     stock points
    inline Eigen::ArrayXd  getValues(const Eigen::ArrayXd &p_ptOfStock) const
    {
        Eigen::ArrayXd ret(m_interpFuncBasis.size());
        for (size_t inode = 0; inode < m_interpFuncBasis.size(); ++inode)
            ret(inode) = m_interpFuncBasis[inode]->apply(p_ptOfStock);
        return ret;
    }

    /// \brief Get the grid
    std::shared_ptr< SpaceGrid >  getGrid() const
    {
        return m_grid ;
    }

    /// \brief Get back the interpolators
    const std::vector< std::shared_ptr<InterpolatorSpectral> >   &getInterpolators() const
    {
        return m_interpFuncBasis;
    }

};
}
#endif
