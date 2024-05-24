
#ifndef ONEDIMSPACEGRID_H
#define  ONEDIMSPACEGRID_H
#include "Eigen/Dense"
#include "reflow/core/utils/comparisonUtils.h"

/** \file OneDimSpaceGrid.h
 * \brief Defines a specialization of the SpaceGrid  object in one dimension
 *        the grid is only a set of points
 * \author Xavier Warin
 */
namespace reflow
{
/// \class OneDimSpaceGrid OneDimSpaceGrid.h
/// define a grid in one dimension
class OneDimSpaceGrid
{
private :

    Eigen::ArrayXd m_values ; ///< set of points with increasing coordinate


public :

    /// \brief Default constructor
    OneDimSpaceGrid() {}

    /// \brief Constructor
    /// \param p_values values of the grid
    OneDimSpaceGrid(const Eigen::ArrayXd &p_values) : m_values(p_values) {}

    /// \brief To a coordinate get back the  mesh number
    /// \param  p_coord   coordinate
    /// \return mesh number associated to the coordinate
    inline int  getMesh(const double   &p_coord) const
    {
        assert(isLesserOrEqual(m_values(0), p_coord));
        int ipos = m_values.size() - 1 ;
        while (isStrictlyLesser(p_coord, m_values(ipos))) ipos--;
        return ipos;

    }

    /// \brief get back the number of steps
    inline int getNbStep() const
    {
        return m_values.size() - 1;
    }

    /// \brief get value
    /// \param p_iStep  position in the array
    inline double getValue(const int &p_iStep) const
    {
        return m_values(p_iStep);
    }
};
}
#endif /* ONEDIMESPACEGRID_H */
