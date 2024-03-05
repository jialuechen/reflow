
#ifndef ONEDIMREGULARSPACEGRID_H
#define  ONEDIMREGULARSPACEGRID_H
#include <assert.h>
#include <algorithm>
#include "libflow/core/utils/comparisonUtils.h"

/** \file OneDimRegularSpaceGrid.h
 * \brief Defines a specialization of the RegularSpaceGrid  object in one dimension
 * \author Xavier Warin
 */
namespace libflow
{
/// \class OneDimRegularSpaceGrid OneDimRegularSpaceGrid.h
/// define a grid in one dimension
class OneDimRegularSpaceGrid
{
private :
    double m_lowValue ; ///< minimal value of the mesh
    double m_step; ///< Step size
    int m_nbStep ; ///< Number of step

public :

    /// \brief Default constructor
    OneDimRegularSpaceGrid() {}

    /// \brief Constructor
    /// \param p_lowValue   minimal value of the grid
    /// \param p_step       step size
    /// \param p_nbStep     number of steps
    OneDimRegularSpaceGrid(const double &p_lowValue, const double &p_step, const  int  &p_nbStep) : m_lowValue(p_lowValue), m_step(p_step), m_nbStep(p_nbStep) {}

    /// \brief To a coordinate get back the  mesh number
    /// \param  p_coord   coordinate
    /// \return mesh number associated to the coordinate
    inline int  getMesh(const double   &p_coord) const
    {
        assert(isLesserOrEqual(m_lowValue, p_coord));
        return std::min(roundIntAbove((p_coord - m_lowValue) / m_step), m_nbStep);

    }

    /// \name get back value
    ///@{
    inline double getLowValue() const
    {
        return  m_lowValue ;
    }
    inline double getStep() const
    {
        return m_step;
    }
    inline int getNbStep() const
    {
        return m_nbStep;
    }
    ///@}
};
}
#endif
