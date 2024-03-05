// Copyright (C) 2016 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#ifndef SPACEGRID_H
#define SPACEGRID_H
#include <array>
#include <memory>
#include <Eigen/Dense>
#include "libflow/core/grids/GridIterator.h"
#include "libflow/core/grids/Interpolator.h"
#include "libflow/core/grids/InterpolatorSpectral.h"

/** \file SpaceGrid.h
 *  \brief Defines a  base class for all the grids
 * \author Xavier Warin
 */
namespace libflow
{

/// \class SpaceGrid SpaceGrid.h
/// Defines a  base class for grids
class SpaceGrid
{
public :
    /// \brief Default constructor
    SpaceGrid() {}

    /// \brief Default destructor
    virtual ~SpaceGrid() {}

    /// \brief Number of points of the grid
    virtual  size_t getNbPoints() const = 0;

    /// \brief get back iterator associated to the grid
    virtual  std::shared_ptr< GridIterator> getGridIterator() const = 0;

    /// \brief get back iterator associated to the grid (multi thread)
    virtual std::shared_ptr< GridIterator> getGridIteratorInc(const int &p_iThread) const = 0;

    /// \brief  Get back interpolator at a point Interpolate on the grid
    /// \param  p_coord   coordinate of the point for interpolation
    /// \return interpolator at the point coordinates on the grid
    virtual std::shared_ptr<Interpolator> createInterpolator(const Eigen::ArrayXd &p_coord) const = 0;

    /// \brief Get back a spectral operator associated to a whole function
    /// \param p_values   Function value at the grids points
    /// \return  the whole interpolated  value function
    virtual std::shared_ptr<InterpolatorSpectral> createInterpolatorSpectral(const Eigen::ArrayXd &p_values) const = 0;

    /// \brief Dimension of the grid
    virtual  int getDimension() const = 0 ;

    /// \brief get back bounds associated to the grid
    /// \return in each dimension give the extreme values (min, max) of the domain
    virtual std::vector <std::array< double, 2>  > getExtremeValues() const = 0;

    /// \brief test if the point is strictly inside the domain
    /// \param   p_point point to test
    /// \return  true if the point is strictly inside the open domain
    virtual bool isStrictlyInside(const Eigen::ArrayXd &p_point) const = 0 ;

    /// \brief test if a point is  inside the grid (boundary include)
    /// \param  p_point point to test
    /// \return true if the point is inside the open domain
    virtual bool isInside(const Eigen::ArrayXd &p_point) const = 0 ;

    /// \brief truncate a point so that it stays inside the domain
    /// \param p_point  point to truncate
    virtual void truncatePoint(Eigen::ArrayXd &p_point) const = 0 ;

};
}
#endif /* SPACEGRID.H */
