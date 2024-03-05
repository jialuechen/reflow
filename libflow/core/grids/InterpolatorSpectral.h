// Copyright (C) 2016 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#ifndef INTERPOLATORSPECTRAL_H
#define INTERPOLATORSPECTRAL_H
#include <Eigen/Dense>
//#include "libflow/core/grids/SpaceGrid.h"

/** \file InterpolatorSpectral.h
 *  \brief Defines an interpolator for a grid : here is a global interpolator, storing the representation of the  function
 *         to interpolate : this interpolation is effective when interpolating the same function many times at different points
 *         Here it is an abstract class
 * \author Xavier Warin
 */
namespace libflow
{

/// forward declaration
class SpaceGrid ;

/// \class InterpolatorSpectral InterpolatorSpectral.h
/// Abstract class for spectral operator
class InterpolatorSpectral
{

public :
    virtual ~InterpolatorSpectral() {}

    /**  \brief  interpolate
     *  \param  p_point  coordinates of the point for interpolation
     *  \return interpolated value
     */
    virtual double apply(const Eigen::ArrayXd &p_point) const = 0;


    /** \brief Affect the grid
     * \param p_grid  the grid to affect
     */
    virtual void setGrid(const libflow::SpaceGrid *p_grid)  = 0 ;

    /** \brief Get back grid associated to operator
     */
    virtual  const libflow::SpaceGrid *getGrid() = 0;
};
}
#endif
