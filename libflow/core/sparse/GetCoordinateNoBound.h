// Copyright (C) 2016 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#ifndef GETCOORDINATENOBOUND_H
#define GETCOORDINATENOBOUND_H
#include <Eigen/Dense>
#include "libflow/core/sparse/sparseGridUtils.h"


/** \file GetCoordinateNoBound.h
 *  \brief Regroup some functions use in sparse grid when no boundary points are present
 *  \author Xavier Warin
 */

namespace libflow
{

/// \class GetCoordinateNoBound GetCoordinateNoBound.h
///   for a level and a position of a point  get its coordinates
class GetCoordinateNoBound
{

public :

    /// \brief Given by  the level and the position of the point, gives it coordinates  in \f$[0,1]\f$
    ///  \param  p_level     level
    ///  \param  p_position position
    ///  \return the coordinates
    Eigen::ArrayXd  operator()(const Eigen::ArrayXc &p_level, const Eigen::ArrayXui   &p_position)
    {
        Eigen::ArrayXd coord(p_level.size());
        for (int id = 0 ; id < p_level.size(); ++id)
        {
            // half size of the mesh
            double xDeltaS2 = deltaSparseMesh[static_cast<size_t>(p_level(id))] ;
            coord(id) = xDeltaS2 * (1. + 2 * p_position(id));
        }
        return coord;
    }
    /// \brief Given by  the level and the position of the point, gives it coordinates  in \f$[0,1]\f$
    ///  \param  p_level     level
    ///  \param  p_position position
    ///  \return the coordinates
    double  operator()(const char &p_level, const unsigned int    &p_position)
    {
        // half size of the mesh
        double xDeltaS2 = deltaSparseMesh[static_cast<size_t>(p_level)] ;
        return  xDeltaS2 * (1. + 2 * p_position);
    }
};
}
#endif
