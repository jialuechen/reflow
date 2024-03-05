
#ifndef GETCOORDINATEBOUND_H
#define GETCOORDINATEBOUND_H
#include <Eigen/Dense>
#include "libflow/core/sparse/sparseGridUtils.h"

/** \file GetCoordinateBound.h
 * \brief Coordinates when boundary points are present
*  \author Xavier Warin
 */

namespace libflow
{
/// \class GetCoordinateBound GetCoordinateBound.h
///   for a level and a position of a point  get its coordinates
class GetCoordinateBound
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
            if (p_level(id) == 1)
            {
                coord(id) = p_position(id) * 0.5;
            }
            else
            {
                // half size of the mesh
                double xDeltaS2 = deltaSparseMesh[static_cast<size_t>(p_level(id))] ;
                coord(id) = xDeltaS2 * (1. + 2 * p_position(id));
            }
        }
        return coord;
    }
};
}
#endif
