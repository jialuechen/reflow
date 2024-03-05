// Copyright (C) 2016 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#ifndef FULLGRID_H
#define FULLGRID_H
#include <iosfwd>
#include <iostream>
#include <Eigen/Dense>
#include "libflow/core/grids/SpaceGrid.h"

/** \file FullGrid.h
 *  \brief Defines a \f$n\f$ dimensional grid  using tensorization in all dimensions
 *  \author Xavier Warin
 */
namespace libflow
{

/// \class FullGrid FullGrid.h
/// Defines full grids
class FullGrid : public SpaceGrid
{

public :

    /// \brief Default constructor
    FullGrid() {}

    /// \brief Default destructor
    virtual ~FullGrid() {}

    /// \brief size of mesh given a lower coordinate
    virtual Eigen::ArrayXd getMeshSize(const Eigen::Ref<const Eigen::ArrayXi > &p_icoord) const = 0;

    /// \name get position coordinate of a point and generical information
    ///@{
    /// lower coordinate of a point (iCoord) in the grid such that each coordinates lies in the mesh defines by \f$ iCoord \f$  and \f$ iCoord+1 \f$
    virtual Eigen::ArrayXi lowerPositionCoord(const Eigen::Ref<const Eigen::ArrayXd > &p_point) const = 0;
    /// upper coordinate of a point (iCoord) in the grid such that each coordinates lies in the mesh defines by \f$ iCoord-1 \f$  and \f$ iCoord \f$
    virtual Eigen::ArrayXi upperPositionCoord(const Eigen::Ref<const Eigen::ArrayXd >   &p_point) const = 0;
    ///@}

    /// \brief transform integer coordinates  to real coordinates
    virtual Eigen::ArrayXd  getCoordinateFromIntCoord(const Eigen::ArrayXi &p_icoord) const = 0;

    /// \brief Coordinate in each direction to global
    virtual int intCoordPerDimToGlobal(const Eigen::ArrayXi &p_iCoord) const = 0;

    ///  number of steps in each dimension
    virtual  const Eigen::ArrayXi    &getDimensions() const = 0;

    /// \brief get sub grid
    /// \param  p_mesh for each dimension give the first point of the mesh and the first outside of the domain
    virtual std::shared_ptr<FullGrid> getSubGrid(const Eigen::Array< std::array<int, 2>, Eigen::Dynamic, 1> &p_mesh) const = 0;


    /// \brief check comptability of the mesh with the number of points in each direction
    /// \param  p_nbPoints   number of points
    virtual bool checkMeshAndPointCompatibility(const int &p_nbPoints) const = 0 ;
};

}

#endif /* FULLGRID_H */
