// Copyright (C) 2014 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#ifndef  RECONSTRUCTPROC0MPI_H
#define  RECONSTRUCTPROC0MPI_H
#include <memory>
#include <Eigen/Dense>
#include "libflow/core/grids/FullGrid.h"

/** reconstructProc0Mpi.h
 *  Permit to reconstruct  the grid on processor 0 of all the points needed for interpolation at a given point
 *
 */
namespace libflow
{

/// \brief Calculate a function with values spread on processor at a given point for all simulations
/// \param p_point  point where reconstruction should be achieved
/// \param p_grid   global grid of the problem
/// \param p_values local values associated to the processor
/// \param p_bdimToSplit    Dimensions to split for parallelism
Eigen::ArrayXd  reconstructProc0Mpi(const Eigen::ArrayXd &p_point, const std::shared_ptr< FullGrid> &p_grid, const std::shared_ptr< Eigen::ArrayXXd >    &p_values,
                                    const Eigen::Array< bool, Eigen::Dynamic, 1>   &p_bdimToSplit, const boost::mpi::communicator &p_world);
}
#endif /* RECONSTRUCTPROC0MPI_H */
