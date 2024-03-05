// Copyright (C) 2021 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#ifndef  RECONSTRUCTPROC0FORINTMPI_H
#define  RECONSTRUCTPROC0FORINTMPI_H
#include <memory>
#include <Eigen/Dense>
#include "libflow/core/grids/RegularSpaceIntGrid.h"

/** reconstructProc0Mpi.h
 *  Permit to reconstruct  the grid on processor 0 of  values associated at a integer point
 *
 */
namespace libflow
{

/// \brief Calculate a function with values spread on processor at a given point for all simulations
/// \param p_point  point where reconstruction should be achieved
/// \param p_grid   global grid of the problem
/// \param p_values local values associated to the processor
/// \param p_bdimToSplit    Dimensions to split for parallelism
/// \param  p_world         MPI communicator
Eigen::ArrayXd  reconstructProc0ForIntMpi(const Eigen::ArrayXi &p_point,
        const std::shared_ptr<libflow::RegularSpaceIntGrid> &p_grid,
        const std::shared_ptr< Eigen::ArrayXXd >    &p_values,
        const Eigen::Array< bool, Eigen::Dynamic, 1>   &p_bdimToSplit,
        const boost::mpi::communicator &p_world);
}
#endif /* RECONSTRUCTPROC0FORINTMPI_H */
