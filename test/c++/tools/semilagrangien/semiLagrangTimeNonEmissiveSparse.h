
#ifndef SEMILAGRANGTIMENONEMISSIVESPARSE_H
#define SEMILAGRANGTIMENONEMISSIVESPARSE_H
#include <memory>
#include <functional>
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <Eigen/Dense>
#include "libflow/core/grids/SparseSpaceGrid.h"
#include "OptimizeSLEmissive.h"

/* \file SemiLagrangTimeNonEmissiveSparse.h
 *  \brief Implement the time recursion  to solve an HJB equation by Semi Lagrangian Schemes using
 */

/// \brief Example of function  achieving a time step resolution of a PDE
/// \param p_grid               The grid defining the resolution domain and the interpolation meshing
/// \param p_optimize           The optimizer  defining a PDE step  (potentially defining the search of an optimal control)
/// \param p_funcInitialValue   Initial value of the PDE (or final value of a control problem)
/// \param p_timeBoundaryFunc   Dirichlet boundary condition
/// \param p_step               time step
/// \param p_nStep              number of time steps
/// \param p_fileToDump        File used to serialize solution at each time step
/// \param p_world             MPI communicator
void  semiLagrangTimeNonEmissiveSparse(const std::shared_ptr<libflow::SparseSpaceGrid> &p_grid,
                                       const std::shared_ptr<libflow::OptimizeSLEmissive> &p_optimize,
                                       const std::function<double(const int &, const Eigen::ArrayXd &)>    &p_funcInitialValue,
                                       const std::function<double(const double &, const int &, const Eigen::ArrayXd &)>   &p_timeBoundaryFunc,
                                       const double &p_step,
                                       const int &p_nStep,
                                       const std::string   &p_fileToDump
#ifdef USE_MPI
                                       , const boost::mpi::communicator &p_world
#endif
                                      ) ;

#endif /* SEMILAGRANGTIMENONEMISSIVESPARSE_H */
