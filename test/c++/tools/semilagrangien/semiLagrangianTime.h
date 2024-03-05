
#ifndef SEMILAGRANGIANTIME_H
#define  SEMILAGRANGIANTIME_H
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <memory>
#include <functional>
#include <Eigen/Dense>
#include "libflow/core/grids/SpaceGrid.h"
#include "libflow/semilagrangien/OptimizerSLBase.h"

/* \file SemiLagrangianTime.h
 *  \brief Implement the time recursion  to solve an HJB equation by Semi Lagrangian Schemes
 */

/// \brief Exemple of function  achieving a time step resolution of a PDE
/// \param p_grid               The grid difining the resolution domain and the interpôlation meshing
/// \param p_optimize           The optimizer  defining a PDE step  (potentially defining the search of an optimal control)
/// \param p_funcInitialValue   Initial value of the PDE (or final value of a control problem)
/// \param p_timeBoundaryFunc   Dirichlet boundary condition
/// \param p_step               time step
/// \param p_nStep              number of time steps
/// \param p_point              point where to get the solution
/// \param p_initialRegime      Initial regime  (for regime switching problems)
/// \param p_funcSolution       Analytic solution  (to compute the max error)
/// \param p_fileToDump        File used to serialize solution at each time step
/// \param p_world             MPI communicator
std::pair<double, double>  semiLagrangianTime(const std::shared_ptr<libflow::SpaceGrid> &p_grid,
        const std::shared_ptr<libflow::OptimizerSLBase > &p_optimize,
        const std::function<double(const int &, const Eigen::ArrayXd &)>    &p_funcInitialValue,
        const std::function<double(const double &, const int &, const Eigen::ArrayXd &)>   &p_timeBoundaryFunc,
        const double &p_step,
        const int &p_nStep,
        const Eigen::ArrayXd &p_point,
        const int &p_initialRegime,
        const std::function<double(const double &, const Eigen::ArrayXd &)> &p_funcSolution,
        const std::string   &p_fileToDump
#ifdef USE_MPI
        , const boost::mpi::communicator &p_world
#endif
                                             ) ;

#endif /* SEMILAGRANGIANTIME_H */
