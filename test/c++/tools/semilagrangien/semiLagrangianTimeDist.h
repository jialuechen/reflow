// Copyright (C) 2016 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#ifndef SEMILAGRANGIANTIMEDIST_H
#define  SEMILAGRANGIANTIMEDIST_H
#include <memory>
#include <functional>
#include <boost/mpi.hpp>
#include <Eigen/Dense>
#include "libflow/core/grids/FullGrid.h"
#include "libflow/semilagrangien/OptimizerSLBase.h"

/* \file SemiLagrangianTimeDist.h
 *  \brief Implement the time recursion  to solve an HJB equation by Semi Lagrangian Schemes using MPI
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
/// \param p_bOneFile              do we store continuation values  in only one file
std::pair<double, double>  semiLagrangianTimeDist(const std::shared_ptr<libflow::FullGrid> &p_grid,
        const std::shared_ptr<libflow::OptimizerSLBase > &p_optimize,
        const std::function<double(const int &, const Eigen::ArrayXd &)>    &p_funcInitialValue,
        const std::function<double(const double &, const int &, const Eigen::ArrayXd &)>   &p_timeBoundaryFunc,
        const double &p_step,
        const int &p_nStep,
        const Eigen::ArrayXd &p_point,
        const int &p_initialRegime,
        const std::function<double(const double &, const Eigen::ArrayXd &)> &p_funcSolution,
        const std::string   &p_fileToDump,
        const bool &p_bOneFile,
        const boost::mpi::communicator &p_world) ;

#endif /* SEMILAGRANGIANTIMEDIST_H */
