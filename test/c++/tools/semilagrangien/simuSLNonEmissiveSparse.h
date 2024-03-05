
#ifndef SIMUSLNONEMISSIVESPARSE_H
#define SIMUSLNONEMISSIVESPARSE_H
#include <Eigen/Dense>
#include <memory>
#include <functional>
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include "libflow/core/grids/SparseSpaceGrid.h"
#include "libflow/semilagrangien/OptimizerSLBase.h"

/** \file simuSLNonEmissive.h
 *  \brief Defines a simple program showing how to use simulation
 *        A sparse grid is used
 *  \author Xavier Warin
 */

/// \brief Simulate the optimal strategy , mpi version
/// \param p_grid                  grid used for PDE
/// \param p_optimize              optimizer defining the optimisation between two time steps
/// \param p_funcFinalValue        function defining the final value (initial value for PDE)
/// \param p_nbStep                 number of step
/// \param p_stateInit             initial state
/// \param p_initialRegime         regime at initial date
/// \param p_nbSimul               number of simulations
/// \param p_fileToDump            name of the file used to dump continuation values in optimization
/// \param p_bOneFile               do we store continuation values  in only one file
/// \param p_nbSimTostore           number of simulations to store
/// \param p_world                 MPI communicator

double simuSLNonEmissiveSparse(
    const std::shared_ptr<libflow::SparseSpaceGrid> &p_grid,
    const std::shared_ptr<libflow::OptimizerSLBase > &p_optimize,
    const std::function<double(const int &, const Eigen::ArrayXd &)>   &p_funcFinalValue,
    const int &p_nbStep,
    const Eigen::ArrayXd &p_stateInit,
    const int &p_nbSimul,
    const std::string   &p_fileToDump,
    const int  &p_nbSimTostore
#ifdef USE_MPI
    , const boost::mpi::communicator &p_world
#endif
) ;

#endif /* SIMUSLNONEMISSIVESPARSE_H */
