
#ifndef SIMUSLNONEMISSIVESPARSE_H
#define SIMUSLNONEMISSIVESPARSE_H
#include <Eigen/Dense>
#include <memory>
#include <functional>
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include "reflow/core/grids/SparseSpaceGrid.h"
#include "reflow/semilagrangien/OptimizerSLBase.h"


double simuSLNonEmissiveSparse(
    const std::shared_ptr<reflow::SparseSpaceGrid> &p_grid,
    const std::shared_ptr<reflow::OptimizerSLBase > &p_optimize,
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
