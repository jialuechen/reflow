
#ifndef SEMILAGRANGTIMENONEMISSIVE_H
#define SEMILAGRANGTIMENONEMISSIVE_H
#include <memory>
#include <functional>
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <Eigen/Dense>
#include "libflow/core/grids/FullGrid.h"
#include "OptimizeSLEmissive.h"

void  semiLagrangTimeNonEmissive(const std::shared_ptr<libflow::FullGrid> &p_grid,
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

#endif /* SEMILAGRANGTIMENONEMISSIVE_H */
