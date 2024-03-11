
#ifndef SIMUSLNONEMISSIVE_H
#define SIMUSLNONEMISSIVE_H
#include <Eigen/Dense>
#include <memory>
#include <functional>
#ifdef USE_MPI
#include "boost/mpi.hpp"
#endif
#include "libflow/core/grids/FullGrid.h"
#include "libflow/semilagrangien/OptimizerSLBase.h"

double simuSLNonEmissive(const std::shared_ptr<libflow::FullGrid> &p_grid,
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

#endif /* SIMUSLNONEMISSIVE_H */
