#ifndef DPTIMESPARSE_H
#define DPTIMESPARSE_H
#include <fstream>
#include <memory>
#include <functional>
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <Eigen/Dense>
#include "reflow/core/grids/SparseSpaceGrid.h"
#include "reflow/dp/OptimizerDPBase.h"

void  DpTimeNonEmissiveSparse(const std::shared_ptr<reflow::SparseSpaceGrid> &p_grid,
                              const std::shared_ptr<reflow::OptimizerDPBase > &p_optimize,
                              const std::shared_ptr<reflow::BaseRegression> &p_regressor,
                              const std::function<double(const int &, const Eigen::ArrayXd &, const Eigen::ArrayXd &)>   &p_funcFinalValue,
                              const std::string   &p_fileToDump
#ifdef USE_MPI
                              , const boost::mpi::communicator &p_world
#endif

                             );

#endif /* DPTIMENONEMISSIVESPARSE_H */
