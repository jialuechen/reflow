#ifndef DPTIME_H
#define DPTIME_H
#include <fstream>
#include <memory>
#include <functional>
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <Eigen/Dense>
#include "reflow/core/grids/SpaceGrid.h"
#include "reflow/dp/OptimizerDPBase.h"


void  DpTimeNonEmissive(const std::shared_ptr<reflow::FullGrid> &p_grid,
                        const std::shared_ptr<reflow::OptimizerDPBase > &p_optimize,
                        const std::shared_ptr<reflow::BaseRegression> &p_regressor,
                        const std::function<double(const int &, const Eigen::ArrayXd &, const Eigen::ArrayXd &)>   &p_funcFinalValue,
                        const std::string   &p_fileToDump
#ifdef USE_MPI
                        , const boost::mpi::communicator &p_world
#endif
                       );

#endif /* DPTIMENONEMISSIVE_H */
