
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
