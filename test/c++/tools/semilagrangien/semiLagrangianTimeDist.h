
#ifndef SEMILAGRANGIANTIMEDIST_H
#define  SEMILAGRANGIANTIMEDIST_H
#include <memory>
#include <functional>
#include <boost/mpi.hpp>
#include <Eigen/Dense>
#include "libflow/core/grids/FullGrid.h"
#include "libflow/semilagrangien/OptimizerSLBase.h"

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
