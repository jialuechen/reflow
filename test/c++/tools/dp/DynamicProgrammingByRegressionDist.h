
#ifndef DYNAMICPROGRAMMINGBYREGRESSIONDIST_H
#define DYNAMICPROGRAMMINGBYREGRESSIONDIST_H
#include <fstream>
#include <memory>
#include <functional>
#include <boost/mpi.hpp>
#include <Eigen/Dense>
#include "reflow/core/grids/SpaceGrid.h"
#include "reflow/dp/OptimizerDPBase.h"

double  DynamicProgrammingByRegressionDist(const std::shared_ptr<reflow::FullGrid> &p_grid,
        const std::shared_ptr<reflow::OptimizerDPBase > &p_optimize,
        std::shared_ptr<reflow::BaseRegression> &p_regressor,
        const std::function<double(const int &, const Eigen::ArrayXd &, const Eigen::ArrayXd &)>   &p_funcFinalValue,
        const Eigen::ArrayXd &p_pointStock,
        const int &p_initialRegime,
        const std::string   &p_fileToDump,
        const bool &p_bOneFile,
        const boost::mpi::communicator &p_world);

#endif /* DYNAMICPROGRAMMINGBYREGRESSIONDIST_H */
