
#ifndef DYNAMICPROGRAMMINGBYREGRESSIONCUTDIST_H
#define DYNAMICPROGRAMMINGBYREGRESSIONCUTDIST_H
#include <fstream>
#include <memory>
#include <functional>
#include <boost/mpi.hpp>
#include <Eigen/Dense>
#include "libflow/core/grids/SpaceGrid.h"
#include "libflow/dp/OptimizerDPCutBase.h"

double  DynamicProgrammingByRegressionCutDist(const std::shared_ptr<libflow::FullGrid> &p_grid,
        const std::shared_ptr<libflow::OptimizerDPCutBase > &p_optimize,
        std::shared_ptr<libflow::BaseRegression> &p_regressor,
        const std::function< Eigen::ArrayXd(const int &, const Eigen::ArrayXd &, const Eigen::ArrayXd &)>   &p_funcFinalValue,
        const Eigen::ArrayXd &p_pointStock,
        const int &p_initialRegime,
        const std::string   &p_fileToDump,
        const bool &p_bOneFile,
        const boost::mpi::communicator &p_world);

#endif /* DYNAMICPROGRAMMINGBYREGRESSIONCUTDIST_H */
