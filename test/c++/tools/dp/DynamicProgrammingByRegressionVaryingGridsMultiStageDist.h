#ifndef DYNAMICPROGRAMMINGBYREGRESSIONVARYINGGRIDSMULTISTAGEDIST_H
#define DYNAMICPROGRAMMINGBYREGRESSIONVARYINGGRIDSMULTISTAGEDIST_H
#include <fstream>
#include <memory>
#include <functional>
#include <boost/mpi.hpp>
#include <Eigen/Dense>
#include "reflow/core/grids/FullGrid.h"
#include "reflow/dp/OptimizerMultiStageDPBase.h"

double  DynamicProgrammingByRegressionVaryingGridsMultiStageDist(const std::vector<double>    &p_timeChangeGrid,
        const std::vector<std::shared_ptr<reflow::FullGrid> >   &p_grids,
        const std::shared_ptr<reflow::OptimizerMultiStageDPBase > &p_optimize,
        std::shared_ptr<reflow::BaseRegression> &p_regressor,
        const std::function<double(const int &, const Eigen::ArrayXd &, const Eigen::ArrayXd &)>   &p_funcFinalValue,
        const Eigen::ArrayXd &p_pointStock,
        const int &p_initialRegime,
        const std::string   &p_fileToDump,
        const bool &p_bOneFile,
        const boost::mpi::communicator &p_world);

#endif /* DYNAMICPROGRAMMINGBYREGRESSIONVARYINGGRIDSMULTISTAGEDIST_H */
