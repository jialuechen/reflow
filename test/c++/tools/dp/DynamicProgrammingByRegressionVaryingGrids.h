
#ifndef DYNAMICPROGRAMMINGBYREGRESSIONVARYINGGRIDS_H
#define DYNAMICPROGRAMMINGBYREGRESSIONVARYINGGRIDS_H
#include <fstream>
#include <memory>
#include <functional>
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <Eigen/Dense>
#include "reflow/core/grids/FullGrid.h"
#include "reflow/dp/OptimizerDPBase.h"

double  DynamicProgrammingByRegressionVaryingGrids(const std::vector<double>    &p_timeChangeGrid,
        const std::vector<std::shared_ptr<reflow::FullGrid> >   &p_grids,
        const std::shared_ptr<reflow::OptimizerDPBase > &p_optimize,
        const std::shared_ptr<reflow::BaseRegression> &p_regressor,
        const std::function<double(const int &, const Eigen::ArrayXd &, const Eigen::ArrayXd &)>   &p_funcFinalValue,
        const Eigen::ArrayXd &p_pointStock,
        const int &p_initialRegime,
        const std::string   &p_fileToDump
#ifdef USE_MPI
        , const boost::mpi::communicator &p_world
#endif
                                                  );

#endif /* DYNAMICPROGRAMMINGBYREGRESSIONVARYINGGRIDS_H */
