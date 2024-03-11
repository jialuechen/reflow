
#ifndef DYNAMICPROGRAMMINGSWITCHINGBYREGRESSIONDIST_H
#define DYNAMICPROGRAMMINGSWITCHINGBYREGRESSIONDIST_H
#include <fstream>
#include <memory>
#include <functional>
#include <boost/mpi.hpp>
#include <Eigen/Dense>
#include "libflow/core/grids/RegularSpaceIntGrid.h"
#include "libflow/dp/OptimizerSwitchBase.h"
#include "libflow/regression/BaseRegression.h"

double  DynamicProgrammingSwitchingByRegressionDist(const std::vector< std::shared_ptr<libflow::RegularSpaceIntGrid> >  &p_grid,
        const std::shared_ptr<libflow::OptimizerSwitchBase > &p_optimize,
        const std::shared_ptr<libflow::BaseRegression> &p_regressor,
        const Eigen::ArrayXi &p_pointState,
        const int &p_initialRegime,
        const std::string   &p_fileToDump,
        const boost::mpi::communicator &p_world
                                                   );

#endif /* DYNAMICPROGRAMMINGSWITCHINGBYREGRESSIONDIST_H */
