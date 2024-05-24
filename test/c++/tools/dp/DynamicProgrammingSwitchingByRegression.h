
#ifndef DYNAMICPROGRAMMINGSWITCHINGBYREGRESSION_H
#define DYNAMICPROGRAMMINGSWITCHINGBYREGRESSION_H
#include <fstream>
#include <memory>
#include <functional>
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <Eigen/Dense>
#include "reflow/core/grids/RegularSpaceIntGrid.h"
#include "reflow/dp/OptimizerSwitchBase.h"
#include "reflow/regression/BaseRegression.h"

double  DynamicProgrammingSwitchingByRegression(const std::vector< std::shared_ptr<reflow::RegularSpaceIntGrid> >  &p_grid,
        const std::shared_ptr<reflow::OptimizerSwitchBase > &p_optimize,
        const std::shared_ptr<reflow::BaseRegression> &p_regressor,
        const Eigen::ArrayXi &p_pointState,
        const int &p_initialRegime,
        const std::string   &p_fileToDump
#ifdef USE_MPI
        , const boost::mpi::communicator &p_world
#endif
                                               );

#endif /* DYNAMICPROGRAMMINGSWITCINGBYREGRESSION_H */
