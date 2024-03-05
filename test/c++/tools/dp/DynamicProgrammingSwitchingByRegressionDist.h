// Copyright (C) 2021 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
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

/* \file DynamicProgrammingByRegressionDist.h
 * \brief Defines a simple  programm  showing how to optimize a problem by dynamic programming using parallel framework and distributing for a pure switching problem with integer state
 *        calculations and data
 *        A simple grid  is used
 * \author Xavier Warin
 */

/// \brief Principal function to optimize  a problem of pure switching problem
/// \param p_grid              grid used for  deterministic integer state for each regime for switching problems
/// \param p_optimize         optimizer defining the optimisation between two time steps
/// \param p_regressor        regressor object
/// \param p_funcFinalValue   function defining the final value
/// \param p_initialPointStock     point stock used for interpolation
/// \param p_initialRegime         regime at initial date
/// \param p_fileToDump            file to dump continuation values
/// \param p_world             MPI communicator
///
double  DynamicProgrammingSwitchingByRegressionDist(const std::vector< std::shared_ptr<libflow::RegularSpaceIntGrid> >  &p_grid,
        const std::shared_ptr<libflow::OptimizerSwitchBase > &p_optimize,
        const std::shared_ptr<libflow::BaseRegression> &p_regressor,
        const Eigen::ArrayXi &p_pointState,
        const int &p_initialRegime,
        const std::string   &p_fileToDump,
        const boost::mpi::communicator &p_world
                                                   );

#endif /* DYNAMICPROGRAMMINGSWITCHINGBYREGRESSIONDIST_H */
