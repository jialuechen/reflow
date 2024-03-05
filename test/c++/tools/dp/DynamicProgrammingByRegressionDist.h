// Copyright (C) 2016 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#ifndef DYNAMICPROGRAMMINGBYREGRESSIONDIST_H
#define DYNAMICPROGRAMMINGBYREGRESSIONDIST_H
#include <fstream>
#include <memory>
#include <functional>
#include <boost/mpi.hpp>
#include <Eigen/Dense>
#include "libflow/core/grids/SpaceGrid.h"
#include "libflow/dp/OptimizerDPBase.h"

/* \file DynamicProgrammingByRegressionDist.h
 * \brief Defines a simple  programm  showing how to optimize a problem by dynamic programming using parallel framework and distributing
 *        calculations and data
 *        A simple grid  is used
 * \author Xavier Warin
 */

/// \brief Principal function to optimize  a problem
/// \param p_grid             grid used for  deterministic state (stocks for example)
/// \param p_optimize         optimizer defining the optimisation between two time steps
/// \param p_regressor        regressor object
/// \param p_funcFinalValue   function defining the final value
/// \param p_initialPointStock     point stock used for interpolation
/// \param p_initialRegime         regime at initial date
/// \param p_fileToDump            file to dump continuation values
/// \param p_bOneFile              do we store continuation values  in only one file
/// \param p_world             MPI communicator
///
double  DynamicProgrammingByRegressionDist(const std::shared_ptr<libflow::FullGrid> &p_grid,
        const std::shared_ptr<libflow::OptimizerDPBase > &p_optimize,
        std::shared_ptr<libflow::BaseRegression> &p_regressor,
        const std::function<double(const int &, const Eigen::ArrayXd &, const Eigen::ArrayXd &)>   &p_funcFinalValue,
        const Eigen::ArrayXd &p_pointStock,
        const int &p_initialRegime,
        const std::string   &p_fileToDump,
        const bool &p_bOneFile,
        const boost::mpi::communicator &p_world);

#endif /* DYNAMICPROGRAMMINGBYREGRESSIONDIST_H */
