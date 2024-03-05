
#ifndef DYNAMICPROGRAMMINGBYREGRESSION_H
#define DYNAMICPROGRAMMINGBYREGRESSION_H
#include <fstream>
#include <memory>
#include <functional>
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <Eigen/Dense>
#include "libflow/core/grids/FullGrid.h"
#include "libflow/dp/OptimizerDPBase.h"

/* \file DynamicProgrammingByRegression.h
 * \brief Defines a simple  programm  showing how to optimize a problem by dynamic programming
 *        A simple grid  is used
 * \author Xavier Warin
 */

/// \brief Principal function to optimize  a problem
/// \param p_grid             grid used for  deterministic state (stocks for example)
/// \param p_optimize          optimizer defining the optimisation between two time steps
/// \param p_regressor         regressor object
/// \param p_funcFinalValue    function defining the final value
/// \param p_pointStock        point stock used for interpolation at initial date
/// \param p_initialRegime     regime at initial date
/// \param p_fileToDump        file to dump continuation values
/// \param p_world             MPI communicator
///
double  DynamicProgrammingByRegression(const std::shared_ptr<libflow::FullGrid> &p_grid,
                                       const std::shared_ptr<libflow::OptimizerDPBase > &p_optimize,
                                       const std::shared_ptr<libflow::BaseRegression> &p_regressor,
                                       const std::function<double(const int &, const Eigen::ArrayXd &, const Eigen::ArrayXd &)>   &p_funcFinalValue,
                                       const Eigen::ArrayXd &p_pointStock,
                                       const int &p_initialRegime,
                                       const std::string   &p_fileToDump
#ifdef USE_MPI
                                       , const boost::mpi::communicator &p_world
#endif
                                      );

#endif /* DYNAMICPROGRAMMINGBYREGRESSION_H */
