
#ifndef DYNAMICPROGRAMMINGSWITCHINGBYREGRESSION_H
#define DYNAMICPROGRAMMINGSWITCHINGBYREGRESSION_H
#include <fstream>
#include <memory>
#include <functional>
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <Eigen/Dense>
#include "libflow/core/grids/RegularSpaceIntGrid.h"
#include "libflow/dp/OptimizerSwitchBase.h"
#include "libflow/regression/BaseRegression.h"

/* \file DynamicProgrammingSwitchingByRegression.h
 * \brief Defines a simple  programm  showing how to optimize a problem by dynamic programming
 *        in the case of a pure switching problem. No stock is involved and then no interpolation is needed
 *        The deterministic state is integer.
 * \author Xavier Warin
 */

/// \brief Principal function to optimize  a problem of switching
/// \param p_grid              grid used for  deterministic integer state for each regime for switching problems
/// \param p_optimize          optimizer defining the optimisation between two time steps
/// \param p_regressor         regressor object
/// \param p_pointState        integer point state used for  at initial date
/// \param p_initialRegime     regime at initial date
/// \param p_fileToDump        file to dump continuation values
/// \param p_world             MPI communicator
///
double  DynamicProgrammingSwitchingByRegression(const std::vector< std::shared_ptr<libflow::RegularSpaceIntGrid> >  &p_grid,
        const std::shared_ptr<libflow::OptimizerSwitchBase > &p_optimize,
        const std::shared_ptr<libflow::BaseRegression> &p_regressor,
        const Eigen::ArrayXi &p_pointState,
        const int &p_initialRegime,
        const std::string   &p_fileToDump
#ifdef USE_MPI
        , const boost::mpi::communicator &p_world
#endif
                                               );

#endif /* DYNAMICPROGRAMMINGSWITCINGBYREGRESSION_H */
