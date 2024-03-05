#ifndef DPTIME_H
#define DPTIME_H
#include <fstream>
#include <memory>
#include <functional>
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <Eigen/Dense>
#include "libflow/core/grids/SpaceGrid.h"
#include "libflow/dp/OptimizerDPBase.h"

/* \file DpTimeNonEmissive.h
 * \brief Defines a simple  program  showing how to optimize a problem by dynamic programming with parallel framework
 *        A simple grid  is used See Aid, Ren, Touzi :
 *        "Transition to non-emissive electricity production under optimal subsidy and endogeneous carbon price"
 * \author Xavier Warin
 */

/// \brief Principal function to optimize  the problem
/// \param p_grid             grid used for  deterministic state (stocks for example)
/// \param p_optimize          optimizer defining the optimisation between two time steps
/// \param p_regressor         regressor object
/// \param p_funcFinalValue    function defining the final value
/// \param p_fileToDump        file to dump continuation values
/// \param p_world              MPI communicator
///
void  DpTimeNonEmissive(const std::shared_ptr<libflow::FullGrid> &p_grid,
                        const std::shared_ptr<libflow::OptimizerDPBase > &p_optimize,
                        const std::shared_ptr<libflow::BaseRegression> &p_regressor,
                        const std::function<double(const int &, const Eigen::ArrayXd &, const Eigen::ArrayXd &)>   &p_funcFinalValue,
                        const std::string   &p_fileToDump
#ifdef USE_MPI
                        , const boost::mpi::communicator &p_world
#endif
                       );

#endif /* DPTIMENONEMISSIVE_H */
