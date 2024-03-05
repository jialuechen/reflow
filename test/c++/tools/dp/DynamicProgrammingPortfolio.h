
#ifndef DYNAMICPROGRAMMINGPORTFOLIO_H
#define DYNAMICPROGRAMMINGPORTFOLIO_H
#include <fstream>
#include <memory>
#include <functional>
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <Eigen/Dense>
#include "libflow/core/grids/SpaceGrid.h"
#include "libflow/core/grids/FullGrid.h"
#include "test/c++/tools/dp/OptimizePortfolioDP.h"

/* \file DynamicProgrammingPortfolio.h
 * \brief Defines a simple  programm  showing how to optimize a portfolio with Monte Carlo (according to a given criterium in expectation)  distributing calculations
 *        A simple grid  is used
 *        The optimization is rather generic.
 * \author Xavier Warin
 */

/// \brief Principal function to optimize  the portfolio
/// \param p_grid                  grid used for portfolio discretization
/// \param p_optimize              optimizer defining the optimization problem for the portfolio
/// \param p_nbMesh                number of mesh in each direction (here only the asset value is discretized)
/// \param p_funcFinalValue        function defining the final value (the payoff associated to  the portfolio)
/// \param p_initialPortfolio      initial portfolio value
/// \param p_fileToDump            file to dump optimal command
/// \param p_world             MPI communicator
///
double  DynamicProgrammingPortfolio(const std::shared_ptr<libflow::FullGrid> &p_grid,
                                    const std::shared_ptr<OptimizePortfolioDP> &p_optimize,
                                    const Eigen::ArrayXi &p_nbMesh,
                                    const std::function<double(const int &, const Eigen::ArrayXd &, const Eigen::ArrayXd &)>  &p_funcFinalValue,
                                    const Eigen::ArrayXd &p_initialPortfolio,
                                    const std::string   &p_fileToDump
#ifdef USE_MPI
                                    , const boost::mpi::communicator &p_world
#endif
                                   );
#endif
