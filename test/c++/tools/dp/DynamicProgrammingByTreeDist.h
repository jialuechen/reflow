
#ifndef DYNAMICPROGRAMMINGBYTREEDIST_H
#define DYNAMICPROGRAMMINGBYTREEDIST_H
#include <fstream>
#include <memory>
#include <functional>
#include <boost/mpi.hpp>
#include <Eigen/Dense>
#include "libflow/core/grids/SpaceGrid.h"
#include "libflow/dp/OptimizerDPTreeBase.h"

/* \file DynamicProgrammingByTreeDist.h
 * \brief Defines a simple  programm  showing how to optimize a problem by dynamic programming using parallel framework and distributing
 *        calculations and data
 *        A simple grid  is used and conditional expectation is calculated by trees
 * \author Xavier Warin
 */

/// \brief Principal function to optimize  a problem using trees
/// \param p_grid             grid used for  deterministic state (stocks for example)
/// \param p_optimize         optimizer defining the optimisation between two time steps
/// \param p_funcFinalValue   function defining the final value
/// \param p_initialPointStock     point stock used for interpolation
/// \param p_initialRegime         regime at initial date
/// \param p_fileToDump            file to dump continuation values
/// \param p_bOneFile              do we store continuation values  in only one file
/// \param p_world             MPI communicator
///
double  DynamicProgrammingByTreeDist(const std::shared_ptr<libflow::FullGrid> &p_grid,
                                     const std::shared_ptr<libflow::OptimizerDPTreeBase > &p_optimize,
                                     const std::function<double(const int &, const Eigen::ArrayXd &, const Eigen::ArrayXd &)>   &p_funcFinalValue,
                                     const Eigen::ArrayXd &p_pointStock,
                                     const int &p_initialRegime,
                                     const std::string   &p_fileToDump,
                                     const bool &p_bOneFile,
                                     const boost::mpi::communicator &p_world);

#endif /* DYNAMICPROGRAMMINGBYTREEDIST_H */
