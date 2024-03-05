
#ifndef DYNAMICPROGRAMMINGBYREGRESSIONVARYINGGRIDSDIST_H
#define DYNAMICPROGRAMMINGBYREGRESSIONVARYINGGRIDSDIST_H
#include <fstream>
#include <memory>
#include <functional>
#include <boost/mpi.hpp>
#include <Eigen/Dense>
#include "libflow/core/grids/FullGrid.h"
#include "libflow/dp/OptimizerDPBase.h"

/* \file DynamicProgrammingByRegressionVaryingGridsDist.h
 * \brief Defines a simple  program  showing how to optimize a problem by dynamic programming using parallel framework and distributing
 *        calculations and data
 *        A simple grid  is used
 * \author Xavier Warin
 */

/// \brief Principal function to optimize  a problem
///        The geometry of the  stocks is time dependent
/// \param p_timeChangeGrid    date for changing grids
/// \param p_grids             grids depending on time
/// \param p_optimize         optimizer defining the optimisation between two time steps
/// \param p_regressor        regressor object
/// \param p_funcFinalValue   function defining the final value
/// \param p_initialPointStock     point stock used for interpolation
/// \param p_initialRegime         regime at initial date
/// \param p_fileToDump            file to dump continuation values
/// \param p_bOneFile              do we store continuation values  in only one file
/// \param p_world             MPI communicator
///
double  DynamicProgrammingByRegressionVaryingGridsDist(const std::vector<double>    &p_timeChangeGrid,
        const std::vector<std::shared_ptr<libflow::FullGrid> >   &p_grids,
        const std::shared_ptr<libflow::OptimizerDPBase > &p_optimize,
        std::shared_ptr<libflow::BaseRegression> &p_regressor,
        const std::function<double(const int &, const Eigen::ArrayXd &, const Eigen::ArrayXd &)>   &p_funcFinalValue,
        const Eigen::ArrayXd &p_pointStock,
        const int &p_initialRegime,
        const std::string   &p_fileToDump,
        const bool &p_bOneFile,
        const boost::mpi::communicator &p_world);

#endif /* DYNAMICPROGRAMMINGBYREGRESSIONVARYINGGRIDSDIST_H */
