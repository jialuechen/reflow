
#ifndef DYNAMICPROGRAMMINGBYTREECUT_H
#define DYNAMICPROGRAMMINGBYTREECUT_H
#include <fstream>
#include <memory>
#include <functional>
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <Eigen/Dense>
#include "libflow/core/grids/FullGrid.h"
#include "libflow/dp/OptimizerDPCutTreeBase.h"

double  DynamicProgrammingByTreeCut(const std::shared_ptr<libflow::FullGrid> &p_grid,
                                    const std::shared_ptr<libflow::OptimizerDPCutTreeBase > &p_optimize,
                                    const std::function< Eigen::ArrayXd(const int &, const Eigen::ArrayXd &, const Eigen::ArrayXd &)>   &p_funcFinalValue,
                                    const Eigen::ArrayXd &p_pointStock,
                                    const int &p_initialRegime,
                                    const std::string   &p_fileToDump
#ifdef USE_MPI
                                    , const boost::mpi::communicator &p_world
#endif
                                   );

#endif /* DYNAMICPROGRAMMINGBYTREECUT_H */
