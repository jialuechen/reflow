
#ifndef DYNAMICPROGRAMMINGBYTREE_H
#define DYNAMICPROGRAMMINGBYTREE_H
#include <fstream>
#include <memory>
#include <functional>
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <Eigen/Dense>
#include "reflow/core/grids/FullGrid.h"
#include "reflow/dp/OptimizerDPTreeBase.h"


double  DynamicProgrammingByTree(const std::shared_ptr<reflow::FullGrid> &p_grid,
                                 const std::shared_ptr<reflow::OptimizerDPTreeBase > &p_optimize,
                                 const std::function<double(const int &, const Eigen::ArrayXd &, const Eigen::ArrayXd &)>   &p_funcFinalValue,
                                 const Eigen::ArrayXd &p_pointStock,
                                 const int &p_initialRegime,
                                 const std::string   &p_fileToDump
#ifdef USE_MPI
                                 , const boost::mpi::communicator &p_world
#endif
                                );

#endif /* DYNAMICPROGRAMMINGBYTREE_H */
