
#ifndef DYNAMICPROGRAMMINGBYTREEDIST_H
#define DYNAMICPROGRAMMINGBYTREEDIST_H
#include <fstream>
#include <memory>
#include <functional>
#include <boost/mpi.hpp>
#include <Eigen/Dense>
#include "libflow/core/grids/SpaceGrid.h"
#include "libflow/dp/OptimizerDPTreeBase.h"

double  DynamicProgrammingByTreeDist(const std::shared_ptr<libflow::FullGrid> &p_grid,
                                     const std::shared_ptr<libflow::OptimizerDPTreeBase > &p_optimize,
                                     const std::function<double(const int &, const Eigen::ArrayXd &, const Eigen::ArrayXd &)>   &p_funcFinalValue,
                                     const Eigen::ArrayXd &p_pointStock,
                                     const int &p_initialRegime,
                                     const std::string   &p_fileToDump,
                                     const bool &p_bOneFile,
                                     const boost::mpi::communicator &p_world);

#endif /* DYNAMICPROGRAMMINGBYTREEDIST_H */
