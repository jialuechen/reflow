
#ifdef USE_MPI
#ifndef DYNAMICPROGRAMMINGPORTFOLIODIST_H
#define DYNAMICPROGRAMMINGPORTFOLIODIST_H
#include <fstream>
#include <memory>
#include <functional>
#include <boost/mpi.hpp>
#include <Eigen/Dense>
#include "reflow/core/grids/SpaceGrid.h"
#include "reflow/core/grids/FullGrid.h"
#include "test/c++/tools/dp/OptimizePortfolioDP.h"

double  DynamicProgrammingPortfolioDist(const std::shared_ptr<reflow::FullGrid> &p_grid,
                                        const std::shared_ptr<OptimizePortfolioDP> &p_optimize,
                                        const Eigen::ArrayXi &p_nbMesh,
                                        const std::function<double(const int &, const Eigen::ArrayXd &, const Eigen::ArrayXd &)>  &p_funcFinalValue,
                                        const Eigen::ArrayXd &p_initialPortfolio,
                                        const std::string   &p_fileToDump,
                                        const bool &p_bOneFile,
                                        const boost::mpi::communicator &p_world);
#endif
#endif
