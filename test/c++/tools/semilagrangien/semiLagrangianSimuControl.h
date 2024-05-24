
#ifndef SEMILAGRANGIANSIMUCONTROL_H
#define SEMILAGRANGIANSIMUCONTROL_H
#include <memory>
#include <Eigen/Dense>
#include <functional>
#ifdef USE_MPI
#include "boost/mpi.hpp"
#endif
#include "reflow/semilagrangien/OptimizerSLBase.h"

double semiLagrangianSimuControl(const std::shared_ptr<reflow::SpaceGrid> &p_grid,
                                 const std::shared_ptr<reflow::OptimizerSLBase > &p_optimize,
                                 const std::function<double(const int &, const Eigen::ArrayXd &)>   &p_funcFinalValue,
                                 const int &p_nbStep,
                                 const Eigen::ArrayXd &p_stateInit,
                                 const int &p_initialRegime,
                                 const int &p_nbSimul,
                                 const std::string   &p_fileToDump
#ifdef USE_MPI
                                 , const boost::mpi::communicator &p_world
#endif
                                ) ;

#endif /* SEMILAGRANGIANSIMUCONTROL_H */
