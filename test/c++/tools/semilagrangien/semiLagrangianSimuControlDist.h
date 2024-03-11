
#ifndef SEMILAGRANGIANSIMUCONTROLDIST_H
#define SEMILAGRANGIANSIMUCONTROLDIST_H
#include <Eigen/Dense>
#include <memory>
#include <functional>
#include <boost/mpi.hpp>
#include "libflow/semilagrangien/OptimizerSLBase.h"

double semiLagrangianSimuControlDist(const std::shared_ptr<libflow::FullGrid> &p_grid,
                                     const std::shared_ptr<libflow::OptimizerSLBase > &p_optimize,
                                     const std::function<double(const int &, const Eigen::ArrayXd &)>   &p_funcFinalValue,
                                     const int &p_nbStep,
                                     const Eigen::ArrayXd &p_stateInit,
                                     const int &p_initialRegime,
                                     const int &p_nbSimul,
                                     const std::string   &p_fileToDump,
                                     const bool &p_bOneFile,
                                     const boost::mpi::communicator &p_world) ;

#endif /* SEMILAGRANGIANSIMUCONTROLDIST_H */
