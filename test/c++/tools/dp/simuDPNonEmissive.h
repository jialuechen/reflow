
#ifndef SIMUDPNONEMISSIVE_H
#define SIMUDPNONEMISSIVE_H
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include "geners/BinaryFileArchive.hh"
#include "libflow/core/grids/SpaceGrid.h"
#include "libflow/core/utils/StateWithStocks.h"
#include "libflow/dp/OptimizerDPBase.h"
#include "libflow/dp/SimulatorDPBase.h"

double simuDPNonEmissive(const std::shared_ptr<libflow::SpaceGrid> &p_grid,
                         const std::shared_ptr<libflow::OptimizerDPBase > &p_optimize,
                         const std::function<double(const int &, const Eigen::ArrayXd &, const Eigen::ArrayXd &)>  &p_funcFinalValue,
                         const Eigen::ArrayXd &p_pointStock,
                         const std::string   &p_fileToDump,
                         const int  &p_nbSimTostore
#ifdef USE_MPI
                         , const boost::mpi::communicator &p_world
#endif
                        );

#endif /* SIMUDPNONEMISSIVE_H */
