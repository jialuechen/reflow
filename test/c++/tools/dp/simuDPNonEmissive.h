
#ifndef SIMUDPNONEMISSIVE_H
#define SIMUDPNONEMISSIVE_H
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include "geners/BinaryFileArchive.hh"
#include "reflow/core/grids/SpaceGrid.h"
#include "reflow/core/utils/StateWithStocks.h"
#include "reflow/dp/OptimizerDPBase.h"
#include "reflow/dp/SimulatorDPBase.h"

double simuDPNonEmissive(const std::shared_ptr<reflow::SpaceGrid> &p_grid,
                         const std::shared_ptr<reflow::OptimizerDPBase > &p_optimize,
                         const std::function<double(const int &, const Eigen::ArrayXd &, const Eigen::ArrayXd &)>  &p_funcFinalValue,
                         const Eigen::ArrayXd &p_pointStock,
                         const std::string   &p_fileToDump,
                         const int  &p_nbSimTostore
#ifdef USE_MPI
                         , const boost::mpi::communicator &p_world
#endif
                        );

#endif /* SIMUDPNONEMISSIVE_H */
