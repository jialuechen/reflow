// Copyright (C) 2016 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
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

/** \file simuDPNonEmissive.h
 *  \brief Defines a simple program showing how to use simulation in a parallel framework
 *         Here Dynamic Programming methods are used to simulate
 *          Aid, Ren, Touzi :
 *        "Transition to non-emissive electricity production under optimal subsidy and endogenous carbon price"
 *        A simple grid  is used
 *  \author Xavier Warin
 */


/// \brief Simulate the optimal strategy , mpi version
/// \param p_grid                   grid used for  deterministic state (stocks for example)
/// \param p_optimize               optimizer defining the optimisation between two time steps
/// \param p_funcFinalValue         function defining the final value
/// \param p_pointStock             initial point stock
/// \param p_fileToDump             name associated to dumped bellman values
/// \param p_nbSimTostore           number of simulations to store
/// \param p_world                  MPI communicator
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
