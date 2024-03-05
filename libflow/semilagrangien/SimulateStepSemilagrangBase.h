// Copyright (C) 2016 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#ifndef SIMULATESTEPSEMILAGRANGBASE_H
#define SIMULATESTEPSEMILAGRANGBASE_H
#include <Eigen/Dense>

/** \file SimulateStepSemilagrangBase.h
 *  \brief  In simulation part, permits to  use  the PDE function  value to
 *          implement an optimal strategy
 *  \author Xavier Warin
 */

namespace libflow
{

/// \class SimulateStepSemilagrangBase SimulateStepSemilagrangBase.h
/// One step in forward simulation
class SimulateStepSemilagrangBase
{

public :

    /// \brief Define one step arbitraging between possible commands
    /// \param p_gaussian       2 dimensional Gaussian array (size : number of Brownians motion by number of simulations)
    /// \param p_statevector    Vector of states for each simulation (size of the state by the number of simulations) in the current regime
    /// \param p_iReg           regime number for each simulation
    /// \param p_phiInOut       actual contract values modified at current time step by applying an optimal command
    virtual void oneStep(const Eigen::ArrayXXd   &p_gaussian, Eigen::ArrayXXd &p_statevector, Eigen::ArrayXi   &p_iReg, Eigen::ArrayXXd  &p_phiInOut) const = 0;

};
}
#endif /* SIMULATESTEPSEMILAGRANGBASE_H */
