
#ifndef SIMULATESTEPBASE_H
#define SIMULATESTEPBASE_H
#include <vector>
#include <Eigen/Dense>
#include "libflow/core/utils/StateWithStocks.h"

/** SimulateStepBase.h
 *  Base class for one time step of simulation
 * \author Xavier Warin
 */

namespace libflow
{
/// \class SimulateStepBase SimulateStepBase.h
/// One time  step  in simulation
class SimulateStepBase
{

public :
    /// \brief Define one step arbitraging between possible commands
    /// \param p_statevector    Vector of states (regime, stock descriptor, uncertainty)
    /// \param p_phiInOut       actual contract value modified at current time step by applying an optimal command (size number of functions by number of simulations) in regression methods and (size number of functions by number of nodes at next date) for trees
    virtual void oneStep(std::vector<StateWithStocks > &p_statevector, Eigen::ArrayXXd  &p_phiInOut) const = 0;
};

}


#endif /* SIMULATESTEPBASE_H */
