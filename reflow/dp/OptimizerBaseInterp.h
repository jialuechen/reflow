
#ifndef OPTIMIZERBASEINTERP_H
#define OPTIMIZERBASEINTERP_H
#include <Eigen/Dense>
#include "reflow/core/utils/StateWithStocks.h"
#include "reflow/core/grids/SpaceGrid.h"
#include "reflow/regression/BaseRegression.h"
#include "reflow/regression/ContinuationValue.h"
#include "reflow/regression/GridAndRegressedValue.h"
#include "reflow/dp/OptimizerBase.h"
#include "reflow/dp/SimulatorDPBase.h"

/** \file OptimizerBaseInterp.h
 *  \brief Define an abstract class for Dynamic Programming problems solved by Monte Carlo methods  with interpolation for Bellman values
 *     \author Xavier Warin
 */

namespace reflow
{

/// \class OptimizerBaseInterp OptimizerBaseInterp.h
///  Base class for optimizer for Dynamic Programming  with interpolation for Bellman values  with and without regression methods
class OptimizerBaseInterp : public OptimizerBase
{


public :

    OptimizerBaseInterp() {}

    virtual ~OptimizerBaseInterp() {}


    /// \brief Defines a step in simulation using interpolation in controls
    /// \param p_grid          grid at arrival step after command
    /// \param p_control       defines the controls
    /// \param p_state         defines the state value (modified)
    /// \param p_phiInOut      defines the value function (modified): size number of functions to follow
    virtual void stepSimulateControl(const std::shared_ptr< reflow::SpaceGrid>   &p_grid, const std::vector< reflow::GridAndRegressedValue  > &p_control,
                                     reflow::StateWithStocks &p_state,
                                     Eigen::Ref<Eigen::ArrayXd> p_phiInOut) const = 0 ;

    /// \brief get the simulator back
    virtual std::shared_ptr< reflow::SimulatorDPBase > getSimulator() const = 0;


};
}
#endif /* OPTIMIZERBASEINTERP_H */
