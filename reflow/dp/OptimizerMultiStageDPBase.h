

#ifndef OPTIMIZERMULTISTAGEDPBASE_H
#define OPTIMIZERMULTISTAGEDPBASE_H
#include <Eigen/Dense>
#include "reflow/core/utils/StateWithStocks.h"
#include "reflow/core/grids/SpaceGrid.h"
#include "reflow/dp/OptimizerBase.h"
#include "reflow/dp/SimulatorMultiStageDPBase.h"
#include "reflow/regression/GridAndRegressedValue.h"
#include "reflow/regression/ContinuationValue.h"

/** \file OptimizerMultiStageDPBase.h
 *  \brief Define an abstract class for Dynamic Programming problems  where each transition problem problem
 *         is itself solved using  a dynamic programming approach
 *     \author Benoit Clair, Xavier Warin
 */

namespace reflow
{

/// \class OptimizerMultiStageDPBase OptimizerMultiStageDPBase.h
///  Base class for optimizer for Dynamic Programming with regressions where each each transition problem
///  is itself solved using a deterministic DP
class OptimizerMultiStageDPBase : public OptimizerBase
{

public :

    OptimizerMultiStageDPBase() {}

    virtual ~OptimizerMultiStageDPBase() {}



    /// \brief defines a step in optimization
    /// \param p_grid           grid at arrival step after command
    /// \param p_stock          coordinates of the stock point to treat
    /// \param p_condEsp        continuation values for each regime
    /// \param p_phiIn          for each regime  gives the solution calculated at the previous step ( next time step by Dynamic Programming resolution) : structure of the 2D array ( nb simulation ,nb stocks )
    /// \return   for each regimes (column) gives the solution for each particle (row)
    [[nodiscard]] virtual  Eigen::ArrayXXd stepOptimize(const std::shared_ptr<reflow::SpaceGrid> &p_grid,
            const Eigen::ArrayXd &p_stock,
            const std::vector< std::shared_ptr<ContinuationValue> > &p_condEsp,
            const std::vector<std::shared_ptr<Eigen::ArrayXXd>> &p_phiIn) const = 0;

    /// \brief defines a step in simulation
    /// Notice that this implementation is not optimal but is convenient if the control is discrete.
    /// By avoiding interpolation in control we avoid non admissible control
    /// Control are recalculated during simulation.
    /// \param p_grid          grid at arrival step after command
    /// \param p_continuation  defines the continuation operator for each regime
    /// \param p_state         defines the state value (modified)
    /// \param p_phiInOut      defines the value functions (modified) : size number of functions  to follow
    virtual void stepSimulate(const std::shared_ptr< reflow::SpaceGrid>   &p_grid, const std::vector< reflow::GridAndRegressedValue  > &p_continuation,
                              reflow::StateWithStocks &p_state,
                              Eigen::Ref<Eigen::ArrayXd> p_phiInOut) const = 0 ;


    // number of regime used in deterministic part
    virtual   int getNbDetRegime() const = 0 ;


    /// \brief get the simulator back
    virtual std::shared_ptr< SimulatorMultiStageDPBase > getSimulator() const = 0;

};
}
#endif /* OPTIMIZERMULTISTAGEDPBASE_H */

