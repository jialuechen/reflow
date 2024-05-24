
#ifndef OPTIMIZERDPTREEBASE_H
#define OPTIMIZERDPTREEBASE_H
#include <Eigen/Dense>
#include "reflow/core/grids/SpaceGrid.h"
#include "reflow/tree/Tree.h"
#include "reflow/tree/StateTreeStocks.h"
#include "reflow/tree/ContinuationValueTree.h"
#include "reflow/tree/GridTreeValue.h"
#include "reflow/dp/SimulatorDPBaseTree.h"
#include "reflow/dp/OptimizerBase.h"

/** \file OptimizerDPTreeBase.h
 *  \brief Define an abstract class for Dynamic Programming problems solved by tree  methods
 *     \author Xavier Warin
 */

namespace reflow
{

/// \class OptimizerDPTreeBase OptimizerDPTreeBase.h
///  Base class for optimizer for Dynamic Programming  with tree methods
class OptimizerDPTreeBase : public OptimizerBase
{


public :

    OptimizerDPTreeBase() {}

    virtual ~OptimizerDPTreeBase() {}

    /// \brief defines the diffusion cone for parallelism
    /// \param  p_regionByProcessor         region (min max) treated by the processor for the different regimes treated
    /// \return returns in each dimension the min max values in the stock that can be reached from the grid p_gridByProcessor for each regime
    virtual std::vector< std::array< double, 2> > getCone(const  std::vector<  std::array< double, 2>  > &p_regionByProcessor) const = 0;

    /// \brief defines the dimension to split for MPI parallelism
    ///        For each dimension return true is the direction can be split
    virtual Eigen::Array< bool, Eigen::Dynamic, 1> getDimensionToSplit() const = 0 ;

    /// \brief defines a step in optimization
    /// \param p_grid      grid at arrival step after command
    /// \param p_stock     coordinates of the stock point to treat
    /// \param p_condEsp   continuation values for each regime
    /// \return   a pair  :
    ///              - for each regimes (column) gives the solution for each node in the tree (row)
    ///              - for each control (column) gives the optimal control for each node in the tree (rows)
    ///              .
    virtual std::pair< Eigen::ArrayXXd, Eigen::ArrayXXd>   stepOptimize(const   std::shared_ptr< reflow::SpaceGrid> &p_grid, const Eigen::ArrayXd   &p_stock,
            const std::vector< reflow::ContinuationValueTree  > &p_condEsp) const = 0;


    /// \brief defines a step in simulation
    /// Notice that this implementation is not optimal but is convenient if the control is discrete.
    /// By avoiding interpolation in control we avoid non admissible control
    /// Control are recalculated during simulation.
    /// \param p_grid          grid at arrival step after command
    /// \param p_continuation  defines the continuation operator for each regime
    /// \param p_state         defines the state value (modified)
    /// \param p_phiInOut      defines the value functions (modified) : size number of functions  to follow
    virtual void stepSimulate(const std::shared_ptr< reflow::SpaceGrid>   &p_grid, const std::vector< reflow::GridTreeValue  > &p_continuation,
                              reflow::StateTreeStocks &p_state,
                              Eigen::Ref<Eigen::ArrayXd> p_phiInOut) const = 0 ;


    /// \brief Defines a step in simulation using interpolation in controls
    /// \param p_grid          grid at arrival step after command
    /// \param p_control       defines the controls
    /// \param p_state         defines the state value (modified)
    /// \param p_phiInOut      defines the value function (modified): size number of functions to follow
    virtual void stepSimulateControl(const std::shared_ptr< reflow::SpaceGrid>   &p_grid, const std::vector< reflow::GridTreeValue  > &p_control,
                                     reflow::StateTreeStocks &p_state,
                                     Eigen::Ref<Eigen::ArrayXd> p_phiInOut) const = 0 ;



    /// \brief Get the number of regimes allowed for the asset to be reached  at the current time step
    ///    If \f$ t \f$ is the current time, and $\f$ dt \f$  the resolution step,  this is the number of regime allowed on \f$[ t- dt, t[\f$
    virtual   int getNbRegime() const = 0 ;

    /// \brief get the simulator back
    virtual std::shared_ptr< reflow::SimulatorDPBaseTree > getSimulator() const = 0;

    /// \brief get back the dimension of the control
    virtual int getNbControl() const = 0 ;

    /// \brief get size of the  function to follow in simulation
    virtual int getSimuFuncSize() const = 0;

};
}
#endif /* OPTIMIZERDPTREEBASE_H */
