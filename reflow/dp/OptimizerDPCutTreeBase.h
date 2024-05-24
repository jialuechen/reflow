
#ifndef OPTIMIZERDPCUTTREEBASE_H
#define OPTIMIZERDPCUTTREEBASE_H
#include <Eigen/Dense>
#include "reflow/core/utils/StateWithStocks.h"
#include "reflow/core/grids/SpaceGrid.h"
#include "reflow/tree/Tree.h"
#include "reflow/tree/StateTreeStocks.h"
#include "reflow/tree/ContinuationCutsTree.h"
#include "reflow/dp/SimulatorDPBaseTree.h"
#include "reflow/dp/OptimizerBase.h"

/** \file OptimizerDPCutBase.h
 *  \brief Define an abstract class for Dynamic Programming problems solved by tree  methods using cust to approximate
 *         Bellman values
 *     \author Xavier Warin
 */

namespace reflow
{

/// \class OptimizerDPCutTreeBase OptimizerDPCutTreeBase.h
///  Base class for optimizer for Dynamic Programming  with tree methods and cuts, so using LP to solve transitional problems
class OptimizerDPCutTreeBase : public OptimizerBase
{


public :

    OptimizerDPCutTreeBase() {}

    virtual ~OptimizerDPCutTreeBase() {}

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
    /// \return    For each regimes (column) gives the solution for each particle , and cut (row)
    ///            For a given simulation , cuts components (C) at a point stock \$ \bar S \f$  are given such that the cut is given by
    ///            \f$  C[0] + \sum_{i=1}^d C[i] (S_i - \bat S_i)   \f$
    virtual Eigen::ArrayXXd   stepOptimize(const   std::shared_ptr< reflow::SpaceGrid> &p_grid, const Eigen::ArrayXd   &p_stock,
                                           const std::vector< reflow::ContinuationCutsTree  > &p_condEsp) const = 0;


    /// \brief defines a step in simulation
    /// Control are recalculated during simulation using a local optimzation using the LP
    /// \param p_grid          grid at arrival step after command
    /// \param p_continuation  defines the continuation operator for each regime
    /// \param p_state         defines the state value (modified)
    /// \param p_phiInOut      defines the value functions (modified) : size number of functions  to follow
    virtual void stepSimulate(const std::shared_ptr< reflow::SpaceGrid>   &p_grid, const std::vector< reflow::ContinuationCutsTree  > &p_continuation,
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
#endif /* OPTIMIZERDPCUTTREEBASE_H */
