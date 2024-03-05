// Copyright (C) 2021 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#ifndef OPTIMIZERSWITCHBASE_H
#define OPTIMIZERSWITCHBASE_H
#include <Eigen/Dense>
#include "libflow/core/utils/types.h"
#include "libflow/core/utils/StateWithIntState.h"
#include "libflow/regression/BaseRegression.h"
#include "libflow/dp/SimulatorDPBase.h"
#include "libflow/core/grids/RegularSpaceIntGrid.h"

/** \file OptimizerSwitchingBase.h
 *  \brief Define an abstract class for Switching problems solved by regression  methods and with an integer state
 *     \author Xavier Warin
 */

namespace libflow
{

/// \class OptimizerSwitchBase OptimizerSwitchBase.h
///  Base class for optimizer for Dynamic Programming  with regression methods
class OptimizerSwitchBase
{


public :

    OptimizerSwitchBase() {}

    virtual ~OptimizerSwitchBase() {}

    /// \brief defines the diffusion cone for parallelism
    /// \param  p_iRegime                   regime used
    /// \param  p_regionByProcessor         region (min max) treated by the processor for the different regimes treated
    /// \return returns in each dimension the min max values in the stock that can be reached from the grid p_gridByProcessor for each regime
    virtual std::vector< std::array<int, 2 > >  getCone(const int &p_iReg, const std::vector< std::array<int, 2 > >   &p_regionByProcessor) const = 0;

    /// \brief defines the dimension to split for MPI parallelism
    ///        For each regime : for each dimension return true is the direction can be split
    virtual std::vector< Eigen::Array< bool, Eigen::Dynamic, 1> > getDimensionToSplit() const = 0 ;

    /// \brief defines a step in optimization
    /// \param p_grid      grid at arrival step after command (integer states) for each regime
    /// \param p_iReg      regime treated
    /// \param p_state     coordinates of the deterministic integer state
    /// \param p_condExp   Conditional expectation operator
    /// \param p_phiIn     for each regime  gives the solution calculated at the previous step ( next time step by Dynamic Programming resolution) : structure of the 2D array ( nb simulation ,nb stocks )
    /// \return   a pair  :
    ///              - for each regimes (column) gives the solution for each particle (row)
    ///              - for each control (column) gives the optimal control for each particle (rows)
    ///              .
    virtual Eigen::ArrayXd    stepOptimize(const   std::vector<std::shared_ptr< RegularSpaceIntGrid> >  &p_grid,
                                           const   int &p_iReg,
                                           const   Eigen::ArrayXi  &p_state,
                                           const   std::shared_ptr< BaseRegression>  &p_condExp,
                                           const   std::vector < std::shared_ptr< Eigen::ArrayXXd > > &p_phiIn) const = 0;


    /// \brief defines a step in simulation
    /// Notice that this implementation is not optimal but is convenient if the control is discrete.
    /// By avoiding interpolation in control we avoid non admissible control
    /// Control are recalculated during simulation.
    /// \param p_grid          grid at arrival step after command for each regime
    /// \param p_condExp       Conditional expectation operator reconstructing conditionnal expectation from basis functions for each state
    /// \param p_basisFunc     Basis functions par each point of the grid state for each regime
    /// \param p_state         defines the state value (modified)
    /// \param p_phiInOut      defines the value functions (modified) : size number of functions  to follow
    virtual void stepSimulate(const std::vector< std::shared_ptr< RegularSpaceIntGrid> >   &p_grid,
                              const std::shared_ptr< BaseRegression>  &p_condExp,
                              const std::vector< Eigen::ArrayXXd >   &p_basisFunc,
                              libflow::StateWithIntState &p_state,
                              Eigen::Ref<Eigen::ArrayXd> p_phiInOut) const = 0 ;


    /// \brief Get the number of regimes allowed for the asset to be reached  at the current time step
    ///    If \f$ t \f$ is the current time, and $\f$ dt \f$  the resolution step,  this is the number of regime allowed on \f$[ t- dt, t[\f$
    virtual   int getNbRegime() const = 0 ;

    /// \brief get the simulator back
    virtual std::shared_ptr< libflow::SimulatorDPBase > getSimulator() const = 0;

    /// \brief get size of the  function to follow in simulation
    virtual int getSimuFuncSize() const = 0;

};
}
#endif /* OPTIMIZERSWITCHBASE_H */
