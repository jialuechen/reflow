
#ifndef OPTIMIZERSDDPBASE_H
#define OPTIMIZERSDDPBASE_H
#include <Eigen/Dense>
#include "libflow/sddp/SDDPCutOptBase.h"
#include "libflow/core/grids/OneDimRegularSpaceGrid.h"
#include "libflow/core/grids/OneDimData.h"
#include "libflow/sddp/SimulatorSDDPBase.h"
#include "libflow/sddp/SimulatorSDDPBaseTree.h"


/** \file OptimizerSDDPBase.h
 *  \brief Define an abstract class for Stochastic Dual Dynamic Programming problems
 *     \author Xavier Warin
 */

namespace libflow
{

/// \class OptimizerSDDPBase OptimizerSDDPBase.h
///  Base class for optimizer for Dynamic Programming
class OptimizerSDDPBase
{


public :

    OptimizerSDDPBase() {}

    virtual ~OptimizerSDDPBase() {}


    /// \brief Optimize the LP during backward resolution
    /// \param p_linCut	cuts used for the PL (Benders for the Bellman value at the end of the time step)
    /// \param p_aState	store the state, and 0.0 values
    /// \param p_particle	the particle n dimensional value associated to the regression
    /// \param p_isample	sample number for independant uncertainties
    /// \return  a vector with the optimal value and the derivatives if the function value with respect to each state
    virtual Eigen::ArrayXd oneStepBackward(const libflow::SDDPCutOptBase &p_linCut, const std::tuple< std::shared_ptr<Eigen::ArrayXd>, int, int > &p_aState, const Eigen::ArrayXd &p_particle, const int &p_isample) const = 0;

    /// \brief Optimize the LP during forward resolution
    /// \param p_aParticle	a particule in simulation part to get back cuts
    /// \param p_linCut	cuts used for the PL  (Benders for the Bellman value at the end of the time step)
    /// \param p_state	store the state, the particle number used in optimization and mesh number associated to the particle. As an input it constains the current state
    /// \param p_stateToStore	for backward resolution we need to store \f$ (S_t,A_{t-1},D_{t-1}) \f$  where p_state in output is \f$ (S_t,A_{t},D_{t}) \f$
    /// \param p_isimu           number of teh simulation used
    virtual double oneStepForward(const Eigen::ArrayXd &p_aParticle, Eigen::ArrayXd &p_state,  Eigen::ArrayXd &p_stateToStore, const libflow::SDDPCutOptBase &p_linCut,
                                  const int &p_isimu) const = 0 ;


    /// \brief update the optimizer for new date
    ///         - In Backward mode, LP resolution achieved at date p_dateNext,
    ///            starting with uncertainties given at date p_date and  evolving to give uncertainty at date p_dateNext,
    ///         - In Forward mode,  LP resolution achieved  at date p_date,
    ///            and uncertainties evolve till date p_dateNext
    ///         .
    virtual void updateDates(const double &p_date, const double &p_dateNext) = 0 ;

    /// \brief Get an admissible state for a given date
    /// \param  p_date   current date
    /// \return an admissible state
    virtual Eigen::ArrayXd oneAdmissibleState(const double &p_date) = 0 ;

    /// \brief get back state size
    virtual int getStateSize() const = 0;

    /// \brief get the backward simulator back
    virtual std::shared_ptr< libflow::SimulatorSDDPBase > getSimulatorBackward() const = 0;

    /// \brief get the forward simulator back
    virtual std::shared_ptr< libflow::SimulatorSDDPBase > getSimulatorForward() const = 0;

};
}
#endif /* OPTIMIZERSDDPBASE_H */
