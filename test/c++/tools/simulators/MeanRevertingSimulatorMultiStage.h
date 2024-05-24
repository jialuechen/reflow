

#ifndef MEANREVERTINGSIMULATORMULTISTAGE_H
#define MEANREVERTINGSIMULATORMULTISTAGE_H
#include <memory>
#include <boost/random.hpp>
#include "reflow/core/utils/constant.h"
#include "reflow/dp/SimulatorDPBase.h"
#include "reflow/dp/SimulatorMultiStageDPBase.h"
#include  "test/c++/tools/simulators/MeanRevertingSimulator.h"


/* \file MeanRevertingSimulatorMultiStage.h
 * \brief Adapt mean reverting to multistage
* \author Xavier Warin
 */

/// \class MeanRevertingSimulatorMultiStage MeanRevertingSimulatorMultiStage.h
/// Ornstein Uhlenbeck simulator adpated to multistage
template< class Curve>
class MeanRevertingSimulatorMultiStage: public MeanRevertingSimulator<Curve>, public SimulatorMultiStageDPBase
{

private :

    int m_nbPeriodInTransition ; ///< permit to get the number of transition  of each each time step

public:


    /// \brief Constructor
    /// \param  p_curve  Initial forward curve
    /// \param  p_sigma  Volatility of each factor
    /// \param  p_mr     Mean reverting per factor
    /// \param  p_r      Interest rate
    /// \param  p_T      Maturity
    /// \param  p_nbStep Number of time step for simulation
    /// \param p_nbSimul Number of simulations for the Monte Carlo
    /// \param p_bForward true if the simulator is forward, false if the simulation is backward
    /// \param p_nbPeriodInTransition  numebr of period in transition
    MeanRevertingSimulatorMultiStage(const std::shared_ptr<Curve> &p_curve,
                                     const Eigen::VectorXd   &p_sigma,
                                     const Eigen::VectorXd    &p_mr,
                                     const double &p_r,
                                     const double &p_T,
                                     const size_t &p_nbStep,
                                     const size_t &p_nbSimul,
                                     const bool &p_bForward,
                                     const int &p_nbPeriodInTransition):
        MeanRevertingSimulator<Curve>(p_curve, p_sigma, p_mr, p_r, p_T, p_nbStep, p_nbSimul, p_bForward), m_nbPeriodInTransition(p_nbPeriodInTransition) {    }


    int getNbPeriodsInTransition() const
    {
        return m_nbPeriodInTransition;
    }
}
#endif /* MEANREVERTINGSIMULATORMULTISTAGE_H */
