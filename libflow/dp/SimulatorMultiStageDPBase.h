

///
#ifndef SIMULATORMULTISTAGEDPBASE_H
#define SIMULATORMULTISTAGEDPBASE_H
#include <Eigen/Dense>
#include "libflow/dp/SimulatorDPBase.h"

/* \file   SimulatorMultiStageDPBase.h
 * \brief  Abstract class for simulators for Dynamic Programming Programms when a deterministic optimization
 *         by DP is achieved on a transition step in time
 * \author Benoit Clair, Xavier Warin
 */

namespace libflow
{
/// \class SimulatorMultiStageDPBase SimulatorMultiStageDPBase.h
/// Abstract class for simulator used in dynamic programming when a deterministic optimization  by DP is achieved on a transition step in time
class SimulatorMultiStageDPBase : public SimulatorDPBase
{

private :

    int m_iperiodCur ; ///< Number of current period during optimization or simulation in deterministic for the current time step

public :

    /// \brief Constructor
    SimulatorMultiStageDPBase(): SimulatorDPBase() {}
    /// \brief Destructor
    virtual ~SimulatorMultiStageDPBase() {}
    /// \brief Returns number of periods of current timestep
    virtual int getNbPeriodsInTransition() const = 0;
    ///< Set period treated
    void setPeriodInTransition(const int &p_iperiodCur)
    {
        m_iperiodCur = p_iperiodCur;
    }
    ///< Get period treated
    int  getPeriodInTransition() const
    {
        return m_iperiodCur ;
    }
};
}
#endif /* SIMULATORMULTISTAGEDPBASE_H */
