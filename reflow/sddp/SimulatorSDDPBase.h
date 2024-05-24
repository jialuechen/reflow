
#ifndef SIMULATORSDDPBASE_H
#define SIMULATORSDDPBASE_H
#include <Eigen/Dense>

/* \file SimulatorBase.h
 * \brief Abstract class for simulators for SDDP method
* \author Xavier Warin
*/
namespace reflow
{
/// \class SimulatorSDDPBase SimulatorSDDPBase.h
/// Abstract class for simulators used for SDDP
class SimulatorSDDPBase
{
public :

    /// \brief Constructor
    SimulatorSDDPBase() {}

    /// \brief Destructor
    virtual ~SimulatorSDDPBase() {}

    /// \brief Get back the number of particles (used in regression part)
    virtual  int getNbSimul() const = 0;
    /// \brief Get back the number of sample used (simulation at each time step , these simulations are independent of the state)
    virtual  int getNbSample() const = 0;
    /// \brief Update the simulator for the date :
    /// \param p_idateCurr   index in date array
    virtual void updateDateIndex(const int &p_idateCur)  = 0;
    /// \brief get one simulation
    /// \param p_isim  simulation number
    /// \return the particle associated to p_isim
    /// \brief get  current Markov state
    virtual  Eigen::VectorXd getOneParticle(const int &p_isim) const = 0;
    /// \brief get  current Markov state
    virtual  Eigen::MatrixXd getParticles() const = 0;
    /// \brief Reset the simulator (to use it again for another SDDP sweep)
    virtual  void resetTime() = 0;
    /// \brief in simulation  part of SDDP reset  time  and reinitialize uncertainties
    /// \param p_nbSimul  Number of simulations to update
    virtual  void updateSimulationNumberAndResetTime(const int &p_nbSimul) = 0;
};
}
#endif /* SIMULATORSDDPBASE_H */
