
#ifndef SIMULATORDPBASE_H
#define SIMULATORDPBASE_H
#include <Eigen/Dense>

/* \file SimulatorDPBase.h
 * \brief Abstract class for simulators for Dynamic Programming Programms
 * \author Xavier Warin
 */

namespace libflow
{
/// \class SimulatorDPBase SimulatorDPBase.h
/// Abstract class for simulator used in dynamic programming
class SimulatorDPBase
{


public :

    /// \brief Constructor
    SimulatorDPBase() {}
    /// \brief Destructor
    virtual ~SimulatorDPBase() {}
    /// \brief get current  markovian state : dimension of the problem for the  first dimension , second dimension the number of Monte Carlo simulations
    virtual Eigen::MatrixXd getParticles() const = 0;
    /// \brief a step forward for simulations
    virtual void  stepForward() = 0;
    /// \brief a step backward for simulations
    virtual void  stepBackward() = 0;
    /// \brief a step forward for simulations
    /// \return  current particles (markovian state as assets for example) (dimension of the problem times simulation number)
    virtual Eigen::MatrixXd  stepForwardAndGetParticles() = 0;
    /// \brief a step backward for simulations
    /// \return  current particles (markovian state as assets for example) (dimension of the problem times simulation number)
    virtual Eigen::MatrixXd stepBackwardAndGetParticles() = 0;
    /// \brief get back dimension of the regression
    virtual int getDimension() const = 0;
    /// \brief get the number of steps
    virtual  int getNbStep() const = 0;
    /// \brief Get the current step size
    virtual double getStep() const = 0;
    /// \brief Get current time
    virtual double getCurrentStep() const = 0 ;
    /// \brief Number of Monte Carlo simulations
    virtual int getNbSimul() const = 0;
    /// \brief Permit to actualize for one time step (interest rate)
    virtual double getActuStep() const = 0;
    /// \brief Permits to actualize at the initial date (interest rate)
    virtual double getActu() const = 0 ;

};
}
#endif /* SIMULATORDPBASE_H */
