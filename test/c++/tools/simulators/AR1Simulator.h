
#ifndef AR1SIMULATOR_H
#define AR1SIMULATOR_H
#include <boost/random.hpp>
#include "reflow/core/utils/constant.h"
#include "reflow/core/utils/comparisonUtils.h"
#include "reflow/dp/SimulatorDPBase.h"

/* \file AR1Simulator.h
 * \brief a Mean reverting simulator
*        The Markovian state is given by
*        \f$ Y_t = \int_0^t e^{-a(t-s)} \sigma dW_s \f$
*        The process satisfies :
*        \f$ dD_t = a(m - D_t) dt + \sigma \sigma dW_s \f$
*        So
*        \f$ D_t = Y_t + (D_0 -m) e^{-a t} + m \f$
* \author Xavier Warin
*/


/// \class AR1Simulator AR1Simulator.h
/// Simple mean reverting process, AR1 version
class AR1Simulator: public reflow::SimulatorDPBase
{

protected :

    double m_D0 ; ///< initial value
    double m_m ; ///< average value of the process
    double m_mr ;  ///<  mean reverting
    double  m_sigma ; ///< Volatility \f$\sigma\f$
    double m_T ; ///< maturity for the simulation
    double m_step ; ///< simulation step
    int  m_nbStep ; ///< number of time steps
    size_t m_nbSimul ; ///< Monte Carlo simulation number
    bool m_bForward ; ///< True if forward mode
    double m_currentStep ; ///< Current time step during resolution
    Eigen::VectorXd m_OUProcess; ///< store the Ornstein Uhlenbeck process (no correlation)
    boost::mt19937 m_generator;  ///< Boost random generator
    boost::normal_distribution<double> m_normalDistrib; ///< Normal distribution
    boost::variate_generator<boost::mt19937 &, boost::normal_distribution<double> > m_normalRand ; ///< Normal generator


    /// \brief a step forward for OU process
    void  forwardStepForOU()
    {
        double stDev = m_sigma * sqrt((1 - exp(-2 * m_mr * m_step)) / (2 * m_mr));
        double expActu = exp(-m_mr * m_step);
        for (size_t is = 0; is < m_nbSimul; ++is)
        {
            double increment = stDev * m_normalRand();
            // update OU process
            m_OUProcess(is) = m_OUProcess(is) * expActu + increment;
        }
    }

    /// \brief a step backward for OU process
    void  backwardStepForOU()
    {
        if (reflow:: isLesserOrEqual(m_currentStep, reflow::zero))
        {
            m_OUProcess.setConstant(0.);
        }
        else
        {
            // use Brownian bridge
            double util = sinh(m_mr * m_currentStep) / sinh(m_mr * (m_currentStep + m_step));
            double variance = pow(m_sigma, 2.) / (2 * m_mr) * ((1 - exp(-2 * m_mr * m_currentStep)) * pow(1 - exp(-m_mr * m_step) * util, 2.) + (1 - exp(-2 * m_mr * m_step)) * pow(util, 2.));
            double stdDev = sqrt(variance);
            for (size_t is  = 0; is < m_nbSimul; ++is)
            {
                double temp =  m_OUProcess(is);
                m_OUProcess(is) = temp * util + stdDev * m_normalRand();
            }
        }
    }


public:

/// \brief Constructor
/// \param  p_D0     Initial value
/// \param  p_m      average value
/// \param  p_sigma  Volatility
/// \param  p_mr     Mean reverting per factor
/// \param  p_T      Maturity
/// \param  p_nbStep   Number of time step for simulation
/// \param  p_nbSimul  Number of simulations for the Monte Carlo
/// \param  p_bForward true if the simulator is forward, false if the simulation is backward
    AR1Simulator(const double p_D0,
                 const double p_m,
                 const double   &p_sigma,
                 const double   &p_mr,
                 const double &p_T,
                 const size_t &p_nbStep,
                 const size_t &p_nbSimul,
                 const bool &p_bForward):
        m_D0(p_D0), m_m(p_m), m_mr(p_mr), m_sigma(p_sigma),  m_T(p_T), m_step(p_T / p_nbStep), m_nbStep(p_nbStep), m_nbSimul(p_nbSimul), m_bForward(p_bForward),
        m_currentStep(p_bForward ? 0. : p_T), m_OUProcess(p_nbSimul),
        m_generator(), m_normalDistrib(), m_normalRand(m_generator, m_normalDistrib)
    {
        if (m_bForward)
            m_OUProcess.setConstant(0.);
        else
        {
            double stDev = m_sigma * sqrt((1 - exp(-2 * m_mr * m_T)) / (2 * m_mr));
            for (size_t is  = 0 ; is < m_nbSimul; ++is)
                m_OUProcess(is) = stDev * m_normalRand();
        }
    }

    /// \brief get  current Markov state
    inline Eigen::MatrixXd getParticles() const
    {
        Eigen::VectorXd ret = m_OUProcess + Eigen::VectorXd::Constant(m_OUProcess.size(), (m_D0 - m_m) * exp(-m_mr * m_currentStep) + m_m);
        Eigen::MatrixXd retMap(1, ret.size());
        for (int is  = 0; is  < ret.size(); ++is)
            retMap(0, is) = std::max(ret(is), 0.);
        return  retMap;
    }


    /// \brief a step forward for simulations
    void  stepForward()
    {
        assert(m_bForward);
        m_currentStep += m_step;
        forwardStepForOU();
    }

    /// \brief a step backward
    void  stepBackward()
    {
        assert(!m_bForward);
        m_currentStep -= m_step;
        backwardStepForOU();
    }

    /// \brief a step forward for simulations
    /// \return  the asset values (asset,simulations)
    Eigen::MatrixXd  stepForwardAndGetParticles()
    {
        assert(m_bForward);
        m_currentStep += m_step;
        forwardStepForOU();
        return getParticles();
    }

    /// \brief a step backward for simulations
    /// \return  the asset values (asset,simulations)
    Eigen::MatrixXd stepBackwardAndGetParticles()
    {
        assert(!m_bForward);
        m_currentStep -= m_step;
        backwardStepForOU();
        return getParticles();
    }


    ///@{
    /// Get back attribute
    inline double getCurrentStep() const
    {
        return m_currentStep;
    }
    inline double getT() const
    {
        return m_T;
    }
    inline double getStep() const
    {
        return m_step;
    }
    inline double getSigma() const
    {
        return m_sigma ;
    }
    inline double getMr() const
    {
        return  m_mr    ;
    }
    inline int getNbSimul() const
    {
        return m_nbSimul;
    }
    int getNbStep() const
    {
        return m_nbStep;
    }
    int getDimension() const
    {
        return 1;
    }
    ///@}
    ///@{
    /// actualization with the interest rate
    inline double getActuStep() const
    {
        return 1. ;
    }
    /// actualization for initial date
    inline double getActu() const
    {
        return 1.;
    }
    ///@}
};


#endif /* AR1SIMULATOR_H */
