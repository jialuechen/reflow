
#ifndef MEANREVERTINGSIMULATOR_H
#define MEANREVERTINGSIMULATOR_H
#include <memory>
#include <boost/random.hpp>
#include "reflow/core/utils/constant.h"
#include "reflow/dp/SimulatorMultiStageDPBase.h"
#include "reflow/sddp/SimulatorSDDPBase.h"


/* \file MeanRevertingSimulator.h
 * \brief Simulate a future deformation with
 *        a given number of uncertainties
 *        A forward and a backward mode (Brownian bridge) are provided for \f$T\f$ given
 *        \f$ dF(t,T) = F(t,T) ( \sum_{i=1}^N e^{-a_i (T-t)} \sigma_i dW^i_t) \f$
 *        In this simple case, no correlations are provided
 *        The Markovian state is given by
 *        $ Y^i_t = \int_0^t e^{-a_i(t-s)} \sigma_i dW^i_s $
 *        A Brownian bridge is used  see  BARCZY-PETER KERN
 *        "SAMPLE PATH DEVIATIONS OF THE WIENER AND THE ORNSTEIN-UHLENBECK PROCESS FROM ITS BRIDGES"
 *        The one dimensional bridge  between \f$t\f$  and  \f$t+dt\f$ is given by :
 *        \f{eqnarray*}{
 *           V & =& \frac{\sigma^2}{2a}}(1-e^{-2a t})(1- e^{-a dt} \frac{\sinh(at)}{\sinh(a(t+dt))})^2 + \\
 *             &  &  \frac{\sigma^2}{2a}} (1-e^{-2a dt}) \frac{\sinh(at)^2}{\sinh(a(t+dt))^2} \\
 *           Y_t & = & Y_{t+dt} (1- e^{-a dt} \frac{\sinh(at)}{\sinh(a(t+dt))} + \sqrt{V} \mathcal{N}(0,1))
 *        \f}
 * \author Xavier Warin
 */


/// \class MeanRevertingSimulator MeanRevertingSimulator.h
/// Ornstein Uhlenbeck simulator
template< class Curve>
class MeanRevertingSimulator: public reflow::SimulatorMultiStageDPBase, public reflow::SimulatorSDDPBase
{
protected :
    Eigen::VectorXd m_mr ; ///<  mean reverting
    Eigen::VectorXd m_sigma ; ///< Volatility \f$\sigma\f$
    double m_trend ; ///< store sum of the \f$ sigma^2/(2a) (1-exp(-2a t)) \f$
    std::shared_ptr< Curve> m_curve; ///< Future curve at initial date (0)
    double m_r ; ///< Interest rate
    double m_T ; ///< maturity of the option
    double m_step ; ///< simulation step
    int  m_nbStep ; ///< number of time steps
    size_t m_nbSimul ; ///< Monte Carlo simulation number
    bool m_bForward ; ///< True if forward mode
    double m_currentStep ; ///< Current time step during resolution
    Eigen::MatrixXd m_OUProcess; ///< store the Ornstein Uhlenbeck process (no correlation)
    boost::mt19937 m_generator;  ///< Boost random generator
    boost::normal_distribution<double> m_normalDistrib; ///< Normal distribution
    boost::variate_generator<boost::mt19937 &, boost::normal_distribution<double> > m_normalRand ; ///< Normal generator
    int m_nbPeriodInTransition ; ///< permit to get the number of transition  of each each time step

    /// \brief Actualize trend
    void actualizeTrend()
    {
        m_trend = 0;
        for (int id = 0 ; id < m_sigma.size(); ++id)
        {
            m_trend += pow(m_sigma(id), 2.) / (2 * m_mr(id)) * (1 - exp(-2 * m_mr(id) * m_currentStep));
        }
        m_trend *= 0.5;
    }
    /// a step forward for OU process
    void  forwardStepForOU()
    {
        for (int id = 0; id < m_OUProcess.rows(); ++id)
        {
            double stDev = m_sigma(id) * sqrt((1 - exp(-2 * m_mr(id) * m_step)) / (2 * m_mr(id)));
            double expActu = exp(-m_mr(id) * m_step);
            for (size_t is = 0; is < m_nbSimul; ++is)
            {
                double increment = stDev * m_normalRand();
                // update OU process
                m_OUProcess(id, is) = m_OUProcess(id, is) * expActu + increment;
            }
        }
    }

    /// a step backward for OU process
    void  backwardStepForOU()
    {
        if (reflow:: isLesserOrEqual(m_currentStep, reflow::zero))
        {
            m_OUProcess.setConstant(reflow::zero);
        }
        else
        {
            for (int id = 0; id < m_OUProcess.rows(); ++id)
            {
                // use brownian bridge
                double util = sinh(m_mr(id) * m_currentStep) / sinh(m_mr(id) * (m_currentStep + m_step));
                double variance = pow(m_sigma(id), 2.) / (2 * m_mr(id)) * ((1 - exp(-2 * m_mr(id) * m_currentStep)) * pow(1 - exp(-m_mr(id) * m_step) * util, 2.) + (1 - exp(-2 * m_mr(id) * m_step)) * pow(util, 2.));
                double stdDev = sqrt(variance);
                for (size_t is  = 0; is < m_nbSimul; ++is)
                {
                    double temp =  m_OUProcess(id, is);
                    m_OUProcess(id, is) = temp * util + stdDev * m_normalRand();
                }
            }
        }
    }


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
    MeanRevertingSimulator(const std::shared_ptr<Curve> &p_curve,
                           const Eigen::VectorXd   &p_sigma,
                           const Eigen::VectorXd    &p_mr,
                           const double &p_r,
                           const double &p_T,
                           const size_t &p_nbStep,
                           const size_t &p_nbSimul,
                           const bool &p_bForward,
                           int p_nbPeriodInTransition = 1):
        m_mr(p_mr), m_sigma(p_sigma), m_curve(p_curve), m_r(p_r),
        m_T(p_T), m_step(p_T / p_nbStep), m_nbStep(p_nbStep), m_nbSimul(p_nbSimul), m_bForward(p_bForward),
        m_currentStep(p_bForward ? 0. : p_T), m_OUProcess(p_sigma.size(), p_nbSimul),
        m_generator(), m_normalDistrib(), m_normalRand(m_generator, m_normalDistrib), m_nbPeriodInTransition(p_nbPeriodInTransition)
    {
        if (m_bForward)
            m_OUProcess.setConstant(0.);
        else
        {
            for (int id = 0; id < m_OUProcess.rows(); ++id)
            {
                double stDev = m_sigma(id) * sqrt((1 - exp(-2 * m_mr(id) * m_T)) / (2 * m_mr(id)));
                for (size_t is  = 0 ; is < m_nbSimul; ++is)
                    m_OUProcess(id, is) = stDev * m_normalRand();
            }
        }
        actualizeTrend();
    }


    /// \brief get  current Markov state
    Eigen::MatrixXd getParticles() const
    {
        return m_OUProcess;
    }

    /// \brief get one simulation
    /// \param p_isim  simulation number
    /// \return the particle associated to p_isim
    /// \brief get  current Markov state
    inline Eigen::VectorXd getOneParticle(const int &p_isim) const
    {
        return m_OUProcess.col(p_isim);
    }


    /// \brief a step forward for simulations
    void  stepForward()
    {
        assert(m_bForward);
        m_currentStep += m_step;
        actualizeTrend();
        forwardStepForOU();
    }

    /// \return  the asset values (asset,simulations)
    void  stepBackward()
    {
        assert(!m_bForward);
        m_currentStep -= m_step;
        actualizeTrend();
        backwardStepForOU();
    }

    /// \brief a step forward for simulations
    /// \return  the asset values (asset,simulations)
    Eigen::MatrixXd  stepForwardAndGetParticles()
    {
        assert(m_bForward);
        m_currentStep += m_step;
        actualizeTrend();
        forwardStepForOU();
        return m_OUProcess;
    }

    /// \brief a step backward for simulations
    /// \return  the asset values (asset,simulations)
    Eigen::MatrixXd stepBackwardAndGetParticles()
    {
        assert(!m_bForward);
        m_currentStep -= m_step;
        actualizeTrend();
        backwardStepForOU();
        return m_OUProcess;
    }


    /// \brief From particles simulation for an  OU process, get spot price
    /// \param p_particles  (dimension of the problem by number of simulations)
    /// \return spot price for all simulations
    Eigen::VectorXd fromParticlesToSpot(const Eigen::MatrixXd &p_particles) const
    {
        Eigen::VectorXd values(p_particles.cols());
        double curveCurrent = m_curve->get(m_currentStep);
        for (size_t is = 0; is < m_nbSimul; ++is)
        {
            values(is) = curveCurrent * exp(-m_trend +  p_particles.col(is).sum());
        }
        return values;
    }

    /// \brief From one particle simulation for an  OU process, get spot price
    /// \param  p_oneParticle  One particle
    /// \return spot value
    inline double fromOneParticleToSpot(const Eigen::VectorXd   &p_oneParticle)  const
    {
        double curveCurrent = m_curve->get(m_currentStep);
        return curveCurrent * exp(-m_trend +  p_oneParticle.sum());
    }

    /// \brief get back asset spot value
    Eigen::VectorXd getAssetValues() const
    {
        return fromParticlesToSpot(m_OUProcess);
    }

    /// \brief get back asset spot value
    /// \param p_isim  simulation particle number
    /// \return spot value for this  particle
    double  getAssetValues(const int &p_isim) const
    {
        return m_curve->get(m_currentStep) * exp(- m_trend + m_OUProcess.col(p_isim).sum());
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
    inline const Eigen::VectorXd &getSigma() const
    {
        return m_sigma ;
    }
    inline const Eigen::VectorXd &getMr() const
    {
        return  m_mr    ;
    }
    inline int getNbSimul() const
    {
        return m_nbSimul;
    }
    inline int  getNbSample() const
    {
        return 1;
    }
    int getNbStep() const
    {
        return m_nbStep;
    }
    int getDimension() const
    {
        return m_sigma.size();
    }
    ///@}

    /// \brief forward or backward update
    /// \param p_date  current date in simulator
    void updateDates(const double &p_date)
    {
        if (m_bForward)
        {
            if (p_date > 0.)
                stepForward();
        }
        else
            stepBackward();
    }
    /// \brief forward or backward update
    /// \param p_idate  current date index in simulator
    void updateDateIndex(const int &p_idate)
    {
        if (m_bForward)
        {
            if (p_idate > 0)
                stepForward();
        }
        else
            stepBackward();
    }


    /// \brief forward or backward update for time
    inline void resetTime()
    {
        if (m_bForward)
        {
            m_currentStep = 0. ;
            m_OUProcess.setConstant(0.);
        }
        else
        {
            // to have same trajectories in backward
            m_currentStep = m_T;
            for (int id = 0; id < m_OUProcess.rows(); ++id)
            {
                double stDev = m_sigma(id) * sqrt((1 - exp(-2 * m_mr(id) * m_T)) / (2 * m_mr(id)));
                for (size_t is  = 0 ; is < m_nbSimul; ++is)
                    m_OUProcess(id, is) = stDev * m_normalRand();
            }
        }
        actualizeTrend();
    }

    /// \brief  update the number of simulations (forward only)
    /// \param p_nbSimul  Number of simulations to update
    inline void updateSimulationNumberAndResetTime(const int &p_nbSimul)
    {
        assert(m_bForward);
        m_nbSimul = p_nbSimul;
        m_OUProcess.resize(m_sigma.size(), p_nbSimul);
        m_currentStep = 0. ;
        m_OUProcess.setConstant(0.);
        actualizeTrend();
    }
    ///@{
    /// actualization with the interest rate
    inline double getActuStep() const
    {
        return exp(-m_step * m_r) ;
    }
    /// actualization for initial date
    inline double getActu() const
    {
        return exp(-m_currentStep * m_r);
    }

    int getNbPeriodsInTransition() const
    {
        return m_nbPeriodInTransition;
    }
    ///@}
};
#endif /* MEANREVERTINGSIMULATOR_H */
