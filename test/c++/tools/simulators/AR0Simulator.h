// Copyright (C) 2016 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#ifndef AR0SIMULATOR_H
#define AR0SIMULATOR_H
#include <boost/random.hpp>
#include "libflow/core/utils/constant.h"
#include "libflow/core/utils/comparisonUtils.h"
#include "libflow/dp/SimulatorDPBase.h"

/* \file AR0Simulator.h
 * \brief  The simulations are just gaussian iid
* \author Xavier Warin
*/


/// \class AR0Simulator AR0Simulator.h
/// Simple mean reverting process, AR0 version
class AR0Simulator: public libflow::SimulatorDPBase
{

protected :

    double  m_sigma ; ///< Volatility \f$\sigma\f$
    double m_T ; ///< maturity for the simulation
    double m_step ; ///< simulation step
    int  m_nbStep ; ///< number of time steps
    size_t m_nbSimul ; ///< Monte Carlo simulation number
    bool m_bForward ; ///< True if forward mode
    double m_currentStep ; ///< Current time step during resolution
    Eigen::VectorXd m_state; ///<state to store
    boost::mt19937 m_generator;  ///< Boost random generator
    boost::normal_distribution<double> m_normalDistrib; ///< Normal distribution
    boost::variate_generator<boost::mt19937 &, boost::normal_distribution<double> > m_normalRand ; ///< Normal generator



public:

    /// \brief Constructor
    /// \param  p_sigma  Volatility
    /// \param  p_T      Maturity
    /// \param  p_nbStep   Number of time step for simulation
    /// \param  p_nbSimul  Number of simulations for the Monte Carlo
    /// \param  p_bForward true if the simulator is forward, false if the simulation is backward
    AR0Simulator(const double   &p_sigma,
                 const double &p_T,
                 const size_t &p_nbStep,
                 const size_t &p_nbSimul,
                 const bool &p_bForward):
        m_sigma(p_sigma),  m_T(p_T), m_step(p_T / p_nbStep), m_nbStep(p_nbStep), m_nbSimul(p_nbSimul), m_bForward(p_bForward),
        m_currentStep(p_bForward ? 0. : p_T), m_state(p_nbSimul),
        m_generator(), m_normalDistrib(), m_normalRand(m_generator, m_normalDistrib)
    {
        if (m_bForward)
            m_state.setConstant(0.);
        else
        {
            for (size_t is = 0; is < m_nbSimul; ++is)
                m_state(is) = m_sigma * m_normalRand();
        }
    }


    /// \brief get  current Markov state
    inline Eigen::MatrixXd getParticles() const
    {
        Eigen::MatrixXd retMap(1, m_nbSimul);
        retMap = m_state.transpose();
        return  retMap;
    }


    /// \brief a step forward for simulations
    void  stepForward()
    {
        assert(m_bForward);
        m_currentStep += m_step;
        for (size_t is = 0; is < m_nbSimul; ++is)
            m_state(is) = m_sigma * m_normalRand();
    }

    /// \brief a step backward
    void  stepBackward()
    {
        assert(!m_bForward);
        m_currentStep -= m_step;
        for (size_t is = 0; is < m_nbSimul; ++is)
            m_state(is) = m_sigma * m_normalRand();
    }

    /// \brief a step forward for simulations
    /// \return  the asset values (asset,simulations)
    Eigen::MatrixXd  stepForwardAndGetParticles()
    {
        assert(m_bForward);
        m_currentStep += m_step;
        for (size_t is = 0; is < m_nbSimul; ++is)
            m_state(is) = m_sigma * m_normalRand();
        return getParticles();
    }

    /// \brief a step backward for simulations
    /// \return  the asset values (asset,simulations)
    Eigen::MatrixXd stepBackwardAndGetParticles()
    {
        assert(!m_bForward);
        m_currentStep -= m_step;
        for (size_t is = 0; is < m_nbSimul; ++is)
            m_state(is) = m_sigma * m_normalRand();
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


#endif /* AR0SIMULATOR_H */
