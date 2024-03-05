// Copyright (C) 2016 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#ifndef BLACKSCHOLESSIMULATOR_H
#define BLACKSCHOLESSIMULATOR_H
#include <boost/random.hpp>
#include <Eigen/Dense>
#include "libflow/core/utils/constant.h"
#include "libflow/core/utils/comparisonUtils.h"
#include "libflow/dp/SimulatorDPBase.h"

/* \file BlackScholesSimulator.h
* \brief Simulate Black Scholes for a set of dates and a given number of
*        simulations.  Brownians are simulated till the end of the period and
*        a Brownian bridge is used while stepping back if backward resolution.
*        A forward mode for simulation is possible too.
*        the model satisfy \f$ dS = S( \mu dt + \sigma dW_t) \f$
*        the correlation between the Brownians is given by \f$ \rho \f$
* \author Xavier Warin
*/

/// \class BlackScholesSimulator BlackScholesSimulator.h
/// Implement a Black Scholes simulator
class BlackScholesSimulator : public libflow::SimulatorDPBase
{
private :

    Eigen::VectorXd m_initialValues ; ///< initial asset values \f$ S_0 \f$
    Eigen::VectorXd m_sigma ; ///< Volatility \f$\sigma\f$
    Eigen::VectorXd m_mu    ; ///< Trend \f$\mu\f$
    Eigen::MatrixXd corrFactTrans ; ///< store lower part of factorized matrix by Choleski
    double m_T ; ///< maturity of the option
    double m_step ; ///< simulation step
    int  m_nbStep ; ///< number of time steps
    size_t m_nbSimul ; ///< Monte Carlo simulation number
    bool m_bForward ; ///< True if forward mode
    double m_currentStep ; ///< Current step during resolution
    Eigen::MatrixXd m_brownian; ///< store the Brownian motion values (here storage dimension \f$ \times \f$ m_bSimul) (no correlation)
    boost::mt19937 m_generator;  ///< Boost random generator
    boost::normal_distribution<double> m_normalDistrib;
    boost::variate_generator<boost::mt19937 &, boost::normal_distribution<double> > m_normalRand ; ///< Normal generator

    /// \brief gets Brownian motions correlated and send back asset values
    Eigen::MatrixXd brownianToAsset() const
    {
        // correlation
        Eigen::MatrixXd correlBrownian(m_initialValues.size(), m_nbSimul);
        for (size_t is = 0; is < m_nbSimul; ++is)
        {
            correlBrownian.col(is) = corrFactTrans * m_brownian.col(is);
        }
        Eigen::MatrixXd assetToReturn(m_initialValues.size(), m_nbSimul);
        for (int id = 0; id < m_initialValues.size(); ++id)
        {
            Eigen::ArrayXd  vecLoc((m_mu(id) - 0.5 * m_sigma(id)*m_sigma(id))*m_currentStep + m_sigma(id)*correlBrownian.array().row(id));
            assetToReturn.row(id) = m_initialValues(id) * vecLoc.exp();
        }
        return  assetToReturn;
    }

    /// a step forward for Brownians
    void  forwardStepForBrownian()
    {
        Eigen::MatrixXd increment(m_initialValues.size(), m_nbSimul);
        double sqrtStep = sqrt(m_step);
        for (size_t is = 0; is < m_nbSimul; ++is)
            for (int id = 0; id < m_initialValues.size(); ++id)
                increment(id, is) = sqrtStep * m_normalRand();
        // update correlated Brownians
        m_brownian += increment;
    }

    /// a step backward for Brownians
    void  backwardStepForBrownian()
    {
        if (libflow::almostEqual(m_currentStep, libflow::zero, libflow::ulp))
            m_brownian.setConstant(libflow::zero);
        else
        {
            double util1 = std::max(m_currentStep / (m_currentStep + m_step), 0.);
            double util2 = sqrt(util1 * m_step);
            // use Brownian bridge
            for (size_t is  = 0; is < m_nbSimul; ++is)
                for (int id = 0; id < m_initialValues.size(); ++id)
                    m_brownian(id, is) = m_brownian(id, is) * util1 + util2 * m_normalRand();

        }
    }


public:

    /// \brief Constructor for Black Scholes simulator, but using an external generator
    /// \param p_initialValues  initial values for assets
    /// \param p_sigma          volatility of assets
    /// \param p_mu             trend for assets
    /// \param p_correl         correlation for assets
    /// \param p_T              maturity
    /// \param p_nbStep         number of time step
    /// \param p_nbSimul        number of simulations
    /// \param p_bForward       true if the simulation is forward, false if backward
    /// \param p_seed           seed generator (optional, here use default boost mersenne twister)
    BlackScholesSimulator(const Eigen::VectorXd   &p_initialValues,
                          const Eigen::VectorXd   &p_sigma,
                          const Eigen::VectorXd   &p_mu,
                          const Eigen::MatrixXd &p_correl,
                          const double &p_T,
                          const size_t &p_nbStep,
                          const size_t &p_nbSimul,
                          const bool &p_bForward,
                          uint32_t  p_seed = 5489u)
        : m_initialValues(p_initialValues), m_sigma(p_sigma),
          m_mu(p_mu), corrFactTrans(p_correl.llt().matrixL()),
          m_T(p_T), m_step(p_T / p_nbStep), m_nbStep(p_nbStep), m_nbSimul(p_nbSimul), m_bForward(p_bForward),
          m_currentStep(p_bForward ? 0. : p_T), m_brownian(p_initialValues.size(), p_nbSimul),
          m_generator(), m_normalDistrib(), m_normalRand(m_generator, m_normalDistrib)
    {
        m_generator.seed(p_seed);
        if (m_bForward)
            m_brownian.setConstant(0.);
        else
        {
            double sqrtT = sqrt(m_T);
            for (size_t is  = 0 ; is < m_nbSimul; ++is)
                for (int id = 0; id < p_initialValues.size(); ++id)
                    m_brownian(id, is) = sqrtT * m_normalRand();
        }
    }

    /// \brief Constructor for Black Scholes simulator
    /// \param p_initialValues  initial values for assets
    /// \param p_sigma          volatility of assets
    /// \param p_mu             trend for assets
    /// \param p_correl         correlation for assets
    /// \param p_T              maturity
    /// \param p_nbStep         number of time step
    /// \param p_nbSimul        number of simulations
    /// \param p_bForward       true if the simulation is forward, false if backward
    /// \param p_generator      external random generator
    BlackScholesSimulator(const Eigen::VectorXd   &p_initialValues,
                          const Eigen::VectorXd   &p_sigma,
                          const Eigen::VectorXd   &p_mu,
                          const Eigen::MatrixXd &p_correl,
                          const double &p_T,
                          const size_t &p_nbStep,
                          const size_t &p_nbSimul,
                          const bool &p_bForward,
                          boost::mt19937 &p_generator)
        : m_initialValues(p_initialValues), m_sigma(p_sigma),
          m_mu(p_mu), corrFactTrans(p_correl.llt().matrixL()),
          m_T(p_T), m_step(p_T / p_nbStep), m_nbStep(p_nbStep), m_nbSimul(p_nbSimul), m_bForward(p_bForward),
          m_currentStep(p_bForward ? 0. : p_T), m_brownian(p_initialValues.size(), p_nbSimul),
          m_normalDistrib(), m_normalRand(p_generator, m_normalDistrib)
    {
        if (m_bForward)
            m_brownian.setConstant(0.);
        else
        {
            double sqrtT = sqrt(m_T);
            for (size_t is  = 0 ; is < m_nbSimul; ++is)
                for (int id = 0; id < p_initialValues.size(); ++id)
                    m_brownian(id, is) = sqrtT * m_normalRand();
        }
    }


    /// \brief get current asset values
    Eigen::MatrixXd getParticles() const
    {
        return brownianToAsset();
    }

    /// \brief a step forward for simulations
    void  stepForward()
    {
        assert(m_bForward);
        m_currentStep += m_step;
        forwardStepForBrownian();
    }

    /// \return  the asset values (asset,simulations)
    void  stepBackward()
    {
        assert(!m_bForward);
        m_currentStep -= m_step;
        backwardStepForBrownian();
    }

    /// \brief a step forward for simulations
    /// \return  the asset values (asset,simulations)
    Eigen::MatrixXd  stepForwardAndGetParticles()
    {
        assert(m_bForward);
        m_currentStep += m_step;
        forwardStepForBrownian();
        return brownianToAsset();
    }

    /// \brief a step backward for simulations
    /// \return  the asset values (asset,simulations)
    Eigen::MatrixXd stepBackwardAndGetParticles()
    {
        assert(!m_bForward);
        m_currentStep -= m_step;
        backwardStepForBrownian();
        return brownianToAsset();
    }

    /// \brief Get back tangent process
    Eigen::MatrixXd getTangent() const
    {
        Eigen::MatrixXd asset = brownianToAsset();
        for (size_t is = 0 ; is < m_nbSimul; ++is)
            asset.col(is).array() /=  m_initialValues.array();
        return asset ;
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
    inline const Eigen::VectorXd &getInitialValues() const
    {
        return m_initialValues;
    }
    inline const Eigen::VectorXd &getSigma() const
    {
        return m_sigma ;
    }
    inline const Eigen::VectorXd &getMu() const
    {
        return  m_mu    ;
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
        return m_sigma.size();
    }
    ///@}
    ///@{
    /// actualization with the interest rate
    inline double getActuStep() const
    {
        return exp(-m_step * m_mu(0)) ;
    }
    /// actualization for initial date
    inline double getActu() const
    {
        return exp(-m_currentStep * m_mu(0));
    }
    ///@}
};

#endif /* BLACKSCHOLESSIMULATOR_H */
