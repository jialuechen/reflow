// Copyright (C) 2016 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#ifndef MEANREVERTINGSIMULATORSDDP_H
#define MEANREVERTINGSIMULATORSDDP_H
#include <memory>
#include <boost/random.hpp>
#include "libflow/core/utils/constant.h"
#include "libflow/dp/SimulatorDPBase.h"
#include "libflow/sddp/SimulatorSDDPBase.h"
#include  "test/c++/tools/simulators/MeanRevertingSimulator.h"


/* \file MeanRevertingSimulatorWithParticles.h
 * \brief Simulate a future deformation  as in MeanRevertingSimulator.h"
 * it  permits to generate some particles for sampling SDDP
 * \author Xavier Warin
 */


/// \class MeanRevertingSimulatorSDDP MeanRevertingSimulatorSDDP.h
/// Ornstein Uhlenbeck simulator
template< class Curve>
class MeanRevertingSimulatorSDDP: public MeanRevertingSimulator<Curve>
{
private :
    int m_dim ; ///< number of uncertainties (inflows, demand  for special SDDP treatment)
    int m_nbSample ; ///< number of samples
    Eigen::ArrayXXd m_particles ; ///< particles to generate (m_dim by m_nbSample) for sampling
    Eigen::ArrayXd m_dates ; ///< simulation dates

public:

/// \brief Constructor used in backward mode
/// \param  p_curve  Initial forward curve
/// \param  p_sigma  Volatility of each factor
/// \param  p_mr     Mean reverting per factor
/// \param  p_dates  Simulation dates
/// \param p_nbSimul Number of simulations for the Monte Carlo
/// \param p_dim     dimension number (inflows, demand for SDDP)
/// \param p_nbSample number of simulations to sample for SDDP part
    MeanRevertingSimulatorSDDP(std::shared_ptr<Curve> &p_curve,
                               const Eigen::VectorXd   &p_sigma,
                               const Eigen::VectorXd    &p_mr,
                               const Eigen::ArrayXd &p_dates,
                               const size_t &p_nbSimul,
                               const int   &p_dim,
                               const int &p_nbSample):
        MeanRevertingSimulator<Curve>(p_curve, p_sigma, p_mr, 0., p_dates(p_dates.size() - 1), p_dates.size() - 1, p_nbSimul, false),
        m_dim(p_dim), m_nbSample(p_nbSample), m_particles(p_dim, p_nbSample * p_nbSimul), m_dates(p_dates)
    {
        // to sample the same in optimization
        MeanRevertingSimulator<Curve>::m_generator.seed(5489u);
        // initialize seed
        for (int is = 0; is < m_nbSample * p_nbSimul; ++is)
            for (int id = 0; id < m_dim; ++id)
                m_particles(id, is) = MeanRevertingSimulator<Curve>::m_normalRand();

    }

/// \brief Constructor used in forward only
/// \param  p_curve  Initial forward curve
/// \param  p_sigma  Volatility of each factor
/// \param  p_mr     Mean reverting per factor
/// \param  p_dates  Simulation dates
/// \param  p_nbStep Number of time step for simulation
/// \param p_nbSimul Number of simulations for the Monte Carlo
/// \param p_dim     dimension number (inflows, demand for SDDP)
    MeanRevertingSimulatorSDDP(std::shared_ptr<Curve> &p_curve,
                               const Eigen::VectorXd   &p_sigma,
                               const Eigen::VectorXd    &p_mr,
                               const Eigen::ArrayXd &p_dates,
                               const size_t &p_nbSimul,
                               const int   &p_dim):
        MeanRevertingSimulator<Curve>(p_curve, p_sigma, p_mr, 0., p_dates(p_dates.size() - 1), p_dates.size() - 1, p_nbSimul, true),
        m_dim(p_dim), m_nbSample(1), m_particles(p_dim, p_nbSimul), m_dates(p_dates)
    {
        for (int is = 0; is < p_nbSimul; ++is)
            for (int id = 0; id < m_dim; ++id)
                m_particles(id, is) = MeanRevertingSimulator<Curve>::m_normalRand();
    }

    ///\brief special function associated to the simulator in Backward (always same simulations)
    /// \param p_idim  uncertainty targeted
    /// \param p_isim  simulation number
    inline double getGaussian(const int &p_idim, const int &p_isim)
    {
        return m_particles(p_idim, p_isim);
    }


    /// \brief Get back the number of samples
    inline int getNbSample() const
    {
        return m_nbSample;
    }

    /// \brief forward or backward update
    /// \param p_idate  date index  in simulator
    void updateDateIndex(const int   &p_idate)
    {
        MeanRevertingSimulator<Curve>::updateDates(m_dates(p_idate));
        for (int is = 0; is < MeanRevertingSimulator<Curve>::m_nbSimul * m_nbSample; ++is)
            for (int id = 0; id < m_dim; ++id)
                m_particles(id, is) = MeanRevertingSimulator<Curve>::m_normalRand();

    }

    /// \brief forward or backward update for time
    inline void resetTime()
    {
        MeanRevertingSimulator<Curve>::resetTime();
        // if backward reset seed to default to restart same generation
        if (!MeanRevertingSimulator<Curve>::m_bForward)
            MeanRevertingSimulator<Curve>::m_generator.seed(5489u);
        // resample to start
        for (int is = 0; is < m_nbSample * MeanRevertingSimulator<Curve>::m_nbSimul; ++is)
            for (int id = 0; id < m_dim; ++id)
                m_particles(id, is) = MeanRevertingSimulator<Curve>::m_normalRand();
    }

    /// \brief  update the number of simulations (forward only)
    /// \param p_nbSimul  Number of simulations to update
    /// \param p_nbSample Number of sample to update, useless here
    inline void updateSimulationNumberAndResetTime(const int &p_nbSimul)
    {
        m_nbSample = 1;
        MeanRevertingSimulator<Curve>::updateSimulationNumberAndResetTime(p_nbSimul);
        m_particles.resize(m_dim, p_nbSimul);
        for (int is = 0; is < p_nbSimul; ++is)
            for (int id = 0; id < m_dim; ++id)
                m_particles(id, is) = MeanRevertingSimulator<Curve>::m_normalRand();
    }

};
#endif /* MEANREVERTINGSIMULATORSDDP_H */
