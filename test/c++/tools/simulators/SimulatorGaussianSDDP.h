
#ifndef  SIMULTORGAUSSIANSDDP_H
#define  SIMULTORGAUSSIANSDDP_H
#include <boost/random.hpp>
#include <Eigen/Dense>
#include "libflow/sddp/SimulatorSDDPBase.h"


/// \class SimulatorGaussianSDDP SimulatorGaussianSDDP.h
/// A simulator for SDDP, just giving some Gaussian drawing
class SimulatorGaussianSDDP : public libflow::SimulatorSDDPBase
{
    boost::mt19937 m_generator;  ///< Boost random generator
    boost::normal_distribution<double> m_normalDistrib;
    boost::variate_generator<boost::mt19937 &, boost::normal_distribution<double> > m_normalRand ; ///	public :
    int m_dim ; ///< number of uncertainties
    int m_nbSample ; ///< number of samples
    Eigen::ArrayXXd m_particles ; ///< particles to generate (m_dim by m_nbSample)
    bool m_bForward ; /// true if forward

public :

    /// \brief Default constructor
    SimulatorGaussianSDDP(): m_generator(), m_normalDistrib(), m_normalRand(m_generator, m_normalDistrib) {}

    /// \brief Constructor for backward
    /// \param p_dim   dimension number
    /// \param p_nbSample number of simulations to get
    SimulatorGaussianSDDP(const int   &p_dim, const int &p_nbSample): m_generator(), m_normalDistrib(), m_normalRand(m_generator, m_normalDistrib), m_dim(p_dim), m_nbSample(p_nbSample), m_particles(p_dim, p_nbSample), m_bForward(false)
    {
        // initialize generator to be sure to have the same sample in backward
        m_generator.seed(5489u);
        for (int is = 0; is < m_nbSample; ++is)
            for (int id = 0; id < m_dim; ++id)
                m_particles(id, is) = m_normalRand();
    }
    /// \brief Constructor for forward
    /// \param p_dim   dimension number
    /// \param p_nbSample number of simulations to get
    SimulatorGaussianSDDP(const int   &p_dim): m_generator(), m_normalDistrib(), m_normalRand(m_generator, m_normalDistrib), m_dim(p_dim), m_nbSample(1), m_bForward(true), m_particles(p_dim, 1)
    {
        for (int is = 0; is < m_nbSample; ++is)
            for (int id = 0; id < m_dim; ++id)
                m_particles(id, is) = m_normalRand();
    }

    /// \brief destructor
    virtual ~SimulatorGaussianSDDP() {}

    ///\brief special function associated to the simulator
    /// \param p_idim  uncertainty targeted
    /// \param p_isim  simulation number
    inline double getGaussian(const int &p_idim, const int &p_isim)
    {
        return m_particles(p_idim, p_isim);
    }
    /// \brief Get back the number of particles
    inline int getNbSimul() const
    {
        return m_nbSample;
    }
    /// \brief Get back the number of samples
    inline int getNbSample() const
    {
        return m_nbSample;
    }
    /// \brief Update the simulator for the date
    inline void updateDateIndex(const int  &p_idate)
    {
        // resample
        for (int is = 0; is < m_nbSample; ++is)
            for (int id = 0; id < m_dim; ++id)
                m_particles(id, is) = m_normalRand();
    }

    /// \brief get one simulation
    /// \param p_isim  simulation number
    /// \return the particle associated to p_isim
    /// \brief get  current Markov state
    inline Eigen::VectorXd getOneParticle(const int &p_isim) const
    {
        return Eigen::VectorXd();
    }
    /// \brief get  current Markov state
    inline Eigen::MatrixXd getParticles() const
    {
        return Eigen::MatrixXd() ;
    }
    /// \brief Reset the simulator (to use it again for another SDDP sweep)
    inline void resetTime()
    {
        // to have same generator in backward
        if (!m_bForward)
        {
            m_generator.seed(5489u);
            for (int is = 0; is < m_nbSample; ++is)
                for (int id = 0; id < m_dim; ++id)
                    m_particles(id, is) = m_normalRand();
        }
    }


    /// \brief in simulation  part of SDDP reset  time  and reinitialize uncertainties (use in forward)
    /// \param p_nbSimul  Number of simulations to update (useless here)
    inline void updateSimulationNumberAndResetTime(const int &p_nbSimul)
    {
        m_nbSample = p_nbSimul;
        m_particles.resize(m_dim,  m_nbSample);
        for (int is = 0; is < m_nbSample; ++is)
            for (int id = 0; id < m_dim; ++id)
                m_particles(id, is) = m_normalRand();
    }
};
#endif /*  SIMULTORGAUSSIANSDDP_H  */
