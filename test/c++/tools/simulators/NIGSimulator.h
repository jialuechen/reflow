
#ifndef NIGSIMULATOR_H
#define NIGSIMULATOR_H
#include <memory>
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <boost/random.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include "geners/BinaryFileArchive.hh"
#include "geners/Record.hh"
#include "geners/Reference.hh"
#include "libflow/core/utils/eigenGeners.h"
#include "libflow/core/utils/constant.h"
#include "libflow/dp/SimulatorDPBase.h"
#include "libflow/sddp/SimulatorSDDPBase.h"


/* \file NIGSimulator.h
 * \brief Simulate assets not linked with NIG process  (Normal Inverse Gaussian)
 * \f$  S_t = S_0 e^{\mu t + X_t}\f$
 * where \f$ X_t \f$ is an NIG process with parameters \f$\alpha, \beta , \delta\f$
 * \author Xavier Warin
 */


/// \class NIGSimulator NIGSimulator.h
///  NIG simulator
class NIGSimulator: public libflow::SimulatorDPBase
{

protected :
    Eigen::VectorXd m_initialValues ; ///< initial asset values \f$ S_0 \f$
    Eigen::VectorXd m_alpha ; ///< NIG alpha
    Eigen::VectorXd m_beta ; ///< NIG Beta
    Eigen::VectorXd m_delta; // delta NIG
    Eigen::VectorXd m_mu ; ///<  trend
    double m_T ; ///< maturity of the option
    double m_step ; ///< simulation step
    int  m_nbStep ; ///< number of time steps
    size_t m_nbSimul ; ///< Monte Carlo simulation number
    bool m_bForward ; ///< True if forward mode
    double m_currentStep ; ///< Current time step during resolution
    std::shared_ptr<gs::BinaryFileArchive> m_arch ;
    Eigen::MatrixXd m_NIGProcess; ///< store the NIG process
    boost::mt19937 m_generator;  ///< Boost random generator
    boost::normal_distribution<double> m_normalDistrib; ///< Normal distribution
    boost::random::uniform_real_distribution<> m_uniformDistrib; ///< Normal distribution
    boost::variate_generator<boost::mt19937 &, boost::normal_distribution<double> > m_normalRand ; ///< Normal generator
    boost::variate_generator<boost::mt19937 &,  boost::random::uniform_real_distribution< > > m_uniformRand ; ///< Normal generator

public:

    /// \brief Constructor
    /// \param p_initialValues  initial values for assets
    /// \param p_alpha          NIG alpha of assets
    /// \param p_beta           NIG beta of assets
    /// \param p_delta          NIG delta
    /// \param p_mu             trend for assets
    /// \param p_T              maturity
    /// \param p_nbStep         number of time step
    /// \param p_nbSimul        number of simulations
    /// \param p_bForward       true if the simulation is forward, false if backward
    NIGSimulator(const Eigen::VectorXd   &p_initialValues,
                 const Eigen::VectorXd   &p_alpha,
                 const Eigen::VectorXd   &p_beta,
                 const Eigen::VectorXd   &p_delta,
                 const Eigen::VectorXd   &p_mu,
                 const double &p_T,
                 const size_t &p_nbStep,
                 const size_t &p_nbSimul,
                 const bool &p_bForward,
                 std::string   &p_nameArch,
#ifdef USE_MPI
                 const boost::mpi::communicator &p_world,
#endif
                 int p_seed = 0)
        : m_initialValues(p_initialValues), m_alpha(p_alpha), m_beta(p_beta), m_delta(p_delta),
          m_mu(p_mu), m_T(p_T), m_step(p_T / p_nbStep), m_nbStep(p_nbStep), m_nbSimul(p_nbSimul), m_bForward(p_bForward),
          m_currentStep(p_bForward ? 0. : p_T), m_NIGProcess(p_initialValues.size(), p_nbSimul),
          m_generator(), m_normalDistrib(),  m_uniformDistrib(0., 1.), m_normalRand(m_generator, m_normalDistrib), m_uniformRand(m_generator, m_uniformDistrib)
    {
        // affect seed
        m_generator.seed(p_seed);
        if (p_world.rank() == 0)
            m_arch = std::make_shared<gs::BinaryFileArchive>(p_nameArch.c_str(), "w");
        m_NIGProcess = Eigen::MatrixXd::Constant(p_mu.size(), m_nbSimul, 0.);
        // initial dump
        if (p_world.rank() == 0)
            *m_arch << gs::Record(m_NIGProcess, "allStep", "NIG");
        Eigen::VectorXd gamma(p_mu.size());
        for (int iAsset = 0; iAsset < p_mu.size() ; ++iAsset)
            gamma(iAsset) = sqrt(m_alpha(iAsset) * m_alpha(iAsset) - m_beta(iAsset) * m_beta(iAsset));
        for (size_t i = 0; i < static_cast<size_t>(m_nbStep); ++i)
        {
            for (size_t is = 0; is < m_nbSimul; ++is)
            {
                // trend
                m_NIGProcess.col(is) += m_mu * m_step;
                for (int iAsset = 0; iAsset < p_mu.size(); ++iAsset)
                {
                    double deltaLoc = m_delta(iAsset) * m_step;
                    double V = pow(m_normalRand(), 2.);
                    double VGam = V / (2 * pow(gamma(iAsset), 2.));
                    double Y1 = deltaLoc / gamma(iAsset) + VGam - sqrt(V * deltaLoc / pow(gamma(iAsset), 3.) + pow(VGam, 2.));
                    double Y2 = pow(deltaLoc / gamma(iAsset), 2.) / Y1;
                    double U = m_uniformRand();
                    double Y;
                    if (U < deltaLoc / (deltaLoc + gamma(iAsset)*Y1))
                        Y = Y1;
                    else
                        Y = Y2;
                    // mixt laws
                    m_NIGProcess(iAsset, is) += m_beta(iAsset) * Y + m_normalRand() * sqrt(Y);
                }
            }
            if (p_world.rank() == 0)
                *m_arch << gs::Record(m_NIGProcess, "allStep", "NIG");
        }
        if (p_world.rank() == 0)
            m_arch->flush();
        p_world.barrier();
        // all processor read
        m_arch = std::make_shared<gs::BinaryFileArchive>(p_nameArch.c_str(), "r");
        if (p_bForward)
            m_currentStep = 0;
        else
            m_currentStep = m_nbStep;
        gs::Reference< Eigen::MatrixXd > (*m_arch, "allStep", "NIG").restore(m_currentStep, &m_NIGProcess);

    }

    /// \brief get  current Markov state
    Eigen::MatrixXd getParticles() const
    {
        Eigen::ArrayXXd particles = m_NIGProcess.array().exp();
        for (int iAsset = 0; iAsset < particles.rows(); ++iAsset)
        {
            particles.row(iAsset) *= m_initialValues(iAsset);
        }
        return  particles;
    }

    /// \brief a step forward for simulations
    void  stepForward()
    {
        m_currentStep += 1;
        gs::Reference< Eigen::MatrixXd > (*m_arch, "allStep", "NIG").restore(m_currentStep, &m_NIGProcess);
    }

    /// \return  the asset values (asset,simulations)
    void  stepBackward()
    {
        m_currentStep -= 1;
        gs::Reference< Eigen::MatrixXd > (*m_arch, "allStep", "NIG").restore(m_currentStep, &m_NIGProcess);
    }

    /// \brief a step forward for simulations
    /// \return  the asset values (asset,simulations)
    Eigen::MatrixXd  stepForwardAndGetParticles()
    {
        m_currentStep += 1;
        gs::Reference< Eigen::MatrixXd > (*m_arch, "allStep", "NIG").restore(m_currentStep, &m_NIGProcess);
        return getParticles();
    }

    /// \brief a step backward for simulations
    /// \return  the asset values (asset,simulations)
    Eigen::MatrixXd stepBackwardAndGetParticles()
    {
        m_currentStep -= 1;
        gs::Reference< Eigen::MatrixXd > (*m_arch, "allStep", "NIG").restore(m_currentStep, &m_NIGProcess);
        return  getParticles();
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
    inline int getNbSimul() const
    {
        return m_nbSimul;
    }
    inline int  getNbSample() const
    {
        return 1;
    }
    inline int getNbStep() const
    {
        return m_nbStep;
    }
    int getDimension() const
    {
        return m_mu.size();
    }
    ///@}


    ///@{
    /// actualization with the interest rate
    inline double getActuStep() const
    {
        return 1 ;
    }
    /// actualization for initial date
    inline double getActu() const
    {
        return 1.;
    }
    ///@}
};
#endif /* NIGSIMULATOR_H */
