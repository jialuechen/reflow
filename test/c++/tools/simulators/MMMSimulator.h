
#ifndef MMMSIMULATOR_H
#define MMMSIMULATOR_H
#include <memory>
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <boost/random.hpp>
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "geners/Record.hh"
#include "geners/Reference.hh"
#include "libflow/core/utils/eigenGeners.h"
#include "libflow/dp/SimulatorDPBase.h"

/* \file MMMSimulator.h
 * \brief Simulate Platen Market Minimal Model
 *        \f$ dS_t = \alpha dt + \sqrt{S_t \alpha_t} dW_t \f$
 *        This is the actualized price
 *        Simulation is exact (see Platen or Warin report)
 *        Simulations are stored using Geners Library
 * \author Xavier Warin
 */

/// \class MMMSimulator MMMSimulator.h
///        Implement MMN simulator
class MMMSimulator : public libflow::SimulatorDPBase
{
private :

    double m_initialValue ; /// actualized initial asset value
    double m_alpha0 ; /// First parameter for economic growth
    double m_eta  ; /// Second parameter for economic growth
    double m_T ; ///< maturity of the option
    double m_step ; ///< simulation step
    size_t  m_nbStep ; ///< number of time steps
    size_t m_nbSimul ; ///< Monte Carlo simulation number
    bool m_bForward ; ///< True if forward mode
    std::shared_ptr<gs::BinaryFileArchive> m_arch ; ///< Archive
    boost::mt19937 m_generator;  ///< Boost random generator
    boost::normal_distribution<double> m_normalDistrib;
    boost::variate_generator<boost::mt19937 &, boost::normal_distribution<double> > m_normalRand ; ///< Normal
    int m_currentStep ; ///< current step number (0 to m_nbStep)
    Eigen::MatrixXd m_SMMM ; ///< simulation  size (1,number of simulation)

public :

    /// \brief constructor
    /// \param p_initialValue  initial value
    /// \param p_alpha0        First parameter for economic growth
    /// \param p_eta           Second parameter for economic growth
    /// \param p_T              maturity
    /// \param p_nbStep         number of time step
    /// \param p_nbSimul        number of simulations
    /// \param p_bForward       true if the simulation is forward, false if backward
    /// \param p_nameArch       name of the archive
    MMMSimulator(const double &p_initialValue,
                 const double &p_alpha0,
                 const double &p_eta,
                 const double &p_T,
                 const size_t &p_nbStep,
                 const size_t &p_nbSimul,
                 const bool    &p_bForward,
                 const std::string   &p_nameArch
#ifdef USE_MPI
                 , const boost::mpi::communicator &p_world
#endif
                ):
        m_initialValue(p_initialValue), m_alpha0(p_alpha0), m_eta(p_eta),
        m_T(p_T), m_step(p_T / p_nbStep), m_nbStep(p_nbStep), m_nbSimul(p_nbSimul), m_bForward(p_bForward),
        m_generator(), m_normalDistrib(), m_normalRand(m_generator, m_normalDistrib)
    {
#ifdef USE_MPI
        if (p_world.rank() == 0)
#endif
            m_arch = std::make_shared<gs::BinaryFileArchive>(p_nameArch.c_str(), "w");
        // local function
        double alphaLoc = m_alpha0;
        double etaLoc = m_eta;
        auto varPhi([alphaLoc, etaLoc](const double & x)
        {
            return (alphaLoc / (4 * etaLoc)) * (exp(etaLoc * x) - 1.);
        });
        std::function< double(const double &x)> fvarPhi(std::cref(varPhi));

        Eigen::VectorXd varPhiTab(m_nbStep + 1);
        varPhiTab(0) = 0;
        for (size_t i = 0; i < m_nbStep; ++i)
            varPhiTab(i + 1) = fvarPhi((i + 1) * m_step);
        Eigen::VectorXd varPhiTabStep(m_nbStep);
        for (size_t i = 0; i < m_nbStep; ++i)
            varPhiTabStep(i) = sqrt(varPhiTab(i + 1) - varPhiTab(i));

        // to store asset value
        m_SMMM =  Eigen::MatrixXd::Constant(1, m_nbSimul, p_initialValue);

        // store initial
#ifdef USE_MPI
        if (p_world.rank() == 0)
#endif
            *m_arch << gs::Record(m_SMMM, "allStep", "SimulMMM");

        Eigen::ArrayXXd incBrown = Eigen::ArrayXXd::Zero(m_nbSimul, 4);
        incBrown.col(0).setConstant(sqrt(p_initialValue));
        for (size_t i = 0; i < m_nbStep; ++i)
        {
            for (size_t is = 0; is < m_nbSimul; ++is)
            {
                incBrown(is, 0) += varPhiTabStep(i) * m_normalRand();
                incBrown(is, 1) += varPhiTabStep(i) * m_normalRand();
                incBrown(is, 2) += varPhiTabStep(i) * m_normalRand();
                incBrown(is, 3) += varPhiTabStep(i) * m_normalRand();
                m_SMMM(0, is) =  pow(incBrown(is, 0), 2.) + pow(incBrown(is, 1), 2) + pow(incBrown(is, 2), 2.) + pow(incBrown(is, 3), 2);
            }
#ifdef USE_MPI
            if (p_world.rank() == 0)
#endif
                *m_arch << gs::Record(m_SMMM, "allStep", "SimulMMM");
        }
#ifdef USE_MPI
        if (p_world.rank() == 0)
#endif
            m_arch->flush();
#ifdef USE_MPI
        p_world.barrier();
#endif
        // all processor read
        m_arch = std::make_shared<gs::BinaryFileArchive>(p_nameArch.c_str(), "r");
        if (p_bForward)
            m_currentStep = 0;
        else
            m_currentStep = m_nbStep;
        gs::Reference< Eigen::MatrixXd > (*m_arch, "allStep", "SimulMMM").restore(m_currentStep, &m_SMMM);

    }

    /// \brief initialize (keep same simulation)
    void resetDirection(const bool &bForward)
    {
        m_bForward = bForward;
        if (m_bForward)
            m_currentStep = 0;
        else
            m_currentStep = m_nbStep;
        gs::Reference< Eigen::MatrixXd > (*m_arch, "allStep", "SimulMMM").restore(m_currentStep, &m_SMMM);
    }


    /// \brief get current asset values
    Eigen::MatrixXd getParticles() const
    {
        return m_SMMM;
    }

    /// \brief a step forward for simulations
    void  stepForward()
    {
        m_currentStep += 1;
        gs::Reference< Eigen::MatrixXd > (*m_arch, "allStep", "SimulMMM").restore(m_currentStep, &m_SMMM);
    }

    /// \return  the asset values (asset,simulations)
    void  stepBackward()
    {
        m_currentStep -= 1;
        gs::Reference< Eigen::MatrixXd > (*m_arch, "allStep", "SimulMMM").restore(m_currentStep, &m_SMMM);
    }

    /// \brief a step forward for simulations
    /// \return  the asset values (asset,simulations)
    Eigen::MatrixXd  stepForwardAndGetParticles()
    {
        m_currentStep += 1;
        gs::Reference< Eigen::MatrixXd > (*m_arch, "allStep", "SimulMMM").restore(m_currentStep, &m_SMMM);
        return m_SMMM;
    }

    /// \brief a step backward for simulations
    /// \return  the asset values (asset,simulations)
    Eigen::MatrixXd stepBackwardAndGetParticles()
    {
        m_currentStep -= 1;
        gs::Reference< Eigen::MatrixXd > (*m_arch, "allStep", "SimulMMM").restore(m_currentStep, &m_SMMM);
        return m_SMMM;
    }
    ///@{
    /// Get back attribute
    inline double getCurrentStep() const
    {
        return m_currentStep * m_step;
    }
    inline double getT() const
    {
        return m_T;
    }
    inline double getStep() const
    {
        return m_step;
    }
    inline Eigen::VectorXd getInitialValues() const
    {
        return Eigen::VectorXd::Constant(1, m_initialValue);
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
    /// actualization with the interest rate  : 0
    inline double getActuStep() const
    {
        return  1. ;
    }
    /// actualization for initial date
    inline double getActu() const
    {
        return  1.;
    }
    ///@}
};

#endif
