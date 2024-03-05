// Copyright (C) 2021 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#ifndef MEANREVERTING1DASSETSSIMULATOR_H
#define MEANREVERTING1DASSETSSIMULATOR_H
#include <memory>
#include <boost/random.hpp>
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include "geners/BinaryFileArchive.hh"
#include "geners/Record.hh"
#include "geners/Reference.hh"
#include "libflow/core/utils/eigenGeners.h"
#include "libflow/core/utils/constant.h"
#include "libflow/dp/SimulatorDPBase.h"


/* \file MeanReverting1DAssetSimulator.h
 * \brief Simulate some futures deformation with a one dimensional Ornstein Uhlenbeck process.
 *        \f$ dF(t,T) = F(t,T)  e^{-a (T-t)} \sigma dW_t  \f$
 *        Each asset follow the previous dynamic and brownian are correlated
 * \author Xavier Warin
 */


/// \class MeanReverting1DAssetSimulator MeanReverting1DAssetSimulator.h
///
template< class Curve>
class MeanReverting1DAssetsSimulator: public libflow::SimulatorDPBase
{
protected :
    std::vector< std::shared_ptr<Curve> >  m_curve;
    std::vector<double>  m_mr ; // mean reverting for each asset
    std::vector<double>  m_sig; // volatility for each asset
    Eigen::MatrixXd m_correl ; // correlation matrix;
    double m_T ; ///< maturity of the option
    double m_step ; ///< simulation step
    int  m_nbStep ; ///< number of time steps
    size_t m_nbSimul ; ///< Monte Carlo simulation number
    bool m_bForward ; ///< True if forward mode
    int  m_currentStep ; ///< Current time step  number
    Eigen::MatrixXd m_OUProcess; ///< store the Ornstein Uhlenbeck process (no correlation)
    boost::mt19937 m_generator;  ///< Boost random generator
    boost::normal_distribution<double> m_normalDistrib; ///< Normal distribution
    boost::variate_generator<boost::mt19937 &, boost::normal_distribution<double> > m_normalRand ; ///< Normal generator
    std::shared_ptr<gs::BinaryFileArchive> m_arch ; // archive to store simulations


public:

/// \brief Constructor
    MeanReverting1DAssetsSimulator(const std::vector< std::shared_ptr<Curve> >   &p_curve,
                                   const std::vector<double>   &p_mr,
                                   const std::vector<double>   &p_sig,
                                   const Eigen::MatrixXd &p_correl,
                                   const double &p_T,
                                   const size_t &p_nbStep,
                                   const size_t &p_nbSimul,
                                   const bool &p_bForward,
                                   const std::string   &p_nameArch,
#ifdef USE_MPI
                                   const boost::mpi::communicator &p_world,
#endif

                                   int p_seed = 1):
        m_curve(p_curve), m_mr(p_mr), m_sig(p_sig), m_correl(p_correl),
        m_T(p_T), m_step(p_T / p_nbStep), m_nbStep(p_nbStep), m_nbSimul(p_nbSimul), m_bForward(p_bForward),
        m_currentStep(p_bForward ? 0. : m_nbStep), m_OUProcess(p_sig.size(), p_nbSimul),
        m_generator(), m_normalDistrib(), m_normalRand(m_generator, m_normalDistrib)
    {
        // affect seed
        m_normalRand.engine().seed(p_seed);
        // simulate and store OU processus
        m_OUProcess = Eigen::MatrixXd::Constant(p_curve.size(), m_nbSimul, 0.);
        // initial dump
#ifdef USE_MPI
        if (p_world.rank() == 0)
        {
#endif
            m_arch = std::make_shared<gs::BinaryFileArchive>(p_nameArch.c_str(), "w");
            *m_arch << gs::Record(m_OUProcess, "allStep", "SimuMRNAssets");
#ifdef USE_MPI
        }
#endif
        for (size_t i = 0; i < m_nbStep; ++i)
        {
            // correlationj of Uncertaintes
            Eigen::MatrixXd correloc = Eigen::MatrixXd::Zero(p_curve.size(), p_curve.size());
            for (size_t i = 0; i < p_curve.size(); ++i)
            {
                for (size_t j = 0; j < i; ++j)
                {
                    double aE1E2 = (1 - exp(-(m_mr[i] + m_mr[j]) * m_step)) / (m_mr[i] + m_mr[j]);
                    double aE1 = sqrt((1 - exp(-2 * m_mr[i] * m_step)) / (2 * m_mr[i]));
                    double aE2 =  sqrt((1 - exp(-2 * m_mr[j] * m_step)) / (2 * m_mr[j]));
                    correloc(i, j) = m_correl(i, j) * aE1E2 / (aE1 * aE2);
                }
            }
            // symmetrize, add identity
            correloc = correloc + correloc.transpose() + Eigen::MatrixXd::Identity(p_curve.size(), p_curve.size());
            // factorize
            Eigen::MatrixXd correlFac = correloc.llt().matrixL();

            // nest on simulations
            for (size_t is = 0; is < m_nbSimul; ++is)
            {
                Eigen::VectorXd normalVar(correlFac.rows());
                for (int id = 0 ; id < correlFac.rows(); ++id)
                {
                    normalVar(id) = m_normalRand();
                }
                Eigen::VectorXd normalVarCorrel = correlFac * normalVar;
                // prices
                for (size_t i = 0; i < p_curve.size(); ++i)
                {
                    double expActuE = exp(- m_mr[i] * m_step);
                    double aE = sqrt((1 - exp(-2 * m_mr[i] * m_step)) / (2 * m_mr[i]));
                    double etypE = m_sig[i] * aE;
                    m_OUProcess(i, is) = m_OUProcess(i, is) * expActuE + etypE * normalVarCorrel(i);
                }

            }
#ifdef USE_MPI
            if (p_world.rank() == 0)
#endif
                *m_arch << gs::Record(m_OUProcess, "allStep", "SimuMRNAssets");
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
        gs::Reference< Eigen::MatrixXd > (*m_arch, "allStep", "SimuMRNAssets").restore(m_currentStep, &m_OUProcess);
    }



    /// \brief initialize (keep same simulation)
    void resetDirection(const bool &bForward)
    {
        m_bForward = bForward;
        if (m_bForward)
            m_currentStep = 0;
        else
            m_currentStep = m_nbStep;
        gs::Reference< Eigen::MatrixXd > (*m_arch, "allStep", "SimuMRNAssets").restore(m_currentStep, &m_OUProcess);
    }


    /// \brief get current regression factors
    Eigen::MatrixXd getParticles() const
    {
        return m_OUProcess;
    }

    /// \brief a step forward for simulations
    void  stepForward()
    {
        m_currentStep += 1;
        gs::Reference< Eigen::MatrixXd > (*m_arch, "allStep", "SimuMRNAssets").restore(m_currentStep, &m_OUProcess);
    }

    /// \return  the asset values (asset,simulations)
    void  stepBackward()
    {
        m_currentStep -= 1;
        gs::Reference< Eigen::MatrixXd > (*m_arch, "allStep", "SimuMRNAssets").restore(m_currentStep, &m_OUProcess);
    }

    /// \brief a step forward for simulations
    /// \return  the asset values (asset,simulations)
    Eigen::MatrixXd  stepForwardAndGetParticles()
    {
        m_currentStep += 1;
        gs::Reference< Eigen::MatrixXd > (*m_arch, "allStep", "SimuMRNAssets").restore(m_currentStep, &m_OUProcess);
        return m_OUProcess;
    }

    /// \brief a step backward for simulations
    /// \return  the asset values (asset,simulations)
    Eigen::MatrixXd stepBackwardAndGetParticles()
    {
        m_currentStep -= 1;
        gs::Reference< Eigen::MatrixXd > (*m_arch, "allStep", "SimuMRNAssets").restore(m_currentStep, &m_OUProcess);
        return m_OUProcess;
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
        return m_curve.size();
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


    /// \brief Special function for the spot
    ///
    Eigen::ArrayXXd getSpot() const
    {
        Eigen::ArrayXXd spot(m_curve.size(), m_nbSimul);
        for (size_t i = 0; i <  m_curve.size(); ++i)
        {
            double trendSpot =  -0.5 * m_sig[i] * m_sig[i] * (1. - exp(-2 * m_mr[i] * m_currentStep * m_step)) / (2.*m_mr[i]);
            for (size_t is = 0; is < m_nbSimul; ++is)
            {
                spot(i, is) = m_curve[i]->get(m_currentStep * m_step) * exp(trendSpot + m_OUProcess(i, is));
            }
        }
        return spot;
    }

    /// \brief from OU uncertainties to spot
    Eigen::ArrayXd fromOUToSpot(const Eigen::ArrayXd &p_OUState) const
    {
        Eigen::ArrayXd spot(p_OUState.size());
        for (int iAsset = 0 ; iAsset < p_OUState.size(); ++iAsset)
        {
            double trendSpot =  -0.5 * m_sig[iAsset] * m_sig[iAsset] * (1. - exp(-2 * m_mr[iAsset] * m_currentStep * m_step)) / (2.*m_mr[iAsset]);
            spot(iAsset) = m_curve[iAsset]->get(m_currentStep * m_step) * exp(trendSpot + p_OUState(iAsset));
        }
        return spot;
    }

};


#endif
