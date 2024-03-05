// Copyright (C) 2016 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#ifndef OPTIMIZESTORAGEWITHDEMANDBASESDDP_H
#define OPTIMIZESTORAGEWITHDEMANDBASESDDP_H
#include "ClpSimplex.hpp"
#include <boost/lexical_cast.hpp>
#include "libflow/sddp/SDDPCutOptBase.h"
#include "libflow/sddp/OptimizerSDDPBase.h"
#include "libflow/core/utils/constant.h"

/** \file OptimizeStorageWithDemandBaseSDDP.h
 * \brief  Suppose that we have m_nbStorage Reservoir to satisfy demand
 *         Each reservoir is characterized by some inflows \f$f\f$ following an AR1 model
 *         \f$ f^{n+1} = \kappa (f^n -f)  + \sigma_f g  + \kappa f \f$ where \f$g\f$ is a Gaussian noise
 *         The goal is to satisfy a demand  following an AR1 model:
 *         \f$ D^{n+1} = k (D^n-D) + \sigma_d \tilde g + k D \f$ where \f$ \tilde g \f$ is a Gaussian noise
 *         When there is a shortage of water, energy is bought at a given price
 * \author Xavier Warin
 */

/// \class OptimizeStorageWithDemandBaseSDDP OptimizeStorageWithDemandBaseSDDP.h
///
template< class Simulator>
class OptimizeStorageWithDemandBaseSDDP: public libflow::OptimizerSDDPBase
{

protected :

    /// \brief Physical constraints for storage
    //@{
    double m_withdrawalRate ;  ///< withdrawal rate (volume) per time step
    int m_nbStorage ; ///< Number of storage
    //@}

    /// \brief AR 1 model for inflows
    //@{
    double m_sigF ; ///< volatility of inflows \f$\sigma_f\f$
    double m_kappaF ; ///<  AR coefficient for inflows \f$ \kappa \f$
    std::shared_ptr<libflow::OneDimData<libflow::OneDimRegularSpaceGrid, double> > m_timeInflowAver ; /// store the average  inflow depending on time
    double m_InflowAver ; ///< Average value for inflows \f$f\f$ at current date
    double m_InflowAverNext ; ///< Average value for inflows \f$f\f$ at next date
    //@}

    /// \brief AR 1 model for demand
    //@{
    double m_sigD ; /// volatility for demand \f$ \sigma_d \f$
    double m_kappaD ; /// AR coefficient for demand  \f$ k \f$
    std::shared_ptr<libflow::OneDimData<libflow::OneDimRegularSpaceGrid, double> > m_timeDAverage; /// store the average demand depending on time
    double m_DAverage ; ///< average value for demand at current date
    double m_DAverageNext ; ///< average value for demand at next time  date
    //@}

    double m_date ; ///< current date
    double m_dateNext ; ///< next date (after current date)

    std::shared_ptr< Simulator> m_simulatorBackward ; ///< for backward simulations
    std::shared_ptr< Simulator> m_simulatorForward ; ///< for forward simulations

    /// \brief LP creation
    /// \param p_linCut               cuts used for the PL   (Benders for the Bellman value at the end of the time step)
    /// \param p_stateLevel           Store the state  : storage levels, inflows levels,  demand level
    /// \param p_constraints          Constraints to add to optimizer
    /// \param p_spot                 spot price
    /// \param p_valueAndDerivatives  optimal value of the function and derivatives
    /// \param p_stateFollowing       To store state after optimal command (if the state is not controlled, in contains the value at following time step : inflows, demand)
    /// \param p_cost                 instantaneous cost
    template< class TConstraint>
    void createAndSolveLP(const libflow::SDDPCutOptBase   &p_linCut, const Eigen::ArrayXd &p_stateLevel,
                          const TConstraint &p_constraints, const double &p_spot, Eigen::ArrayXd &p_valueAndDerivatives,
                          Eigen::ArrayXd &p_stateFollowing, double &p_cost) const
    {
        // optimizer matrix constraints
        Eigen::ArrayXi rows(5 * m_nbStorage + 3); // row position
        Eigen::ArrayXi columns(5 * m_nbStorage + 3) ; // columns  position
        Eigen::ArrayXd elements(5 * m_nbStorage + 3) ; // constraints matrix values
        // bounds associated to matrix constraints
        Eigen::ArrayXd lowBoundConst(2 * m_nbStorage + 2);
        Eigen::ArrayXd upperBoundConst(2 * m_nbStorage + 2) ;

        // bounds on values
        Eigen::ArrayXd lowBound(3 * m_nbStorage + 3);
        Eigen::ArrayXd upperBound(3 * m_nbStorage + 3);
        for (int isto = 0 ; isto < m_nbStorage; ++isto)
        {
            // withdrawal
            lowBound(isto) = - m_withdrawalRate;
            upperBound(isto) = 0;
            // storage bounds
            lowBound(isto + m_nbStorage) = 0;
            upperBound(isto + m_nbStorage) = libflow::infty;
            // inflows bounds
            lowBound(isto + 2 * m_nbStorage) = 0;
            upperBound(isto + 2 * m_nbStorage) = libflow::infty;
        }
        // add demand variable
        lowBound(3 * m_nbStorage) = 0.;
        upperBound(3 * m_nbStorage) = libflow::infty;

        // quantity bought on the spot
        lowBound(3 * m_nbStorage + 1) = 0.;
        upperBound(3 * m_nbStorage + 1) = libflow::infty;
        // for  fictitious data for bellman
        lowBound(3 * m_nbStorage + 2) = - libflow::infty;
        upperBound(3 * m_nbStorage + 2) = libflow::infty;

        // objective function
        Eigen::ArrayXd objFunc = Eigen::ArrayXd::Zero(3 * m_nbStorage + 3);
        objFunc(3 * m_nbStorage + 1) = p_spot;
        objFunc(3 * m_nbStorage + 2) = 1;

        // flow constraints for storage
        for (int isto = 0 ; isto < m_nbStorage  ; ++isto)
        {
            int ipos = 3 * isto;
            rows(ipos) =  isto;
            columns(ipos) = isto;
            elements(ipos) = -1;
            rows(ipos + 1) = isto;
            columns(ipos + 1) = isto + m_nbStorage;
            elements(ipos + 1) =  1;
            rows(ipos + 2) = isto;
            columns(ipos + 2) = isto + 2 * m_nbStorage;
            elements(ipos + 2) =  -1;
            lowBoundConst(isto) = p_stateLevel(isto) ;
            upperBoundConst(isto) = p_stateLevel(isto);
        }
        // flow contraint for inflows
        for (int isto = 0 ; isto < m_nbStorage  ; ++isto)
        {
            int ipos = 3 * m_nbStorage + isto;
            rows(ipos) =  m_nbStorage + isto;
            columns(ipos) = isto + 2 * m_nbStorage;
            elements(ipos) = 1;
            lowBoundConst(m_nbStorage + isto) = p_stateFollowing(m_nbStorage + isto) ;
            upperBoundConst(m_nbStorage + isto) = p_stateFollowing(m_nbStorage + isto);
        }

        // flow for demand
        rows(4 * m_nbStorage) = 2 * m_nbStorage;
        columns(4 * m_nbStorage) = 3 * m_nbStorage;
        elements(4 * m_nbStorage) = 1;
        lowBoundConst(2 * m_nbStorage) =  p_stateFollowing(2 * m_nbStorage);
        upperBoundConst(2 * m_nbStorage) =   p_stateFollowing(2 * m_nbStorage) ;

        // constraint for demand
        rows(4 * m_nbStorage + 1) = 2 * m_nbStorage + 1;
        columns(4 * m_nbStorage + 1) = 3 * m_nbStorage + 1; // bought on market
        elements(4 * m_nbStorage + 1) = 1;
        rows(4 * m_nbStorage + 2) = 2 * m_nbStorage + 1;
        columns(4 * m_nbStorage + 2) = 3 * m_nbStorage; // demand at next date
        elements(4 * m_nbStorage + 2) = -1;
        for (int isto = 0 ; isto < m_nbStorage  ; ++isto)
        {
            rows(4 * m_nbStorage + isto + 3) = 2 * m_nbStorage + 1;
            columns(4 * m_nbStorage + isto + 3) = isto;
            elements(4 * m_nbStorage + isto + 3) = -1;
        }
        lowBoundConst(2 * m_nbStorage + 1) =  0.;
        upperBoundConst(2 * m_nbStorage + 1) =  0. ;

        //  add cuts for bellman value to constraints
        p_constraints.addConstraints(p_linCut, m_nbStorage,  rows, columns, elements, lowBoundConst, upperBoundConst);

        //  model
        ClpSimplex  model;

#ifdef NDEBUG
        model.setLogLevel(0);
#endif

        model.loadProblem(CoinPackedMatrix(false, rows.data(), columns.data(), elements.data(), elements.size()), lowBound.data(), upperBound.data(), objFunc.data(), lowBoundConst.data(), upperBoundConst.data());

        ClpSolve solvectl;
        solvectl.setSolveType(ClpSolve::usePrimal);
        //solvectl.setSolveType(ClpSolve::useDual);
        solvectl.setPresolveType(ClpSolve::presolveOn);
        model.initialSolve(solvectl);

        bool modelSolved = model.isProvenOptimal();

        if (!modelSolved)
            std::cout << "[problemLP::solveModel] : Warning Linear program could not be solved optimally somehow\n";

        // optimal value
        p_valueAndDerivatives(0) = model.objectiveValue();

        // duals
        double *dual = model.dualRowSolution();
        for (int isto = 0; isto < m_nbStorage ; ++isto)
            p_valueAndDerivatives(1 + isto) = dual[isto];
        for (int isto = 0; isto < m_nbStorage ; ++isto)
            p_valueAndDerivatives(m_nbStorage + 1 + isto) = dual[isto + m_nbStorage] * m_kappaF;
        p_valueAndDerivatives(2 * m_nbStorage + 1) = dual[2 * m_nbStorage] * m_kappaD;

        // primal
        double *columnPrimal = model.primalColumnSolution();
        // for each stock
        for (int isto = 0 ; isto < m_nbStorage ; ++isto)
            p_stateFollowing(isto) =  columnPrimal[isto + m_nbStorage];
        // cost
        p_cost = p_spot * columnPrimal[3 * m_nbStorage + 1] ;
    }

public :


/// \brief Constructor for the storage problem
/// \param   p_withdrawalRate      withdrawal rate (volume) per time step
/// \param   p_nbStorage           Number of storage
/// \param   p_sigF                Volatility for inflows
/// \param   p_kappaF              AR coefficient for inflows
/// \param   p_timeInflowAver      Average inflow
/// \param   p_sigD                volatility for demand
/// \param   p_kappaD              AR coefficient for demand
/// \param   p_timeDAverage        average demand
    /// \param   p_simulatorBackward   backward  simulator
    /// \param   p_simulatorForward    Forward simulator
    OptimizeStorageWithDemandBaseSDDP(const double &p_withdrawalRate,  const int &p_nbStorage,
                                      const double &p_sigF,  const double &p_kappaF,
                                      const std::shared_ptr<libflow::OneDimData<libflow::OneDimRegularSpaceGrid, double> >    &p_timeInflowAver,
                                      const  double   &p_sigD, const double &p_kappaD,
                                      const std::shared_ptr<libflow::OneDimData<libflow::OneDimRegularSpaceGrid, double> > &p_timeDAverage,
                                      const std::shared_ptr<Simulator> &p_simulatorBackward,
                                      const std::shared_ptr<Simulator> &p_simulatorForward): m_withdrawalRate(p_withdrawalRate),
        m_nbStorage(p_nbStorage), m_sigF(p_sigF), m_kappaF(p_kappaF),  m_timeInflowAver(p_timeInflowAver), m_sigD(p_sigD),
        m_kappaD(p_kappaD), m_timeDAverage(p_timeDAverage), m_simulatorBackward(p_simulatorBackward), m_simulatorForward(p_simulatorForward)
    {}



    /// brief update the optimizer for new dates (current date and next dates)
    void updateDates(const double &p_date, const double &p_dateNext)
    {
        m_date = p_date ;
        if (libflow::isLesserOrEqual(0., p_date))
        {
            m_InflowAver = m_timeInflowAver->get(m_date);
            m_DAverage = m_timeDAverage->get(m_date);
        }
        m_dateNext = p_dateNext ;
        m_InflowAverNext = m_timeInflowAver->get(m_dateNext);
        m_DAverageNext = m_timeDAverage->get(m_dateNext);
    }

    /// \brief Get an admissible state for a given date
    /// \param  p_date   current date
    /// \return an admissible state
    Eigen::ArrayXd oneAdmissibleState(const double &p_date)
    {
        Eigen::ArrayXd toRet(2 * m_nbStorage + 1);
        toRet.head(m_nbStorage) = Eigen::ArrayXd::Constant(m_nbStorage, 0);
        toRet.segment(m_nbStorage, m_nbStorage) = Eigen::ArrayXd::Constant(m_nbStorage, m_timeInflowAver->get(0));
        toRet(2 * m_nbStorage) = m_timeDAverage->get(p_date);
        return toRet;
    }


    /// \brief get back state size : number of storage + number of inflows + one for demand
    inline int getStateSize() const
    {
        return 2 * m_nbStorage + 1;
    }

    /// \brief get the backward simulator back
    std::shared_ptr< libflow::SimulatorSDDPBase > getSimulatorBackward() const
    {
        return m_simulatorBackward ;
    }

    /// \brief get the forward simulator back
    std::shared_ptr< libflow::SimulatorSDDPBase > getSimulatorForward() const
    {
        return m_simulatorForward ;
    }

};
#endif /* OPTIMIZESTORAGEWITHDEMANDBASESDDP_H */
