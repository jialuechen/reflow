
#ifndef OPTIMIZERESERVOIRWITHINFLOWSSDDP_H
#define OPTIMIZERESERVOIRWITHINFLOWSSDDP_H
#include "ClpSimplex.hpp"
#include <boost/lexical_cast.hpp>
#include "libflow/sddp/SDDPCutOptBase.h"
#include "libflow/sddp/OptimizerSDDPBase.h"
#include "libflow/core/grids/OneDimRegularSpaceGrid.h"
#include "libflow/core/grids/OneDimData.h"
#include "libflow/core/utils/comparisonUtils.h"
#include "libflow/core/utils/constant.h"

/** \file OptimizeReservoirWithInflowsSDDP.h
 * \brief  Suppose that we have m_nbStorage Reservoirs to satisfy demand
 *         At each time step  the demand follows a random Gaussian noise, and inflows too
 * \author Xavier Warin
 */

/// \class AddConstraintReservoir OptimizeReservoirWithInflowsSDDP.h
/// Add constraints to the Bellman value with cuts
class AddConstraintReservoir
{

public:
    /// \brief  add constraints to Bellman value
    /// \param  p_linCut          cuts stored
    /// \param  p_nbStorage       number of storage
    /// \param  p_rows           rows for matrix contraints
    /// \param  p_columns        columns for matrix contraints
    /// \param  p_elements       A matrix elements
    /// \param  p_lowBoundConst  lower constraint \f$ lc\f$  on matrix \f$ lc \le A x \f$
    /// \param  p_upperBoundConst upper constraint \f$ uc\f$  on matrix \f$ A x \le uc \f$
    void addConstraints(const libflow::SDDPCutOptBase &p_linCut, int p_nbStorage, Eigen::ArrayXi &p_rows,   Eigen::ArrayXi   &p_columns,  Eigen::ArrayXd   &p_elements,
                        Eigen::ArrayXd    &p_lowBoundConst,  Eigen::ArrayXd   &p_upperBoundConst) const
    {
        // get back cuts
        // here we use particle 0 because cuts are independent of a particle number
        Eigen::ArrayXd aParticle;
        Eigen::ArrayXXd  cuts =  p_linCut.getCutsAssociatedToAParticle(aParticle);
        int iBellPos = p_nbStorage * 2 + 1; // offset in variables
        int idecToStock = p_nbStorage; // p_nbStorage first values  used for withdrawal
        int isizeInit = p_elements.size();
        p_rows.conservativeResize(isizeInit + (p_nbStorage + 1)*cuts.cols());
        p_columns.conservativeResize(isizeInit + (p_nbStorage + 1)*cuts.cols());
        p_elements.conservativeResize(isizeInit + (p_nbStorage + 1)*cuts.cols());
        int ibound = p_lowBoundConst.size();
        p_lowBoundConst.conservativeResize(ibound + cuts.cols());
        p_upperBoundConst.conservativeResize(ibound + cuts.cols());
        // add cuts
        for (int icut = 0 ; icut < cuts.cols() ; ++icut)
        {
            double affineValue = cuts(0, icut);
            int ipos = isizeInit + (p_nbStorage + 1) * icut;
            p_rows(ipos) = ibound + icut;
            p_columns(ipos) = iBellPos;
            p_elements(ipos) = 1;
            for (int isto = 0 ; isto < p_nbStorage ; ++isto)
            {
                p_rows(ipos + isto + 1) = ibound + icut;
                p_columns(ipos + isto + 1) = idecToStock + isto;
                double derivStorage =  cuts(1 + isto, icut);
                p_elements(ipos + isto + 1) = -derivStorage;
            }
            p_lowBoundConst(ibound + icut) = affineValue;
            p_upperBoundConst(ibound + icut) = libflow::infty;
        }
    }
};


/// \class OptimizeReservoirWithInflowsSDDP OptimizeReservoirWithInflowsSDDP.h
///
template< class Simulator>
class OptimizeReservoirWithInflowsSDDP: public libflow::OptimizerSDDPBase
{

private :

    /// \brief Physical constraints for storage
    //@{
    double m_withdrawalRate ;  ///< withdrawal rate (volume) per time step
    int m_nbStorage ; ///< Number of storage
    double	m_initialLevel ; ///< initial level of the storage
    //@}

    /// \brief Gaussian random inflows
    //@{
    double m_sigF ; ///< volatility of inflows \f$\sigma_f\f$
    std::shared_ptr<libflow::OneDimData<libflow::OneDimRegularSpaceGrid, double> > m_timeInflowAver ; /// store the average  inflow depending on time
    double m_InflowAver ; ///< Average value for inflows \f$f\f$ at current date
    double m_InflowAverNext ; ///< Average value for inflows \f$f\f$ at next date
    //@}

    /// \brief Gaussian random  demand
    //@{
    double m_sigD ; /// volatility for demand \f$ \sigma_d \f$
    std::shared_ptr<libflow::OneDimData<libflow::OneDimRegularSpaceGrid, double> > m_timeDAverage; /// store the average demand depending on time
    double m_DAverage ; ///< average value for demand at current date
    double m_DAverageNext ; ///< average value for demand at next time  date
    //@}

    std::shared_ptr<libflow::OneDimData<libflow::OneDimRegularSpaceGrid, double> > m_timeSpot; ///< store the grid of spot price depending on time
    double m_spot ; ///< deterministic spot price at current date
    double m_spotNext ; ///< deterministic spot price at next date

    double m_date ; ///< current date

    std::shared_ptr< Simulator> m_simulatorBackward ; // for backward simulations
    std::shared_ptr< Simulator> m_simulatorForward ; // for forward simulations

    /// \brief LP creation
    /// \param p_linCut               cuts used for the PL   (Benders for the Bellman value at the end of the time step)
    /// \param p_stateLevel           Store the state  : storage levels, inflows levels,  demand level
    /// \param p_constraints          Constraints to add to optimizer
    /// \param p_spot                 spot price
    /// \param p_valueAndDerivatives  optimal value of the function and derivatives
    /// \param p_inflows              inflows at current time step
    /// \param p_demand               demand at current time step
    /// \param p_stateFollowing       state following
    /// \param p_cost                 instantaneous cost
    template< class TConstraint>
    void createAndSolveLP(const libflow::SDDPCutOptBase   &p_linCut, const Eigen::ArrayXd &p_stateLevel,
                          const TConstraint &p_constraints,  const double &p_spot, Eigen::ArrayXd &p_valueAndDerivatives,
                          const Eigen::ArrayXd &p_inflows, const double &p_demand,
                          Eigen::ArrayXd &p_stateFollowing, double &p_cost) const
    {
        // optimizer matrix constraints
        Eigen::ArrayXi rows(3 * m_nbStorage + 1); // row position
        Eigen::ArrayXi columns(3 * m_nbStorage + 1) ; // columns  position
        Eigen::ArrayXd elements(3 * m_nbStorage + 1) ; // constraints matrix values
        // bounds associated to matrix constraints
        Eigen::ArrayXd lowBoundConst(m_nbStorage + 1);
        Eigen::ArrayXd upperBoundConst(m_nbStorage + 1) ;

        // bounds on values
        Eigen::ArrayXd lowBound(2 * m_nbStorage + 2);
        Eigen::ArrayXd upperBound(2 * m_nbStorage + 2);
        for (int isto = 0 ; isto < m_nbStorage; ++isto)
        {
            // withdrawal
            lowBound(isto) = - m_withdrawalRate;
            upperBound(isto) = 0;
            // storage bounds
            lowBound(isto + m_nbStorage) = 0;
            upperBound(isto + m_nbStorage) = libflow::infty;
        }
        // quantity bought on the spot
        lowBound(2 * m_nbStorage) = 0.;
        upperBound(2 * m_nbStorage) = libflow::infty;
        // for  fictitious data for bellman
        lowBound(2 * m_nbStorage + 1) = - libflow::infty;
        upperBound(2 * m_nbStorage + 1) = libflow::infty;

        // objective function
        Eigen::ArrayXd objFunc = Eigen::ArrayXd::Zero(2 * m_nbStorage + 2);
        objFunc(2 * m_nbStorage) = p_spot;
        objFunc(2 * m_nbStorage + 1) = 1;

        // flow constraints
        for (int isto = 0 ; isto < m_nbStorage  ; ++isto)
        {
            int ipos = 2 * isto;
            rows(ipos) =  isto;
            columns(ipos) = isto;
            elements(ipos) = -1;
            rows(ipos + 1) = isto;
            columns(ipos + 1) = isto + m_nbStorage;
            elements(ipos + 1) =  1;
            lowBoundConst(isto) = p_stateLevel(isto) + p_inflows(isto) ;
            upperBoundConst(isto) = p_stateLevel(isto) + p_inflows(isto) ;
        }
        // constraint for demand
        rows(2 * m_nbStorage) = m_nbStorage;
        columns(2 * m_nbStorage) = 2 * m_nbStorage;
        elements(2 * m_nbStorage) = 1;
        for (int isto = 0 ; isto < m_nbStorage  ; ++isto)
        {
            rows(2 * m_nbStorage + isto + 1) = m_nbStorage;
            columns(2 * m_nbStorage + isto + 1) = isto;
            elements(2 * m_nbStorage + isto + 1) = -1;
        }
        lowBoundConst(m_nbStorage) = p_demand;
        upperBoundConst(m_nbStorage) = p_demand;
        //  add cuts for bellman value to constraints
        p_constraints.addConstraints(p_linCut, m_nbStorage, rows, columns, elements, lowBoundConst, upperBoundConst);

        //  model
        ClpSimplex  model;

#ifdef NDEBUG
        model.setLogLevel(0);
#endif

        model.loadProblem(CoinPackedMatrix(false, rows.data(), columns.data(), elements.data(), elements.size()), lowBound.data(), upperBound.data(), objFunc.data(), lowBoundConst.data(), upperBoundConst.data());

        ClpSolve solvectl;
        //solvectl.setSolveType(ClpSolve::usePrimal);
        solvectl.setSolveType(ClpSolve::useDual);
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

        // primal
        double *columnPrimal = model.primalColumnSolution();
        // for each stock
        for (int isto = 0 ; isto < m_nbStorage ; ++isto)
        {
            p_stateFollowing(isto) =  columnPrimal[isto + m_nbStorage];
        }
        // cost
        p_cost = p_spot * columnPrimal[2 * m_nbStorage] ;
    }

public :


/// \brief Constructor for the storage problem
/// \param   p_initialLevel        initial level of the reservoir
/// \param   p_withdrawalRate      withdrawal rate (volume) per time step
/// \param   p_nbStorage           Number of storage
/// \param   p_sigF                Volatility for inflows
/// \param   p_timeInflowAver      Average inflow
/// \param   p_sigD                volatility for demand
/// \param   p_timeDAverage        average demand
/// \param   p_timeSpot            spot price
    /// \param   p_simulatorBackward   backward  simulator
    /// \param   p_simulatorForward    Forward simulator
    OptimizeReservoirWithInflowsSDDP(const double &p_initialLevel,
                                     const double &p_withdrawalRate,  const int &p_nbStorage,
                                     const double &p_sigF,  const std::shared_ptr<libflow::OneDimData<libflow::OneDimRegularSpaceGrid, double> >    &p_timeInflowAver,
                                     const  double   &p_sigD,  const std::shared_ptr<libflow::OneDimData<libflow::OneDimRegularSpaceGrid, double> > &p_timeDAverage,
                                     const  std::shared_ptr<libflow::OneDimData<libflow::OneDimRegularSpaceGrid, double> >   &p_timeSpot,
                                     const std::shared_ptr<Simulator> &p_simulatorBackward,
                                     const std::shared_ptr<Simulator> &p_simulatorForward):
        m_initialLevel(p_initialLevel),  m_withdrawalRate(p_withdrawalRate),
        m_nbStorage(p_nbStorage), m_sigF(p_sigF),    m_timeInflowAver(p_timeInflowAver),  m_sigD(p_sigD), m_timeDAverage(p_timeDAverage), m_timeSpot(p_timeSpot),
        m_simulatorBackward(p_simulatorBackward), m_simulatorForward(p_simulatorForward)
    {}



    /// \brief Optimize the LP during backward resolution
    /// \param p_linCut            cuts used for the PL   (Benders for the Bellman value at the end of the time step)
    /// \param p_aState            Store the state, and 0, 0 values
    /// \param p_particle          Here no regression , so empty array
    /// \param p_isample            sample number for independent uncertainties
    /// \return  a vector with the optimal value and the derivatives if the function value with respect to each state (here the stocks)
    Eigen::ArrayXd oneStepBackward(const libflow::SDDPCutOptBase &p_linCut, const std::tuple< std::shared_ptr<Eigen::ArrayXd>, int, int > &p_aState,
                                   const Eigen::ArrayXd &p_particle, const int &p_isample) const
    {
        // constraints
        AddConstraintReservoir constraints;
        // Creation  and PL resolution
        Eigen::ArrayXd inflows(m_nbStorage);
        double demand  ;
        if (libflow::isLesserOrEqual(0., m_date))
        {
            for (int isto = 0; isto < m_nbStorage; ++isto)
                inflows(isto) = std::max(m_InflowAverNext + m_sigF * m_simulatorBackward->getGaussian(isto, p_isample), 0.);
            demand  = std::max(m_DAverageNext + m_sigD * m_simulatorBackward->getGaussian(m_nbStorage, p_isample), 0.);
        }
        else
        {
            inflows.setConstant(m_InflowAverNext);
            demand = m_DAverageNext;
        }
        double cost ;
        Eigen::ArrayXd stateFollowing(m_nbStorage);
        Eigen::ArrayXd  valueAndDerivatives(1 + m_nbStorage);
        createAndSolveLP(p_linCut, *std::get<0>(p_aState), constraints, m_spotNext, valueAndDerivatives, inflows, demand,   stateFollowing, cost);
        return valueAndDerivatives;
    }


    /// \brief Optimize the LP during forward resolution
    /// \param p_aParticle         a particle in simulation part to get back cuts
    /// \param p_linCut            cuts used for the PL   (Benders for the Bellman value at the end of the time step)
    /// \param p_state             Store the state, the particle number used in optimization and mesh number associated to the particle. As an input it contains the current state
    /// \param p_stateToStore      For backward resolution we need to store \f$ (S_t,A_{t-1},D_{t-1}) \f$  where p_state in output is \f$ (S_t,A_{t},D_{t}) \f$
    /// \param p_isimu           number of the simulation used
    double  oneStepForward(const Eigen::ArrayXd &p_aParticle,  Eigen::ArrayXd &p_state, Eigen::ArrayXd &p_stateToStore,
                           const libflow::SDDPCutOptBase &p_linCut,
                           const int &p_isimu) const
    {
        // optimizer constraints
        AddConstraintReservoir constraints;
        // suppose that at m_date = 0 we we on an average scenario
        Eigen::ArrayXd inflows(m_nbStorage);
        double demand ;
        if (libflow::isLesserOrEqual(m_date, 0.))
        {
            inflows.setConstant(m_InflowAver);
            demand = m_DAverage ;
        }
        else
        {
            // store new inflows and demand : cut such that it is positive
            for (int isto = 0; isto < m_nbStorage; ++isto)
                inflows(isto) = std::max(m_InflowAver + m_sigF * m_simulatorForward->getGaussian(isto, p_isimu), 0.);
            demand  = std::max(m_DAverage + m_sigD * m_simulatorForward->getGaussian(m_nbStorage, p_isimu), 0.);
        }
        Eigen::ArrayXd stateFollowing(m_nbStorage);
        Eigen::ArrayXd  valueAndDerivatives(1 + m_nbStorage);
        double cost ;
        // solve
        createAndSolveLP(p_linCut, p_state, constraints, m_spot, valueAndDerivatives, inflows, demand,   stateFollowing, cost);
        // keep state
        p_state = stateFollowing;
        p_stateToStore = stateFollowing;
        return cost;
    }

    /// brief update the optimizer for new date
    void updateDates(const double &p_date, const double &p_dateNext)
    {
        m_date = p_date ;
        if (libflow::isLesserOrEqual(0., p_date))
        {
            m_InflowAver = m_timeInflowAver->get(m_date);
            m_DAverage = m_timeDAverage->get(m_date);
            m_spot = m_timeSpot->get(m_date);
        }
        m_InflowAverNext = m_timeInflowAver->get(p_dateNext);
        m_DAverageNext = m_timeDAverage->get(p_dateNext);
        m_spotNext = m_timeSpot->get(p_dateNext);
    }

    /// \brief Get an admissible state for a given date
    /// \param p_date   current date
    /// \return an admissible state
    Eigen::ArrayXd oneAdmissibleState(const double   &p_date)
    {
        Eigen::ArrayXd toRet = Eigen::ArrayXd::Constant(m_nbStorage, m_initialLevel);
        return toRet;
    }


    /// \brief get back state size : number of storage + number of inflows + one for demand
    inline int getStateSize() const
    {
        return m_nbStorage;
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
#endif
