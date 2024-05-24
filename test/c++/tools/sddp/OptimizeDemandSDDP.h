#ifndef OPTIMIZEDEMANDSDDP_H
#define OPTIMIZEDEMANDSDDP_H
#include "ClpSimplex.hpp"
#include  "reflow/core/utils/constant.h"
#include "reflow/sddp/SDDPCutBase.h"
#include "reflow/sddp/OptimizerSDDPBase.h"
#include "reflow/core/grids/OneDimRegularSpaceGrid.h"
#include "reflow/core/grids/OneDimData.h"
#include "reflow/core/utils/comparisonUtils.h"

/** \file OptimizeDemandSDDP.h
 * \brief  Suppose that we have  the demand follows an AR 1 model
 *        \f$ D^{n+1} = k (D^n-D) + \sigma_d  g + k D\f$ where \f$  g \f$ is a Gaussian noise
 *         Suppose that we want to satisfy the demand, buying energy at a given price $P$
 *        The value of the contract is
 *        \f{eqnarray*}{
 *         V & = & \mathcal{E}( \sum_{i=0}^N D_i) \\
 *           & = & (N+1)D_0 P
 *        \f}
 *        This problem can be solve by SDDP introducing a constraint of flow type on the dynamic of the demand
 * \author Xavier Warin
 */

/// \class OptimizeDemandSDDP OptimizeDemandSDDP.h
///
template< class Simulator>
class OptimizeDemandSDDP : public reflow::OptimizerSDDPBase
{

private :

    /// \brief AR 1 model for demand
    //@{
    double m_sigD ; /// volatility for demand \f$ \sigma_d \f$
    double m_kappaD ; /// AR coefficient for demand  \f$ k \f$
    std::shared_ptr<reflow::OneDimData<reflow::OneDimRegularSpaceGrid, double> > m_timeDAverage; /// store the average demand depending on time
    double m_DAverage ; ///< average value for demand  at current date
    double m_DAverageNext ; ///< average value for demand at next time  date
    //@}

    double m_spot ; ///< deterministic spot price
    std::shared_ptr< Simulator> m_simulatorBackward ; // for backward simulations
    std::shared_ptr< Simulator> m_simulatorForward ; // for forward simulations


    double m_date ; ///< current date
    double m_dateNext ; ///< next date (after current date)

    /// \brief LP creation
    /// \param p_linCut               cuts used for the PL   (Benders for the Bellman value at the end of the time step)
    /// \param p_stateLevel           Store the state  : storage levels, inflows levels,  demand level
    /// \param p_spot                 Spot use if the LP
    /// \param p_valueAndDerivatives  optimal value of the function and derivatives
    /// \param p_stateFollowing       To store state after optimal command ( demand)
    /// \param p_cost                 instantaneous cost
    void createAndSolveLP(const reflow::SDDPCutOptBase   &p_linCut, const Eigen::ArrayXd &p_stateLevel,
                          Eigen::ArrayXd &p_valueAndDerivatives,
                          double &p_stateFollowing, double &p_cost) const
    {
        // get back cuts
        // here we use particle 0 because cuts are independent of a particle number
        Eigen::ArrayXd aParticle;
        Eigen::ArrayXXd  cuts =  p_linCut.getCutsAssociatedToAParticle(aParticle);

        // optimizer matrix constraints
        Eigen::ArrayXi rows(2 * cuts.cols() + 1); // row position
        Eigen::ArrayXi columns(2 * cuts.cols() + 1) ; // columns  position
        Eigen::ArrayXd elements(2 * cuts.cols() + 1) ; // constraints matrix values
        // objective function
        Eigen::ArrayXd objFunc(2);
        // bounds on values
        Eigen::ArrayXd lowBound(2);
        Eigen::ArrayXd upperBound(2);
        // bounds associated to matrix constraints
        Eigen::ArrayXd lowBoundConst(cuts.cols() + 1);
        Eigen::ArrayXd upperBoundConst(cuts.cols() + 1) ;

        // facing the market, add bellman value
        objFunc(0) = m_spot ;
        objFunc(1) = 1;
        // for demand
        lowBound(0) = 0 ;
        upperBound(0) = reflow::infty;
        // for  fictitious data for bellman
        lowBound(1) = - reflow::infty;
        upperBound(1) = reflow::infty;
        // first constraint (flow constraint)
        rows(0) = 0 ;
        columns(0) = 0 ;
        elements(0) = 1. ;
        lowBoundConst(0) = p_stateFollowing;
        upperBoundConst(0) = p_stateFollowing;

        for (int icut = 0 ; icut < cuts.cols(); ++icut)
        {
            rows(2 * icut + 1) = icut + 1;
            columns(2 * icut + 1) = 1;
            elements(2 * icut + 1) = 1.;
            rows(2 * icut + 2) = icut + 1;
            columns(2 * icut + 2) = 0;
            elements(2 * icut + 2) = -cuts(1, icut);  // deriv demand
            lowBoundConst(icut + 1) = cuts(0, icut); // affine value
            upperBoundConst(icut + 1) = reflow::infty ;
        }
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
        // derivative for demand
        p_valueAndDerivatives(1) = dual[0] * m_kappaD;

        // primal
        double *columnPrimal = model.primalColumnSolution();
        // cost
        p_cost =  m_spot * columnPrimal[0];
    }

public :


    /// \brief Constructor for the demand problem
    /// \param   p_sigD                volatility for demand
    /// \param   p_kappaD              AR coefficient for demand
    /// \param   p_timeDAverage        average demand
    /// \param   p_spot                Spot price
    /// \param   p_simulatorBackward   backward  simulator
    /// \param   p_simulatorForward    Forward simulator
    OptimizeDemandSDDP(const  double   &p_sigD, const double &p_kappaD,
                       const std::shared_ptr<reflow::OneDimData<reflow::OneDimRegularSpaceGrid, double> > &p_timeDAverage,
                       const double &p_spot,
                       const std::shared_ptr<Simulator> &p_simulatorBackward,
                       const std::shared_ptr<Simulator> &p_simulatorForward): m_sigD(p_sigD), m_kappaD(p_kappaD), m_timeDAverage(p_timeDAverage),
        m_spot(p_spot), m_simulatorBackward(p_simulatorBackward), m_simulatorForward(p_simulatorForward)
    {}



    /// \brief Optimize the LP during backward resolution
    /// \param p_linCut            cuts used for the PL   (Benders for the Bellman value at the end of the time step)
    /// \param p_aState            Store the state, and 0, 0 values
    /// \param p_particle          Here no regression , so empty array
    /// \param p_isample            sample number for independent uncertainties
    /// \return  a vector with the optimal value and the derivatives if the function value with respect to each state (here the stocks)
    Eigen::ArrayXd oneStepBackward(const reflow::SDDPCutOptBase &p_linCut, const std::tuple< std::shared_ptr<Eigen::ArrayXd>, int, int > &p_aState,
                                   const Eigen::ArrayXd &p_particle, const int &p_isample) const
    {
        // Creation  and PL resolution : demand should stay positive
        double stateFollowing  = (*std::get<0>(p_aState))(0) ;
        if (reflow::isLesserOrEqual(0., m_date))
            stateFollowing = std::max(m_kappaD * (stateFollowing - m_DAverage) + m_DAverageNext + m_sigD * m_simulatorBackward->getGaussian(0, p_isample), 0.);
        Eigen::ArrayXd  valueAndDerivatives(2);
        double cost ;
        createAndSolveLP(p_linCut, *std::get<0>(p_aState), valueAndDerivatives, stateFollowing, cost);
        return valueAndDerivatives;
    }


    /// \brief Optimize the LP during forward resolution
    /// \param p_aParticle         a particle in simulation part to get back cuts
    /// \param p_linCut            cuts used for the PL   (Benders for the Bellman value at the end of the time step)
    /// \param p_state             Store the state, the particle number used in optimization and mesh number associated to the particle. As an input it contains the current state
    /// \param p_stateToStore      For backward resolution we need to store \f$ (S_t,A_{t-1},D_{t-1}) \f$  where p_state in output is \f$ (S_t,A_{t},D_{t}) \f$
    /// \param p_isimu             number of the simulation used
    double  oneStepForward(const Eigen::ArrayXd &p_aParticle, Eigen::ArrayXd &p_state,  Eigen::ArrayXd &p_stateToStore,
                           const reflow::SDDPCutOptBase &p_linCut,
                           const int &p_isimu) const
    {
        // Creation  and PL resolution
        double stateFollowing = p_state(0);
        Eigen::ArrayXd  valueAndDerivatives(2);
        double cost ;
        createAndSolveLP(p_linCut, p_state, valueAndDerivatives, stateFollowing, cost);
        // go forward for uncertainties (demand should stay positive)
        p_stateToStore  = p_state ;
        p_state(0) =  std::max(m_kappaD * (p_state(0) - m_DAverage) + m_DAverageNext + m_sigD * m_simulatorForward->getGaussian(0, p_isimu), 0.);
        return cost ;
    }

    /// brief update the optimizer for new date
    void updateDates(const double &p_date, const double &p_dateNext)
    {
        m_date = p_date ;
        if (reflow::isLesserOrEqual(0., p_date))
            m_DAverage = m_timeDAverage->get(m_date);
        m_dateNext = p_dateNext ;
        m_DAverageNext = m_timeDAverage->get(m_dateNext);

    }

    /// \brief Get an admissible state for a given date
    /// \param p_date   current date
    /// \return an admissible state
    Eigen::ArrayXd oneAdmissibleState(const double   &p_date)
    {
        Eigen::ArrayXd toRet(1);
        toRet(0) = m_timeDAverage->get(0); // bad estimation for test purpose
        return toRet;
    }


    /// \brief get back state size : number of storage + number of inflows + one for demand
    inline int getStateSize() const
    {
        return 1;
    }

    /// \brief get the backward simulator back
    std::shared_ptr< reflow::SimulatorSDDPBase > getSimulatorBackward() const
    {
        return m_simulatorBackward ;
    }

    /// \brief get the forward simulator back
    std::shared_ptr< reflow::SimulatorSDDPBase > getSimulatorForward() const
    {
        return m_simulatorForward ;
    }

};
#endif
