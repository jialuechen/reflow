
#ifndef OPTIMIZEGASTORAGESDDP_H
#define OPTIMIZEGASTORAGESDDP_H
#include "ClpSimplex.hpp"
#include <boost/lexical_cast.hpp>
#include <Eigen/Dense>
#include "reflow/sddp/SDDPCutOptBase.h"
#include "reflow/sddp/OptimizerSDDPBase.h"
#include "reflow/core/utils/constant.h"

/** \file OptimizeGasStorageSDDP.h
 * \brief Solve a fictitious N stock problem connecting N gas storage
 *        This is the function used in SDDP to optimize a transition problem between two time steps
 * \author Xavier Warin
 */


/// \class AddConstraintNStock OptimizeGasStorageSDDP.h
/// Add constraints to the Bellman value with cuts
class AddConstraintNStock
{
protected:

    /// \brief calculate cuts
    /// \param  p_linCut         cuts stored
    virtual inline 	 Eigen::ArrayXXd calCuts(const reflow::SDDPCutOptBase &p_linCut) const
    {
        return  Eigen::ArrayXXd() ;
    }

public:
    /// \brief  add constraints to Bellman value
    /// \param  p_linCut         cuts stored
    /// \param  p_nbStorage      number of storage
    /// \param  p_rows           rows for matrix contraints
    /// \param  p_columns        columns for matrix contraints
    /// \param  p_elements       A matrix elements
    /// \param  p_lowBoundConst  lower constraint \f$ lc\f$  on matrix \f$ lc \le A x \f$
    /// \param  p_upperBoundConst upper constraint \f$ uc\f$  on matrix \f$ A x \le uc \f$

    void addConstraints(const reflow::SDDPCutOptBase &p_linCut, int p_nbStorage,  Eigen::ArrayXi &p_rows,   Eigen::ArrayXi   &p_columns,  Eigen::ArrayXd   &p_elements,
                        Eigen::ArrayXd    &p_lowBoundConst,  Eigen::ArrayXd   &p_upperBoundConst) const
    {
        // get back cuts
        Eigen::ArrayXXd  cuts = calCuts(p_linCut);
        int iBellPos = p_nbStorage * 3;
        int idecToStock = 2 * p_nbStorage;
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
                double deriv =  std::max(cuts(1 + isto, icut), 0.);
                p_elements(ipos + isto + 1) = -deriv;
            }
            p_lowBoundConst(ibound + icut) = - reflow::infty;
            p_upperBoundConst(ibound + icut) = affineValue;
        }
    }
};

/// \class AddConstraintOptimizerNstock OptimizeGasStorageSDDP.h
/// defines constraint using cuts in optimization part
class AddConstraintOptimizerNstock : public AddConstraintNStock
{
private:
    /// \brief simulation number
    int m_isim ;

    /// \brief calculate cuts
    /// \param  p_linCut         cuts stored
    virtual inline 	 Eigen::ArrayXXd calCuts(const reflow::SDDPCutOptBase &p_linCut) const
    {
        return  p_linCut.getCutsAssociatedToTheParticle(m_isim);
    }

public :

    /// \brief Constructor
    /// \param p_isim  simulation number used in optimization part
    AddConstraintOptimizerNstock(const int   &p_isim): m_isim(p_isim) {};

};

/// \class AddConstraintSimulatorNstock OptimizeGasStorageSDDP.h
/// defines constraint using cuts in simulation part
class AddConstraintSimulatorNstock: public AddConstraintNStock
{
private:

    /// \brief Uncertainty used in regression
    Eigen::ArrayXd m_alea;

    /// \brief calculate cuts
    /// \param  p_linCut         cuts stored
    virtual inline 	 Eigen::ArrayXXd calCuts(const reflow::SDDPCutOptBase &p_linCut) const
    {
        return  p_linCut.getCutsAssociatedToAParticle(m_alea);
    }

public :

    /// \brief Constructor
    /// \param p_alea   uncertainty obtained during simulation part
    AddConstraintSimulatorNstock(const Eigen::ArrayXd   &p_alea): m_alea(p_alea) {};
};


/// \class OptimizeGasStorageSDDP OptimizeGasStorageSDDP.h
/// The N gas storage are similar and connected, such that on can transfer gas from one storage to another one.
///  The gain is only obtained by sell gas on the market, so the value of the $N$ storage is equal to the value of a single storage multiplied by N
/// No constraints on the storage at the end of optimization period (so the storage will be empty)
/// - when injecting the gain is  \f$ - C_{inj} ( S+ \kappa_{inj} )\f$
/// - when withdrawing the gain is  \f$  C_{with} ( S- \kappa_{with} )\f$
/// .
template< class Simulator>
class OptimizeGasStorageSDDP : public reflow::OptimizerSDDPBase
{
private :

    /// \brief Physical constraints for equivalent storage
    //@{
    double m_maxLevel; ///< storage level
    double m_injectionRate ; ///< injection  capacity (volume) per time step \f$ C_{inj} \f$
    double m_withdrawalRate ; ///< withdrawal rate (volume) per time step \f$ C_{with} \f$
    double m_injectionCost; ///< injection cost \f$ \kappa_{inj} \f$ per volume unit
    double m_withdrawalCost ; /// withdrawal cost  \f$ \kappa_{with} \f$ per volume unit
    //@}
    int m_nbStorage ; ///< Number of storage
    std::shared_ptr< Simulator> m_simulatorBackward ; // for backward simulations
    std::shared_ptr< Simulator> m_simulatorForward ; // for forward simulations


    /// \brief LP creation
    /// \param p_linCut               cuts used for the PL   (Benders for the Bellman value at the end of the time step)
    /// \param p_stockLevel           Store the stock level associated to the state
    /// \param p_payInjection         cost for injection (>0)
    /// \param payWithdrawal          gain for withdrawal (>0)
    /// \param p_contraints           associated to the cuts
    /// \param p_valueAndDerivatives  optimal value of the function and derivatives
    /// \param p_stateFollowing       To store state after optimal command
    /// \param p_gain                 instantaneous gain associated to the optimal strategy
    template< class TConstraint>
    void createAndSolveLP(const reflow::SDDPCutOptBase   &p_linCut, const Eigen::ArrayXd &p_stockLevel,
                          const Eigen::ArrayXd &p_payInjection, const Eigen::ArrayXd   &p_payWithdrawal, const TConstraint &p_constraints, Eigen::ArrayXd &p_valueAndDerivatives,
                          Eigen::ArrayXd &p_stateFollowing, double &p_gain) const
    {
        // optimizer matrix constraints
        Eigen::ArrayXi rows(3 * m_nbStorage); // row position
        Eigen::ArrayXi columns(3 * m_nbStorage) ; // columns  position
        Eigen::ArrayXd elements(3 * m_nbStorage) ; // constraints matrix values
        // bounds associated to matrix constraints
        Eigen::ArrayXd lowBoundConst(m_nbStorage);
        Eigen::ArrayXd upperBoundConst(m_nbStorage) ;

        // bounds on values
        Eigen::ArrayXd lowBound(3 * m_nbStorage + 1);
        Eigen::ArrayXd upperBound(3 * m_nbStorage + 1);
        for (int isto = 0 ; isto < m_nbStorage; ++isto)
        {
            // injecion
            lowBound(isto) = 0;
            upperBound(isto) = m_injectionRate;
            // withdrawal
            lowBound(isto + m_nbStorage) = - m_withdrawalRate;
            upperBound(isto + m_nbStorage) = 0;
            // storage bounds
            lowBound(isto + 2 * m_nbStorage) = 0;
            upperBound(isto + 2 * m_nbStorage) = m_maxLevel;
        }
        // for  fictitious data for bellman
        lowBound(3 * m_nbStorage) = - reflow::infty;
        upperBound(3 * m_nbStorage) = reflow::infty;
        // objective function
        Eigen::ArrayXd objFunc = Eigen::ArrayXd::Zero(3 * m_nbStorage + 1);
        for (int isto = 0 ; isto < m_nbStorage ; ++isto)
        {
            objFunc(isto) = -p_payInjection(isto);
            objFunc(isto + m_nbStorage) = -p_payWithdrawal(isto);
        }
        objFunc(3 * m_nbStorage) = 1;

        // flow constraints
        int idecToFinal = 2 * m_nbStorage;
        for (int isto = 0 ; isto < m_nbStorage  ; ++isto)
        {
            int ipos = 3 * isto;
            rows(ipos) =  isto;
            columns(ipos) = isto;
            elements(ipos) = -1;
            rows(ipos + 1) =  isto;
            columns(ipos + 1) = isto + m_nbStorage;
            elements(ipos + 1) = -1;
            rows(ipos + 2) = isto;
            columns(ipos + 2) = isto + idecToFinal;
            elements(ipos + 2) =  1;
            lowBoundConst(isto) = p_stockLevel(isto);
            upperBoundConst(isto) = p_stockLevel(isto);
        }

        //  add cuts for bellman value to constraints
        p_constraints.addConstraints(p_linCut, m_nbStorage, rows, columns, elements, lowBoundConst, upperBoundConst);

        //  model
        ClpSimplex  model;

#ifdef NDEBUG
        model.setLogLevel(0);
#endif

        model.loadProblem(CoinPackedMatrix(false, rows.data(), columns.data(), elements.data(), elements.size()), lowBound.data(), upperBound.data(), objFunc.data(), lowBoundConst.data(), upperBoundConst.data());
        model.setOptimizationDirection(-1) ; // maximize

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
            p_stateFollowing(isto) = p_stockLevel(isto) + columnPrimal[isto] + columnPrimal[isto + m_nbStorage];
        }
        // gain
        p_gain = 0. ;
        for (int isto = 0 ; isto < m_nbStorage  ; ++isto)
        {
            p_gain += - p_payInjection(isto) * columnPrimal[isto] - p_payWithdrawal(isto) * columnPrimal[isto + m_nbStorage];
        }
    }

public :

    /// \brief Constructor
    /// \param  p_maxLevel          size of the storage
    /// \param  p_injectionRate     injection rate per time step
    /// \param  p_withdrawalRate    withdrawal rate between two time steps
    /// \param  p_injectionCost     injection cost
    /// \param  p_withdrawalCost    withdrawal cost
    /// \param  p_nbStorage        number of storage
    /// \param   p_simulatorBackward   backward  simulator
    /// \param   p_simulatorForward    Forward simulator
    OptimizeGasStorageSDDP(const double &p_maxLevel, const double   &p_injectionRate, const double &p_withdrawalRate,
                           const double &p_injectionCost, const double &p_withdrawalCost, const int &p_nbStorage,
                           const std::shared_ptr<Simulator> &p_simulatorBackward,
                           const std::shared_ptr<Simulator> &p_simulatorForward) :
        m_maxLevel(p_maxLevel), m_injectionRate(p_injectionRate), m_withdrawalRate(p_withdrawalRate), m_injectionCost(p_injectionCost), m_withdrawalCost(p_withdrawalCost),
        m_nbStorage(p_nbStorage), m_simulatorBackward(p_simulatorBackward), m_simulatorForward(p_simulatorForward) {}




    /// \brief Optimize the LP during backward resolution
    /// \param p_linCut            cuts used for the PL   (Benders for the Bellman value at the end of the time step)
    /// \param p_aState            Store the state, the particle number used in optimization (if conditional cuts) and mesh number associated to the particle
    /// \param p_particle          The particle n dimensional value associated to the regression
    /// \param p_isample            sample number for independent uncertainties
    /// \return  a vector with the optimal value and the derivatives if the function value with respect to each state (here the stocks)
    Eigen::ArrayXd oneStepBackward(const reflow::SDDPCutOptBase &p_linCut, const std::tuple< std::shared_ptr<Eigen::ArrayXd>, int, int > &p_aState,
                                   const Eigen::ArrayXd &p_particle, const int &p_isample) const
    {
        // constraints
        AddConstraintOptimizerNstock constraints(std::get<1>(p_aState));
        // spot value
        double spotPrice = m_simulatorBackward->fromOneParticleToSpot(p_particle);
        // payoff
        Eigen::ArrayXd payInjection = Eigen::ArrayXd::Constant(m_nbStorage, spotPrice  + m_injectionCost);
        Eigen::ArrayXd payWithdrawal = Eigen::ArrayXd::Constant(m_nbStorage, spotPrice - m_withdrawalCost);
        // Creation  and PL resolution
        Eigen::ArrayXd stateFollowing(m_nbStorage);
        Eigen::ArrayXd  valueAndDerivatives(1 + m_nbStorage);
        double gain ;
        createAndSolveLP(p_linCut, *std::get<0>(p_aState), payInjection, payWithdrawal, constraints, valueAndDerivatives, stateFollowing, gain);
        return valueAndDerivatives;
    }


    /// \brief Optimize the LP during forward resolution
    /// \param p_aParticle         a particle in simulation part to get back conditional cuts
    /// \param p_state             Store the state, the particle number used in optimization (if conditional cuts) and mesh number associated to the particle. As an input it constraints the current state
    /// \param p_stateToStore      For backward resolution we need to store \f$ (S_t,A_{t-1},D_{t-1}) \f$  where p_state in output is \f$ (S_t,A_{t},D_{t}) \f$
    /// \param p_linCut            cuts used for the PL   (Benders for the Bellman value at the end of the time step)
    double  oneStepForward(const Eigen::ArrayXd &p_aParticle, Eigen::ArrayXd &p_state, Eigen::ArrayXd &p_stateToStore,
                           const reflow::SDDPCutOptBase &p_linCut, const int &) const
    {
        // optimizer constraints
        AddConstraintSimulatorNstock constraints(p_aParticle);

        // spot value
        double spotPrice = m_simulatorForward->fromOneParticleToSpot(p_aParticle);
        // payoff
        Eigen::ArrayXd payInjection = Eigen::ArrayXd::Constant(m_nbStorage, spotPrice + m_injectionCost);
        Eigen::ArrayXd payWithdrawal = Eigen::ArrayXd::Constant(m_nbStorage, spotPrice - m_withdrawalCost);
        // Creation  and PL resolution
        Eigen::ArrayXd stateFollowing(m_nbStorage);
        Eigen::ArrayXd  valueAndDerivatives(1 + m_nbStorage);
        double gain ;
        createAndSolveLP(p_linCut, p_state, payInjection, payWithdrawal, constraints, valueAndDerivatives, stateFollowing, gain);
        p_state = stateFollowing;
        p_stateToStore = stateFollowing;
        return gain ;
    }

    /// brief update the optimizer for new date
    void updateDates(const double &p_date, const double &p_dateNext) {}

    /// \brief Get an admissible state for a given date
    /// \param p_date   current date
    /// \return an admissible state
    Eigen::ArrayXd oneAdmissibleState(const double   &p_date)
    {
        Eigen::ArrayXd toRet = Eigen::ArrayXd::Constant(m_nbStorage, m_maxLevel * 0.5);
        return toRet;
    }


    /// \brief get back state size
    inline int getStateSize() const
    {
        return m_nbStorage;
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
