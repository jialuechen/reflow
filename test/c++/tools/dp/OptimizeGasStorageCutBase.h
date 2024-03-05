
#ifndef OPTIMIZEGASSTORAGETCUTBASE_H
#define OPTIMIZEGASSTORAGETCUTBASE_H
#include "ClpSimplex.hpp"
#include <memory>
#include <Eigen/Dense>
#include "libflow/core/grids/SpaceGrid.h"

/** \file OptimizeGasStorageCutBase.h
 *  \brief  Simple example of a gas storage optimizer
 *          - injection rate, withdrawal rates are independent of the storage level
 *          - the size of the storage is constant, minimum gas level is 0
 *          .
 *          Designed to work in parallel/multi threaded framework
 *          The local  optimization is solved using a LP on one time step
 *  \author Xavier Warin
 */

/// \class OptimizeGasStorageCutBase OptimizeGasStorageCutBase.h
/// Defines a simple gas storage for optimization and simulation using cuts and a LP solver
/// No constraints on the storage at the end of optimization period (so the storage will be empty)
/// - when injecting the gain is  \f$ - C_{inj} ( S+ \kappa_{inj} )\f$
/// - when withdrawing the gain is  \f$  C_{with} ( S- \kappa_{with} )\f$
/// .
class OptimizeGasStorageCutBase
{
protected :
    /// \brief Physical constraints
    //@{
    double m_injectionRate ; ///< injection  capacity (volume) per time step \f$ C_{inj} \f$
    double m_withdrawalRate ; ///< withdrawal rate (volume) per time step \f$ C_{with} \f$
    double m_injectionCost; ///< injection cost \f$ \kappa_{inj} \f$ per volume unit
    double m_withdrawalCost ; /// withdrawal cost  \f$ \kappa_{with} \f$ per volume unit
    //@}

public :

    /// \brief Constructor
    /// \param  p_injectionRate     injection rate per time step
    /// \param  p_withdrawalRate    withdrawal rate between two time steps
    /// \param  p_injectionCost     injection cost
    /// \param  p_withdrawalCost    withdrawal cost
    OptimizeGasStorageCutBase(const double   &p_injectionRate, const double &p_withdrawalRate,
                              const double &p_injectionCost, const double &p_withdrawalCost):
        m_injectionRate(p_injectionRate), m_withdrawalRate(p_withdrawalRate), m_injectionCost(p_injectionCost), m_withdrawalCost(p_withdrawalCost) {}

    ///  \brief LP creation
    ///  \param p_cuts           array of cuts for current  for current stock points
    ///  \param p_stock         currentstock level
    ///  \param p_grid          grid for storage
    ///  \param p_spot          spot price
    ///  \param p_valueAndDerivatives  function value and derivative for current simulation
    ///  \param p_stateFollowing storage reached
    ///  \param p_gain           gain reached
    void   createAndSolveLP(const Eigen::ArrayXXd &p_cuts,
                            const double   &p_stock,
                            const   std::shared_ptr< libflow::SpaceGrid> &p_grid,
                            const double &p_spot,
                            Eigen::ArrayXd &p_valueAndDerivatives,
                            Eigen::ArrayXd &p_stateFollowing,
                            double &p_gain) const
    {
        // bound for storage
        std::vector <std::array< double, 2>  > extremVal = p_grid->getExtremeValues();
        Eigen::ArrayXi rows(3 + 2 * p_cuts.cols()); // row position
        Eigen::ArrayXi columns(3 + 2 * p_cuts.cols()) ; // columns  position
        Eigen::ArrayXd elements(3 + 2 * p_cuts.cols()) ; // constraints matrix values
        // bounds on values
        Eigen::ArrayXd lowBound(4 + p_cuts.cols());
        Eigen::ArrayXd upperBound(4 + p_cuts.cols());

        // injection
        lowBound(0) = 0;
        upperBound(0) = m_injectionRate;
        // withdrawal
        lowBound(1) = - m_withdrawalRate;
        upperBound(1) = 0. ;
        // bound on storage
        lowBound(2) = extremVal[0][0];
        upperBound(2) = extremVal[0][1];
        // for Bellman cuts
        lowBound(3) = - libflow::infty;
        upperBound(3) =  libflow::infty;
        // objective function
        Eigen::ArrayXd objFunc = Eigen::ArrayXd::Zero(4);
        objFunc(0) = -(p_spot + m_injectionCost);
        objFunc(1) = -(p_spot - m_withdrawalCost);
        objFunc(3) = 1.;
        // flow constraints
        rows(0) = 0;
        columns(0) = 0;
        elements(0) = -1;
        rows(1) = 0;
        columns(1) = 1;
        elements(1) = -1;
        rows(2) = 0;
        columns(2) = 2;
        elements(2) = 1;
        // bounds associated to matrix constraints
        Eigen::ArrayXd lowBoundConst(1 + p_cuts.cols());
        Eigen::ArrayXd upperBoundConst(1 + p_cuts.cols());
        lowBoundConst(0) = p_stock;
        upperBoundConst(0) = p_stock;

        // add cuts
        for (int icut = 0; icut < p_cuts.cols(); ++icut)
        {
            double affineValue = p_cuts(0, icut);
            int ipos = 3 + 2 * icut;
            rows(ipos) =  1 + icut;
            columns(ipos) = 3;
            elements(ipos) = 1;
            rows(ipos + 1) = 1 + icut;
            columns(ipos + 1) = 2;
            double deriv =  std::max(p_cuts(1, icut), 0.);
            elements(ipos + 1) = -deriv ;
            lowBoundConst(1 + icut) =  -libflow::infty;
            upperBoundConst(1 + icut) = affineValue ;
        }
        //  model
        ClpSimplex  model;
        //#ifdef NDEBUG
        model.setLogLevel(0);
        //#endif

        model.loadProblem(CoinPackedMatrix(false, rows.data(), columns.data(), elements.data(), elements.size()), lowBound.data(), upperBound.data(), objFunc.data(), lowBoundConst.data(), upperBoundConst.data());

        model.setOptimizationDirection(-1) ; // maximize

        ClpSolve solvectl;
        //solvectl.setSolveType(ClpSolve::usePrimal);
        solvectl.setSolveType(ClpSolve::useDual);
        solvectl.setPresolveType(ClpSolve::presolveOn);
        model.initialSolve(solvectl);

        bool modelSolved = model.isProvenOptimal();


        // optimal values
        p_valueAndDerivatives(0) = model.objectiveValue();

        // duals
        double *dual = model.dualRowSolution();
        p_valueAndDerivatives(1) = dual[0];

        // primal
        double *columnPrimal = model.primalColumnSolution();
        // for each stock
        p_stateFollowing(0) = p_stock + columnPrimal[0] + columnPrimal[1];

        // gain
        p_gain =  - (p_spot + m_injectionCost) * columnPrimal[0] - (p_spot - m_withdrawalCost) * columnPrimal[1];

        if (!modelSolved)
        {
            std::cout << "[problemLP::solveModel] : Warning Linear program could not be solved optimally somehow\n";
            abort();
        }
    }

}
;
#endif /* OPTIMIZEGASSTORAGECUTBASE_H */
