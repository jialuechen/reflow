
#ifndef TRANSITIONSTEPSEMILAGRANGBASE_H
#define  TRANSITIONSTEPSEMILAGRANGBASE_H
#include <vector>
#include <memory>
#include <functional>
#include <Eigen/Dense>

/** \file TransitionStepSemilagrangBase.h
 * \brief Solve one  step in optimization of an explicit semi Lagrangian scheme
 * \author Xavier Warin
 */
namespace reflow
{
/// \class TransitionStepSemilagrangBase  TransitionStepSemilagrangBase.h
///        One step of semi Lagrangian scheme
class TransitionStepSemilagrangBase
{

    /// \brief One time step for resolution
    /// \param p_phiIn         for each regime the function value ( on the grid)
    /// \param p_time          current date
    /// \param p_boundaryFunc  Function at the boundary to impose Dirichlet conditions  (depending on regime and position)
    /// \return     solution obtained after one step of dynamic programming and the optimal control
    virtual std::pair< std::vector< std::shared_ptr< Eigen::ArrayXd > >, std::vector< std::shared_ptr< Eigen::ArrayXd >  >  > oneStep(const std::vector< std::shared_ptr< Eigen::ArrayXd > > &p_phiIn, const double &p_time,   const std::function<double(const int &, const Eigen::ArrayXd &)> &p_boundaryFunc) const = 0 ;
};
}

#endif
