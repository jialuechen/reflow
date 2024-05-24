
#ifndef FINALSTEPDP_H
#define FINALSTEPDP_H
#include <functional>
#include <memory>
#include <Eigen/Dense>
#include "reflow/core/grids/SpaceGrid.h"

/** \file FinalStepDP.h
 *  \brief permits to affect a final value to a  problem
 * \author Xavier Warin
 * \todo Developp MPI for regimes too
 */

namespace reflow
{
///\class FinalStepDP FinalStepDP.h
///       Last time step in dynamic programming : set the final values.  used
class FinalStepDP
{
private :

    std::shared_ptr<SpaceGrid>  m_pGridCurrent; ///< grid at final time step
    int m_nDim ; ///< Dimension of the grid
    int m_nbRegime ; ///< Number of regimes

public :

    /// \brief Constructor with general grids
    /// \param p_pGridCurrent   grid describing the whole problem
    /// \param p_nbRegime       numbers of regime treated
    FinalStepDP(const  std::shared_ptr<SpaceGrid> &p_pGridCurrent, const int &p_nbRegime);


    ///\brief Fill in array with values
    /// \param p_funcValue    function giving the final value for each regime (arguments are  the state :  regime number, point coordinates ,  array of simulations corresponding to stochastic non controlled state)
    /// \param p_particles     simulations at final date (First dimension  : size of the stochastic non controlled state,  second dimension : the  number of particles)
    /// \return values on the grid : for each regime number of simulations by number of stock points
    std::vector<std::shared_ptr< Eigen::ArrayXXd > >  operator()(const std::function<double(const int &, const Eigen::ArrayXd &, const Eigen::ArrayXd &)>      &p_funcValue,
            const Eigen::ArrayXXd &p_particles) const;

};
}
#endif /*  FINALSTEPDP_H */
