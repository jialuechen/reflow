
#ifndef FINALSTEPDPCUT_H
#define FINALSTEPDPCUT_H
#include <functional>
#include <memory>
#include <Eigen/Dense>
#include "reflow/core/grids/SpaceGrid.h"

/** \file FinalStepDP.h
 *  \brief permits to affect some cut  values to a  problem. Developped for problems with cuts.
 * \author Xavier Warin
 * \todo Developp MPI for regimes too
 */

namespace reflow
{
///\class FinalStepDPCut FinalStepDPCut.h
///       Last time step in dynamic programming  with cuts: set the final cuts values.
class FinalStepDPCut
{
private :

    std::shared_ptr<SpaceGrid>  m_pGridCurrent; ///< grid at final time step
    int m_nDim ; ///< Dimension of the grid
    int m_nbRegime ; ///< Number of regimes

public :

    /// \brief Constructor with general grids
    /// \param p_pGridCurrent   grid describing the whole problem
    /// \param p_nbRegime       numbers of regime treated
    FinalStepDPCut(const  std::shared_ptr<SpaceGrid> &p_pGridCurrent, const int &p_nbRegime);


    ///\brief Fill in array with values
    /// \param p_funcValue    function giving the final cut values for each regime (arguments are  the state :  regime number, point coordinates ,  array of simulations corresponding to stochastic non controlled state)
    /// \param p_particles     simulations at final date (First dimension  : size of the stochastic non controlled state,  second dimension : the  number of particles)
    /// \return cuts values on the grid : for each regime (number of simulations * nb cuts) by number of stock points
    std::vector<std::shared_ptr< Eigen::ArrayXXd > >  operator()(const std::function< Eigen::ArrayXd(const int &, const Eigen::ArrayXd &, const Eigen::ArrayXd &)>      &p_funcValue,
            const Eigen::ArrayXXd &p_particles) const;

};
}
#endif /*  FINALSTEPDPCUT_H */
