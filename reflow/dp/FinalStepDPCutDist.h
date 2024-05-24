
#ifndef FINALSTEPDPCUTDIST_H
#define FINALSTEPDPCUTDIST_H
#include <functional>
#include <memory>
#include <boost/mpi.hpp>
#include <Eigen/Dense>
#include "reflow/core/grids/FullGrid.h"

/** \file FinalStepDPCutDist.h
 *  \brief permits to affect a final value to a  problem
 * \author Xavier Warin
 * \todo Develop MPI for regimes too
 */

namespace reflow
{
///\class FinalStepDPCutDist FinalStepDPCutDist.h
///       Last time step in dynamic programming : set the final values
class FinalStepDPCutDist
{
private :

    std::shared_ptr<FullGrid>  m_pGridCurrent; ///< grid at final time step
    std::shared_ptr<FullGrid>    m_gridCurrentProc ; ///< local grid  treated by the processor
    int m_nDim ; ///< Dimension of the grid
    int m_nbRegime ; ///< Number of regimes

public :

    /// \brief Constructor
    /// \param p_pGridCurrent   grid describing the whole problem
    /// \param p_nbRegime       numbers of regime treated
    /// \param p_bdimToSplit    Dimensions to split for parallelism
    /// \param p_world          MPI communicator
    FinalStepDPCutDist(const  std::shared_ptr<FullGrid> &p_pGridCurrent, const int &p_nbRegime, const Eigen::Array< bool, Eigen::Dynamic, 1>   &p_bdimToSplit,
                       const boost::mpi::communicator &p_world);

    ///\brief Fill in array with values
    /// \param p_funcValue    function giving the final cut values  (arguments are  the state :  regime, point coordinates ,  array of simulations corresponding to stochastic non controlled state)
    /// \param p_particles     simulations at final date (First dimension  : size of the stochastic non controlled state,  second dimension : the  number of particles)
    /// \return cuts values on the grid : for each regime (number of simulations * nb cuts) by number of stock points
    std::vector<std::shared_ptr< Eigen::ArrayXXd > >  operator()(const std::function< Eigen::ArrayXd(const int &, const Eigen::ArrayXd &, const Eigen::ArrayXd &)>     &p_funcValue,
            const Eigen::ArrayXXd &p_particles) const;

    /// \brief get back local grid associated to current step
    inline std::shared_ptr<FullGrid>   getGridCurrentProc()const
    {
        return m_gridCurrentProc ;
    }
};

}
#endif /*  FINALSTEPDPCUTDIST_H */
