
#ifndef INITIALVALUEDIST_H
#define INITIALVALUEDIST_H
#include <functional>
#include <memory>
#include <boost/mpi.hpp>
#include <Eigen/Dense>
#include "reflow/core/grids/FullGrid.h"

/** \file InitialValueDist.h
 *  \brief permits to affect a final value to a regression problem
 *         distributing data and calculations.
 * \author Xavier Warin
 * \todo Develop MPI for regimes too
 */

namespace reflow
{
///\class InitialValueDist InitialValueDist.h
///       Last time step in dynamic programming : set the final values
class InitialValueDist
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
    InitialValueDist(const  std::shared_ptr<FullGrid> &p_pGridCurrent, const int &p_nbRegime, const Eigen::Array< bool, Eigen::Dynamic, 1>   &p_bdimToSplit,
                     const boost::mpi::communicator &p_world);

    ///\brief Fill in array with values
    /// \param p_funcValue    function giving the final value  (arguments are  the state :  regime, point coordinates )
    /// \return values on the grid
    std::vector<std::shared_ptr< Eigen::ArrayXd > >  operator()(const std::function<double(const int &, const Eigen::ArrayXd &)>   &p_funcValue) const;

    /// \brief get back local grid associated to current step
    inline std::shared_ptr<FullGrid>   getGridCurrentProc()const
    {
        return m_gridCurrentProc ;
    }
};

}
#endif /*  FINALSTEPREGRESSIONDP_H */
