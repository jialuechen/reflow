
#ifndef INITIALVALUE_H
#define INITIALVALUE_H
#include <functional>
#include <memory>
#include <Eigen/Dense>
#include "libflow/core/grids/SpaceGrid.h"

/** \file InitialValue.h
 *  \brief permits to affect an initial value for the resolution of a PDE by Semi Lagrangian methods
 * \author Xavier Warin
 */

namespace libflow
{
///\class InitialValue InitialValue.h
///       Last time step in dynamic programming : set the final values. Thread used.
class InitialValue
{
private :

    std::shared_ptr<SpaceGrid>  m_pGridCurrent; ///< grid at final time step
    int m_nDim ; ///< Dimension of the grid
    int m_nbRegime ; ///< Number of regimes

public :

    /// \brief Constructor
    /// \param p_pGridCurrent   grid describing the whole problem
    /// \param p_nbRegime       numbers of regime treated
    InitialValue(const  std::shared_ptr<SpaceGrid> &p_pGridCurrent, const int &p_nbRegime);

    ///\brief Fill in array with values
    /// \param p_funcValue    function giving the initial value for each regime (arguments are  the state :  regime number, coordinate of the point)
    /// \return values on the grid
    std::vector<std::shared_ptr< Eigen::ArrayXd > >  operator()(const std::function<double(const int &, const Eigen::ArrayXd &)>     &p_funcValue) const;

};
}
#endif /*  INITIALVALUE_H */
