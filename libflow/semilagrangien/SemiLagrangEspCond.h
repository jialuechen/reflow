// Copyright (C) 2016 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#ifndef SEMILAGRANGESPCOND_H
#define SEMILAGRANGESPCOND_H
#include <Eigen/Dense>
#include <map>
#include <array>
#include <vector>
#include "libflow/core/utils/constant.h"
#include "libflow/core/grids/InterpolatorSpectral.h"

/** \file SemiLagrangEspCond.h
 *  \brief Semi Lagrangian method for process \f$ d x_t = b dt + \sigma dW_t \f$
 *  where \f$ X_t,  b \f$  with values in \f$ {\mathbb R}^n \f$ , \f$ \sigma \f$ a \f$ \mathbf{R}^n
 *  \times  \mathbf{R}^m \f$  matrix and \f$ W_t \f$ with values in \f$ \mathbf{R}^m \f$
 */

namespace libflow
{

/// \class  SemiLagrangEspCond SemiLagrangEspCond.h
/// calculate semi Lagrangian operator for previously defined process.
class SemiLagrangEspCond
{
    ///\brief interpolator
    std::shared_ptr<InterpolatorSpectral> m_interpolator;

    /// \brief store extremal values for the grid (min, max coordinates in each dimension)
    std::vector <std::array< double, 2>  >  m_extremalValues;

    /// \brief Do we use modification of volatility  to stay in the domain
    bool m_bModifVol ;

public :

    /// \brief Constructor
    /// \param  p_interpolator    Interpolator storing the grid
    /// \param  p_extremalValues  Extremal values of the grid
    /// \param  p_bModifVol        do we modify volatility to stay in the domain.
    ///                            If activated, when not modification of volatility give a point inside the domain, truncation is achieved
    SemiLagrangEspCond(const std::shared_ptr<InterpolatorSpectral> &p_interpolator, const std::vector <std::array< double, 2>  > &p_extremalValues, const bool &p_bModifVol);

    /// \brief Calculate \f$ \frac{1}{2d} \sum_{i=1}^d \phi(x+ b dt + \sigma_i \sqrt{dt})+  \phi(x+ b dt - \sigma_i \sqrt{dt} \f$
    ///        where \f$ \sigma_i \f$ is column \f$ i\f$ of \f$ \sigma \f$
    /// \param p_x                 beginning point
    /// \param p_b                 trend
    /// \param p_sig               volatility matrix
    /// \param  p_dt               Time step size
    /// \return  (the value calculated,true) if point inside the domain, otherwise (0., false)
    std::pair<double, bool>  oneStep(const Eigen::ArrayXd   &p_x, const Eigen::ArrayXd &p_b, const Eigen::ArrayXXd &p_sig, const double &p_dt) const;


};
}
#endif
