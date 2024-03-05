// Copyright (C) 2016 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#ifndef   FINALVALUEFUNCTION_H
#define   FINALVALUEFUNCTION_H
#include <Eigen/Dense>
#include <iostream>
#include "libflow/core/utils/comparisonUtils.h"


/** \file FinalValueFunction.h
 *  \brief Final value function for swings
 */
/// \class FinalValueFunction FinalValueFunction.h
///  final function payoff for swing
template< class PayOff >
class FinalValueFunction
{
private :

    PayOff m_pay ;
    int m_nExerc;

public :
    /// \brief Constructor
    FinalValueFunction(const PayOff &p_pay, const int &p_nExerc): m_pay(p_pay), m_nExerc(p_nExerc) {}


/// \brief final function en optimization
/// \param  p_stock  position in the stock
/// \param  p_state  position in the stochastic state
    inline double operator()(const int &, const Eigen::ArrayXd &p_stock, const Eigen::ArrayXd   &p_state) const
    {
        if (libflow::almostEqual(p_stock(0), static_cast<double>(m_nExerc), 10))
        {
            return 0.;
        }
        else
        {
            return  m_pay.apply(p_state.matrix());
        }
    }
};




#endif /* FINALVALUEFUNCTION_H */
