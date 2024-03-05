
#ifndef EUROPEANOPTIONS_H
#define EUROPEANOPTIONS_H
#include "test/c++/tools/NormalCumulativeDistribution.h"


/** \file EuropeanOptions.h
 *  \brief Calculate Black Scholes values for put and call
 *  \author Xavier Warin
 */


/// \class CallOption EuropeanOptions.h
/// Classical call option
class CallOption
{
public :
    /// \brief Constructor
    CallOption() {}

/// \brief value
///  \param p_S        asset value
///  \param p_sigma   volatility
///  \param p_r       interest rate
///  \param p_strike  strike
///  \param p_mat     maturity
///  \return option value
    double operator()(const double &p_S, const double &p_sigma, const double &p_r, const double   &p_strike, const double &p_mat) const
    {
        double d1 = (log(p_S / p_strike) + (p_r + 0.5 * p_sigma * p_sigma) * p_mat) / (p_sigma * sqrt(p_mat));
        double  d2 = d1 - p_sigma * sqrt(p_mat);
        return p_S * NormalCumulativeDistribution()(d1) - p_strike * exp(-p_r * p_mat) * NormalCumulativeDistribution()(d2);
    }

};

/// \class PutOption EuropeanOptions.h
/// Classical put option
class PutOption
{
public :
    /// \brief Constructor
    PutOption() {}

/// \brief value
///  \param p_S        asset value
///  \param p_sigma   volatility
///  \param p_r       interest rate
///  \param p_strike  strike
///  \param p_mat     maturity
///  \return option value
    double operator()(const double &p_S, const double &p_sigma, const double &p_r, const double   &p_strike, const double &p_mat) const
    {
        double d1 = (log(p_S / p_strike) + (p_r + 0.5 * p_sigma * p_sigma) * p_mat) / (p_sigma * sqrt(p_mat));
        double  d2 = d1 - p_sigma * sqrt(p_mat);
        return -p_S * NormalCumulativeDistribution()(-d1) + p_strike * exp(-p_r * p_mat) * NormalCumulativeDistribution()(-d2);
    }

};
#endif /* EUROPEANOPTIONS_H */
