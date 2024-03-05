
#ifndef  COMPARISONUTILS_H
#define COMPARISONUTILS_H
#include <limits>
#include <cstdlib>
#include <cmath>
#include <iostream>

/** \file comparisonUtils.h
 *  \brief Provide comparison stuff
 *  \author Xavier Warin
 */

namespace libflow
{

/// \brief test if two numbers are almost equal
template<class T>
typename std::enable_if < !std::numeric_limits<T>::is_integer, bool >::type
almostEqual(T x, T y, int ulp)
{
    // the machine epsilon has to be scaled to the magnitude of the larger value
    // and multiplied by the desired precision in ULPs (units in the last place)
    return std::abs(x - y) <= std::numeric_limits<T>::epsilon() * std::abs(x + y) * ulp;
}
/// \brief test if lesser or equal to with numerical tolerance
template<class T>
inline bool isLesserOrEqual(const T &x, const T &y)
{
    if (std::fabs(x) > 1e-7)
        return (x  <= y  + 100 * (std::fabs(y) + std::fabs(x)) * std::numeric_limits<T>::epsilon());
    else
        return (x  <= y  + 1000 * std::numeric_limits<T>::epsilon());
}
/// \brief test is strictly below
template< class T >
inline bool isStrictlyLesser(const T &x, const T &y)
{
    return (x < y - 1e3 * std::numeric_limits<T>::epsilon());
}
/// \brief test is strictly above
template< class T >
inline bool isStrictlyMore(const T &x, const T &y)
{
    return (x > y + 1e3 * std::numeric_limits<T>::epsilon());
}
/// \brief calculate round of a number with numerical tolerance  (by above)
template< class T >
inline int roundIntAbove(const T &x)
{
    return static_cast<int>(x * (1 + 1e3 * std::numeric_limits<T>::epsilon())  + 1e3 * std::numeric_limits<T>::epsilon());
}
/// \brief calculate round of a number with numerical tolerance  (by below)
template< class T >
inline int roundIntBelow(const T &x)
{
    return static_cast<int>(x * (1 - 1e3 * std::numeric_limits<T>::epsilon()));
}
}
#endif /* COMPARISONUTILS_H */
