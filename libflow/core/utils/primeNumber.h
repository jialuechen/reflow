// Copyright (C) 2016 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#ifndef PRIMENUMBER_H
#define PRIMENUMBER_H
#include <vector>
/** \file primeNumber.h
 *  \brief Decompose a number in prime numbers
 */
namespace libflow
{

///\fn std::vector<int> primeNumber(int  n)
/// \param n  number to recompose
/// \return vector of prime terms
std::vector<int> primeNumber(int  n);

}
#endif /* PRIMENUMBER_H */
