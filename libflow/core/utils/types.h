// Copyright (C) 2016 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#ifndef TYPES_H
#define TYPES_H
#include <memory>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <Eigen/Dense>

/** \file types.h
 * \brief defines some complicated type  to lighten the code
 * \author Xavier Warin
 */

#define SubMeshIntCoord     Eigen::Array<  std::array<int, 2 >, Eigen::Dynamic, 1 > ///< N dimensional mesh shortcut

#endif
