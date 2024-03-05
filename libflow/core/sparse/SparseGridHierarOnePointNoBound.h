// Copyright (C) 2016 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#ifndef SPARSEGRIDHIERARONEPOINTNOBOUND_H
#define SPARSEGRIDHIERARONEPOINTNOBOUND_H
#include "libflow/core/sparse/SparseGridHierarOnePoint.h"

/** \file SparseGridHierarOnePointNoBound.h
 * \brief Base class to Hierarchize a single point
 * \author Xavier Warin
 */

namespace libflow
{

/// \class SparseGridHierarOnePointNoBound SparseGridHierarOnePointNoBound.h
/// Abstract class for only only point Hierarchization without boundary points
class SparseGridHierarOnePointNoBound : SparseGridHierarOnePoint
{
public :

    /// \brief Default constructor
    SparseGridHierarOnePointNoBound() {}

    /// \brief Default destructor
    virtual ~SparseGridHierarOnePointNoBound() {}
};
}

#endif /* SPARSEGRIDHIERARONEPOINTNOBOUND_H */
