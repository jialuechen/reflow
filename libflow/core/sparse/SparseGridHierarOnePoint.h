// Copyright (C) 2016 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#ifndef SPARSEGRIDHIERARONEPOINT_H
#define SPARSEGRIDHIERARONEPOINT_H

/** \file SparseGridHierarOnePoint.h
 * \brief Base class to Hierarchize a single point
 * \author Xavier Warin
 */

namespace libflow
{

/// \class SparseGridHierarOnePoint SparseGridHierarOnePoint.h
/// Abstract class for only only point Hierarchization
class SparseGridHierarOnePoint
{
public :
    /// \brief Default constructor
    SparseGridHierarOnePoint() {}

    /// \brief Default destructor
    virtual ~SparseGridHierarOnePoint() {}
};
}

#endif /* SPARSEGRIDHIERARONEPOINT_H */
