
#ifndef SPARSEGRIDHIERARONEPOINTBOUND_H
#define SPARSEGRIDHIERARONEPOINTBOUND_H
#include "libflow/core/sparse/SparseGridHierarOnePoint.h"

/** \file SparseGridHierarOnePointBound.h
 * \brief Base class to Hierarchize a single point
 * \author Xavier Warin
 */

namespace libflow
{

/// \class SparseGridHierarOnePointBound SparseGridHierarOnePointBound.h
/// Abstract class for only only point Hierarchization with boundary points
class SparseGridHierarOnePointBound : SparseGridHierarOnePoint
{
public :

    /// \brief Default constructor
    SparseGridHierarOnePointBound() {}

    /// \brief Default destructor
    virtual ~SparseGridHierarOnePointBound() {}
};
}

#endif /* SPARSEGRIDHIERARONEPOINTBOUND_H */
