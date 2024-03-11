
#ifndef SPARSEGRIDHIERARONEPOINTNOBOUND_H
#define SPARSEGRIDHIERARONEPOINTNOBOUND_H
#include "libflow/core/sparse/SparseGridHierarOnePoint.h"

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
