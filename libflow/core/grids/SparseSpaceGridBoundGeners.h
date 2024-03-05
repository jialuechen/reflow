
#ifndef SPARSESPACEGRIDBOUNDGENERS_H
#define SPARSESPACEGRIDBOUNDGENERS_H
#include <geners/AbsReaderWriter.hh>
#include "libflow/core/grids/SparseSpaceGridBound.h"
#include "libflow/core/grids/SpaceGridGeners.h"

/** \file SparseSpaceGridBoundGeners.h
 * \brief Define non intrusive  serialization with random access
*  \author Xavier Warin
 */

// Concrete reader/writer for class SparseSpaceGridBound
// Note publication of SparseSpaceGridBound as "wrapped_type".
struct SparseSpaceGridBoundGeners: public gs::AbsReaderWriter<libflow::SpaceGrid>
{
    typedef libflow::SpaceGrid wrapped_base;
    typedef libflow::SparseSpaceGridBound wrapped_type;

    // Methods that have to be overridden from the base
    bool write(std::ostream &, const wrapped_base &, bool p_dumpId) const override;
    wrapped_type *read(const gs::ClassId &p_id, std::istream &p_in) const override;

    // The class id for SparseSpaceGridBound  will be needed both in the "read" and "write"
    // methods. Because of this, we will just return it from one static
    // function.
    static const gs::ClassId &wrappedClassId();
};

gs_specialize_class_id(libflow::SparseSpaceGridBound, 1)
gs_declare_type_external(libflow::SparseSpaceGridBound)
gs_associate_serialization_factory(libflow::SparseSpaceGridBound, StaticSerializationFactoryForSpaceGrid)

#endif  /* SPARSESPACEGRIDBOUNDGENERS_H */
