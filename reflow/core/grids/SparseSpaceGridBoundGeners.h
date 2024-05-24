
#ifndef SPARSESPACEGRIDBOUNDGENERS_H
#define SPARSESPACEGRIDBOUNDGENERS_H
#include <geners/AbsReaderWriter.hh>
#include "reflow/core/grids/SparseSpaceGridBound.h"
#include "reflow/core/grids/SpaceGridGeners.h"

struct SparseSpaceGridBoundGeners: public gs::AbsReaderWriter<reflow::SpaceGrid>
{
    typedef reflow::SpaceGrid wrapped_base;
    typedef reflow::SparseSpaceGridBound wrapped_type;

    // Methods that have to be overridden from the base
    bool write(std::ostream &, const wrapped_base &, bool p_dumpId) const override;
    wrapped_type *read(const gs::ClassId &p_id, std::istream &p_in) const override;

    // The class id for SparseSpaceGridBound  will be needed both in the "read" and "write"
    // methods. Because of this, we will just return it from one static
    // function.
    static const gs::ClassId &wrappedClassId();
};

gs_specialize_class_id(reflow::SparseSpaceGridBound, 1)
gs_declare_type_external(reflow::SparseSpaceGridBound)
gs_associate_serialization_factory(reflow::SparseSpaceGridBound, StaticSerializationFactoryForSpaceGrid)

#endif  /* SPARSESPACEGRIDBOUNDGENERS_H */
