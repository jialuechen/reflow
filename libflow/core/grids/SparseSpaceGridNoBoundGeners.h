
#ifndef SPARSESPACEGRIDNOBOUNDGENERS_H
#define SPARSESPACEGRIDNOBOUNDGENERS_H
#include <geners/AbsReaderWriter.hh>
#include "libflow/core/grids/SparseSpaceGridNoBound.h"
#include "libflow/core/grids/SpaceGridGeners.h"

struct SparseSpaceGridNoBoundGeners: public gs::AbsReaderWriter<libflow::SpaceGrid>
{
    typedef libflow::SpaceGrid wrapped_base;
    typedef libflow::SparseSpaceGridNoBound wrapped_type;

    // Methods that have to be overridden from the base
    bool write(std::ostream &, const wrapped_base &, bool p_dumpId) const override;
    wrapped_type *read(const gs::ClassId &p_id, std::istream &p_in) const override;

    // The class id for SparseSpaceGridNoBound  will be needed both in the "read" and "write"
    // methods. Because of this, we will just return it from one static
    // function.
    static const gs::ClassId &wrappedClassId();
};

gs_specialize_class_id(libflow::SparseSpaceGridNoBound, 1)
gs_declare_type_external(libflow::SparseSpaceGridNoBound)
gs_associate_serialization_factory(libflow::SparseSpaceGridNoBound, StaticSerializationFactoryForSpaceGrid)

#endif  /* SPARSESPACEGRIDNOBOUNDGENERS_H */
