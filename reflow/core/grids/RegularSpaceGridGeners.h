
#ifndef REGULARSPACEGRIDGENERS_H
#define REGULARSPACEGRIDGENERS_H
#include <geners/AbsReaderWriter.hh>
#include "reflow/core/grids/RegularSpaceGrid.h"
#include "reflow/core/grids/SpaceGridGeners.h"

struct RegularSpaceGridGeners: public gs::AbsReaderWriter<reflow::SpaceGrid>
{
    typedef reflow::SpaceGrid wrapped_base;
    typedef reflow::RegularSpaceGrid wrapped_type;

    // Methods that have to be overridden from the base
    bool write(std::ostream &, const wrapped_base &, bool p_dumpId) const override;
    wrapped_type *read(const gs::ClassId &p_id, std::istream &p_in) const override;

    // The class id for RegularSpaceGrid  will be needed both in the "read" and "write"
    // methods. Because of this, we will just return it from one static
    // function.
    static const gs::ClassId &wrappedClassId();
};

gs_specialize_class_id(reflow::RegularSpaceGrid, 1)
gs_declare_type_external(reflow::RegularSpaceGrid)
gs_associate_serialization_factory(reflow::RegularSpaceGrid, StaticSerializationFactoryForSpaceGrid)

#endif  /* REGULARSPACEGRIDGENERS_H */
