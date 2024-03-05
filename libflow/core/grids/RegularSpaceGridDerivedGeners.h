
#ifndef REGULARSPACEGRIDDERIVEDGENERS_H
#define REGULARSPACEGRIDDERIVEDGENERS_H
#include <geners/AbsReaderWriter.hh>
#include "libflow/core/grids/RegularSpaceGrid.h"
#include "libflow/core/grids/FullGridGeners.h"

/** \file RegularSpaceGridGeners.h
 * \brief Define non intrusive  serialization with random access
*  \author Xavier Warin
 */

// Concrete reader/writer for class RegularSpaceGrid
// Note publication of RegularSpaceGrid as "wrapped_type".
struct RegularSpaceGridDerivedGeners: public gs::AbsReaderWriter<libflow::FullGrid>
{
    typedef libflow::FullGrid wrapped_base;
    typedef libflow::RegularSpaceGrid wrapped_type;

    // Methods that have to be overridden from the base
    bool write(std::ostream &, const wrapped_base &, bool p_dumpId) const override;
    wrapped_type *read(const gs::ClassId &p_id, std::istream &p_in) const override;

    // The class id for RegularSpaceGrid  will be needed both in the "read" and "write"
    // methods. Because of this, we will just return it from one static
    // function.
    static const gs::ClassId &wrappedClassId();
};

gs_specialize_class_id(libflow::RegularSpaceGrid, 1)
gs_declare_type_external(libflow::RegularSpaceGrid)
gs_associate_serialization_factory(libflow::RegularSpaceGrid, StaticSerializationFactoryForFullGrid)

#endif  /* REGULARSPACEGRIDDERIVEDGENERS_H */
