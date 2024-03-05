// Copyright (C) 2016 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#ifndef GENERALSPACEGRIDDERIVEDGENERS_H
#define GENERALSPACEGRIDDERIVEDGENERS_H
#include <geners/AbsReaderWriter.hh>
#include "libflow/core/grids/GeneralSpaceGrid.h"
#include "libflow/core/grids/FullGridGeners.h"

/** \file GeneralSpaceGridGeners.h
 * \brief Define non intrusive  serialization with random acces
*  \author Xavier Warin
 */

// Concrete reader/writer for class GeneralSpaceGrid
// Note publication of GeneralSpaceGrid as "wrapped_type".
struct GeneralSpaceGridDerivedGeners: public gs::AbsReaderWriter<libflow::FullGrid>
{
    typedef libflow::FullGrid wrapped_base;
    typedef libflow::GeneralSpaceGrid wrapped_type;

    // Methods that have to be overriden from the base
    bool write(std::ostream &, const wrapped_base &, bool p_dumpId) const override;
    wrapped_type *read(const gs::ClassId &p_id, std::istream &p_in) const override;

    // The class id forGeneralSpaceGrid  will be needed both in the "read" and "write"
    // methods. Because of this, we will just return it from one static
    // function.
    static const gs::ClassId &wrappedClassId();
};

gs_specialize_class_id(libflow::GeneralSpaceGrid, 1)
gs_declare_type_external(libflow::GeneralSpaceGrid)
gs_associate_serialization_factory(libflow::GeneralSpaceGrid, StaticSerializationFactoryForFullGrid)

#endif  /* GENERALSPACEGRIDDERIVEDGENERS_H */
