
#ifndef FULLGRIDGENERS_H
#define FULLGRIDGENERS_H
#include "reflow/core/grids/FullGrid.h"
#include <geners/AbsReaderWriter.hh>
#include <geners/associate_serialization_factory.hh>

/** \file FullGridGeners.h
 * \brief Define non intrusive  serialization with random acces
*  \author Xavier Warin
 */

/// \¢lass
///  I/O factory for classes derived from .
// Note publication of the base class and absence of public constructors.
class SerializationFactoryForFullGrid : public gs::DefaultReaderWriter<reflow::FullGrid>
{
    typedef DefaultReaderWriter<reflow::FullGrid> Base;
    friend class gs::StaticReaderWriter<SerializationFactoryForFullGrid>;
    SerializationFactoryForFullGrid();
};

// SerializationFactoryForFullGrid wrapped into a singleton
typedef gs::StaticReaderWriter<SerializationFactoryForFullGrid> StaticSerializationFactoryForFullGrid;

gs_specialize_class_id(reflow::FullGrid, 1)
gs_declare_type_external(reflow::FullGrid)
gs_associate_serialization_factory(reflow::FullGrid, StaticSerializationFactoryForFullGrid)

#endif  /* FULLGRIDGENERS_H */
