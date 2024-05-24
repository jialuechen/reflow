
#ifndef SPACEGRIDGENERS_H
#define SPACEGRIDGENERS_H
#include "reflow/core/grids/SpaceGrid.h"
#include <geners/AbsReaderWriter.hh>
#include <geners/associate_serialization_factory.hh>

class SerializationFactoryForSpaceGrid : public gs::DefaultReaderWriter<reflow::SpaceGrid>
{
    typedef DefaultReaderWriter<reflow::SpaceGrid> Base;
    friend class gs::StaticReaderWriter<SerializationFactoryForSpaceGrid>;
    SerializationFactoryForSpaceGrid();
};

// SerializationFactoryForSpaceGrid wrapped into a singleton
typedef gs::StaticReaderWriter<SerializationFactoryForSpaceGrid> StaticSerializationFactoryForSpaceGrid;

gs_specialize_class_id(reflow::SpaceGrid, 1)
gs_declare_type_external(reflow::SpaceGrid)
gs_associate_serialization_factory(reflow::SpaceGrid, StaticSerializationFactoryForSpaceGrid)

#endif
