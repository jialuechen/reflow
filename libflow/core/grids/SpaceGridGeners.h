
#ifndef SPACEGRIDGENERS_H
#define SPACEGRIDGENERS_H
#include "libflow/core/grids/SpaceGrid.h"
#include <geners/AbsReaderWriter.hh>
#include <geners/associate_serialization_factory.hh>

class SerializationFactoryForSpaceGrid : public gs::DefaultReaderWriter<libflow::SpaceGrid>
{
    typedef DefaultReaderWriter<libflow::SpaceGrid> Base;
    friend class gs::StaticReaderWriter<SerializationFactoryForSpaceGrid>;
    SerializationFactoryForSpaceGrid();
};

// SerializationFactoryForSpaceGrid wrapped into a singleton
typedef gs::StaticReaderWriter<SerializationFactoryForSpaceGrid> StaticSerializationFactoryForSpaceGrid;

gs_specialize_class_id(libflow::SpaceGrid, 1)
gs_declare_type_external(libflow::SpaceGrid)
gs_associate_serialization_factory(libflow::SpaceGrid, StaticSerializationFactoryForSpaceGrid)

#endif
