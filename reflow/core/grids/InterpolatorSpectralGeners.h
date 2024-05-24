
#ifndef INTERPOLATORSPECTRALGENERS_H
#define INTERPOLATORSPECTRALGENERS_H
#include "reflow/core/grids/InterpolatorSpectral.h"
#include <geners/AbsReaderWriter.hh>
#include <geners/associate_serialization_factory.hh>

class SerializationFactoryForInterpolatorSpectral : public gs::DefaultReaderWriter<reflow::InterpolatorSpectral>
{
    typedef DefaultReaderWriter<reflow::InterpolatorSpectral> Base;
    friend class gs::StaticReaderWriter<SerializationFactoryForInterpolatorSpectral>;
    SerializationFactoryForInterpolatorSpectral();
};

// SerializationFactoryForInterpolatorSpectral wrapped into a singleton
typedef gs::StaticReaderWriter<SerializationFactoryForInterpolatorSpectral> StaticSerializationFactoryForInterpolatorSpectral;

gs_specialize_class_id(reflow::InterpolatorSpectral, 1)
gs_declare_type_external(reflow::InterpolatorSpectral)
gs_associate_serialization_factory(reflow::InterpolatorSpectral, StaticSerializationFactoryForInterpolatorSpectral)

#endif
