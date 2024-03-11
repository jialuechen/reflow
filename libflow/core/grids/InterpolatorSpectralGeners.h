
#ifndef INTERPOLATORSPECTRALGENERS_H
#define INTERPOLATORSPECTRALGENERS_H
#include "libflow/core/grids/InterpolatorSpectral.h"
#include <geners/AbsReaderWriter.hh>
#include <geners/associate_serialization_factory.hh>

class SerializationFactoryForInterpolatorSpectral : public gs::DefaultReaderWriter<libflow::InterpolatorSpectral>
{
    typedef DefaultReaderWriter<libflow::InterpolatorSpectral> Base;
    friend class gs::StaticReaderWriter<SerializationFactoryForInterpolatorSpectral>;
    SerializationFactoryForInterpolatorSpectral();
};

// SerializationFactoryForInterpolatorSpectral wrapped into a singleton
typedef gs::StaticReaderWriter<SerializationFactoryForInterpolatorSpectral> StaticSerializationFactoryForInterpolatorSpectral;

gs_specialize_class_id(libflow::InterpolatorSpectral, 1)
gs_declare_type_external(libflow::InterpolatorSpectral)
gs_associate_serialization_factory(libflow::InterpolatorSpectral, StaticSerializationFactoryForInterpolatorSpectral)

#endif
