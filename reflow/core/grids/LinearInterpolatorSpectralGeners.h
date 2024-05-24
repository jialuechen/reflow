
#ifndef LINEARINTERPOLATORSPECTRALGENERS_H
#define LINEARINTERPOLATORSPECTRALGENERS_H
#include <geners/AbsReaderWriter.hh>
#include "reflow/core/grids/LinearInterpolatorSpectral.h"
#include "reflow/core/grids/InterpolatorSpectralGeners.h"

struct LinearInterpolatorSpectralGeners: public gs::AbsReaderWriter<reflow::InterpolatorSpectral>
{
    typedef reflow::InterpolatorSpectral wrapped_base;
    typedef reflow::LinearInterpolatorSpectral wrapped_type;

    // Methods that have to be overridden from the base
    bool write(std::ostream &, const wrapped_base &, bool p_dumpId) const override;
    wrapped_type *read(const gs::ClassId &p_id, std::istream &p_in) const override;

    // The class id for LinearInterpolatorSpectral  will be needed both in the "read" and "write"
    // methods. Because of this, we will just return it from one static
    // function.
    static const gs::ClassId &wrappedClassId();
};

gs_specialize_class_id(reflow::LinearInterpolatorSpectral, 1)
gs_declare_type_external(reflow::LinearInterpolatorSpectral)
gs_associate_serialization_factory(reflow::LinearInterpolatorSpectral, StaticSerializationFactoryForInterpolatorSpectral)

#endif
