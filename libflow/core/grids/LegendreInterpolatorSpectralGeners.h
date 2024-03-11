
#ifndef LEGENDREINTERPOLATORSPECTRALGENERS_H
#define LEGENDREINTERPOLATORSPECTRALGENERS_H
#include <geners/AbsReaderWriter.hh>
#include "libflow/core/grids/LegendreInterpolatorSpectral.h"
#include "libflow/core/grids/InterpolatorSpectralGeners.h"


struct LegendreInterpolatorSpectralGeners: public gs::AbsReaderWriter<libflow::InterpolatorSpectral>
{
    typedef libflow::InterpolatorSpectral wrapped_base;
    typedef libflow::LegendreInterpolatorSpectral wrapped_type;

    // Methods that have to be overridden from the base
    bool write(std::ostream &, const wrapped_base &, bool p_dumpId) const override;
    wrapped_type *read(const gs::ClassId &p_id, std::istream &p_in) const override;

    // The class id for LegendreInterpolatorSpectral  will be needed both in the "read" and "write"
    // methods. Because of this, we will just return it from one static
    // function.
    static const gs::ClassId &wrappedClassId();
};

gs_specialize_class_id(libflow::LegendreInterpolatorSpectral, 1)
gs_declare_type_external(libflow::LegendreInterpolatorSpectral)
gs_associate_serialization_factory(libflow::LegendreInterpolatorSpectral, StaticSerializationFactoryForInterpolatorSpectral)

#endif
