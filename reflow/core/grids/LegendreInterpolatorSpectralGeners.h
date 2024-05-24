
#ifndef LEGENDREINTERPOLATORSPECTRALGENERS_H
#define LEGENDREINTERPOLATORSPECTRALGENERS_H
#include <geners/AbsReaderWriter.hh>
#include "reflow/core/grids/LegendreInterpolatorSpectral.h"
#include "reflow/core/grids/InterpolatorSpectralGeners.h"


struct LegendreInterpolatorSpectralGeners: public gs::AbsReaderWriter<reflow::InterpolatorSpectral>
{
    typedef reflow::InterpolatorSpectral wrapped_base;
    typedef reflow::LegendreInterpolatorSpectral wrapped_type;

    // Methods that have to be overridden from the base
    bool write(std::ostream &, const wrapped_base &, bool p_dumpId) const override;
    wrapped_type *read(const gs::ClassId &p_id, std::istream &p_in) const override;

    // The class id for LegendreInterpolatorSpectral  will be needed both in the "read" and "write"
    // methods. Because of this, we will just return it from one static
    // function.
    static const gs::ClassId &wrappedClassId();
};

gs_specialize_class_id(reflow::LegendreInterpolatorSpectral, 1)
gs_declare_type_external(reflow::LegendreInterpolatorSpectral)
gs_associate_serialization_factory(reflow::LegendreInterpolatorSpectral, StaticSerializationFactoryForInterpolatorSpectral)

#endif
