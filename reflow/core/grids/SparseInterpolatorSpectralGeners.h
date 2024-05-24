
#ifndef SPARSEINTERPOLATORSPECTRALGENERS_H
#define SPARSEINTERPOLATORSPECTRALGENERS_H
#include <geners/AbsReaderWriter.hh>
#include "reflow/core/grids/SparseInterpolatorSpectral.h"
#include "reflow/core/grids/InterpolatorSpectralGeners.h"


struct SparseInterpolatorSpectralGeners: public gs::AbsReaderWriter<reflow::InterpolatorSpectral>
{
    typedef reflow::InterpolatorSpectral wrapped_base;
    typedef reflow::SparseInterpolatorSpectral wrapped_type;

    // Methods that have to be overridden from the base
    bool write(std::ostream &, const wrapped_base &, bool p_dumpId) const override;
    wrapped_type *read(const gs::ClassId &p_id, std::istream &p_in) const override;

    // The class id for SparseInterpolatorSpectral  will be needed both in the "read" and "write"
    // methods. Because of this, we will just return it from one static
    // function.
    static const gs::ClassId &wrappedClassId();
};

gs_specialize_class_id(reflow::SparseInterpolatorSpectral, 1)
gs_declare_type_external(reflow::SparseInterpolatorSpectral)
gs_associate_serialization_factory(reflow::SparseInterpolatorSpectral, StaticSerializationFactoryForInterpolatorSpectral)

#endif
