
#ifndef LINEARINTERPOLATORSPECTRALGENERS_H
#define LINEARINTERPOLATORSPECTRALGENERS_H
#include <geners/AbsReaderWriter.hh>
#include "libflow/core/grids/LinearInterpolatorSpectral.h"
#include "libflow/core/grids/InterpolatorSpectralGeners.h"


/** \file LinearInterpolatorSpectralGeners.h
 * \brief Define non intrusive  serialization with random acces
*  \author Xavier Warin
 */

// Concrete reader/writer for class LinearInterpolatorSpectral
// Note publication of LinearInterpolatorSpectral as "wrapped_type".
struct LinearInterpolatorSpectralGeners: public gs::AbsReaderWriter<libflow::InterpolatorSpectral>
{
    typedef libflow::InterpolatorSpectral wrapped_base;
    typedef libflow::LinearInterpolatorSpectral wrapped_type;

    // Methods that have to be overridden from the base
    bool write(std::ostream &, const wrapped_base &, bool p_dumpId) const override;
    wrapped_type *read(const gs::ClassId &p_id, std::istream &p_in) const override;

    // The class id for LinearInterpolatorSpectral  will be needed both in the "read" and "write"
    // methods. Because of this, we will just return it from one static
    // function.
    static const gs::ClassId &wrappedClassId();
};

gs_specialize_class_id(libflow::LinearInterpolatorSpectral, 1)
gs_declare_type_external(libflow::LinearInterpolatorSpectral)
gs_associate_serialization_factory(libflow::LinearInterpolatorSpectral, StaticSerializationFactoryForInterpolatorSpectral)

#endif
