#include "geners/IOException.hh"
#include "geners/GenericIO.hh"
#include "libflow/core/grids/LinearInterpolatorSpectralGeners.h"
#include "libflow/core/utils/eigenGeners.h"

using namespace libflow;
using namespace std;

bool LinearInterpolatorSpectralGeners::write(std::ostream &p_of, const wrapped_base &p_base,
        const bool p_dumpId) const
{
    // If necessary, write out the class id
    const bool status = p_dumpId ? wrappedClassId().write(p_of) : true;

    // Write the object data out
    if (status)
    {
        const wrapped_type &w = dynamic_cast<const wrapped_type &>(p_base);
        gs::write_item(p_of, w.getValues());
    }

    // Return "true" on success
    return status && !p_of.fail();
}

LinearInterpolatorSpectral *LinearInterpolatorSpectralGeners::read(const gs::ClassId &p_id, std::istream &p_in) const
{
    // Validate the class id. You might want to implement
    // class versioning here.
    wrappedClassId().ensureSameId(p_id);

    // Read in the object data
    unique_ptr< Eigen::ArrayXd > values  = gs::read_item< Eigen::ArrayXd  >(p_in);

    // Check that the stream is in a valid state
    if (p_in.fail()) throw gs::IOReadFailure("In BIO::read: input stream failure");

    // Return the object
    return new LinearInterpolatorSpectral(*values);
}

const gs::ClassId &LinearInterpolatorSpectralGeners::wrappedClassId()
{
    static const gs::ClassId wrapId(gs::ClassId::makeId<wrapped_type>());
    return wrapId;
}

