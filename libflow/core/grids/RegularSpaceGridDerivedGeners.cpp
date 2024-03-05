#include "geners/IOException.hh"
#include "geners/GenericIO.hh"
#include "libflow/core/grids/RegularSpaceGridDerivedGeners.h"
#include "libflow/core/utils/eigenGeners.h"

using namespace libflow;
using namespace std;

bool RegularSpaceGridDerivedGeners::write(ostream &p_of, const wrapped_base &p_base,
        const bool p_dumpId) const
{
    // If necessary, write out the class id
    const bool status = p_dumpId ? wrappedClassId().write(p_of) : true;

    // Write the object data out
    if (status)
    {
        const wrapped_type &w = dynamic_cast<const wrapped_type &>(p_base);
        int isize = w.getLowValues().size();
        gs::write_pod(p_of, isize);
        gs::write_pod_array(p_of, w.getLowValues().data(), isize);
        gs::write_pod_array(p_of, w.getStep().data(), isize);
        gs::write_pod_array(p_of, w.getNbStep().data(), isize);
    }

    // Return "true" on success
    return status && !p_of.fail();
}

RegularSpaceGrid *RegularSpaceGridDerivedGeners::read(const gs::ClassId &p_id, istream &p_in) const
{
    // Validate the class id. You might want to implement
    // class versioning here.
    wrappedClassId().ensureSameId(p_id);

    // Read in the object data
    int isize = 0;
    gs::read_pod(p_in, &isize);
    Eigen::ArrayXd lowValue(isize), step(isize);
    gs::read_pod_array(p_in, lowValue.data(), isize);
    gs::read_pod_array(p_in, step.data(), isize);
    Eigen::ArrayXi nbstep(isize);
    gs::read_pod_array(p_in, nbstep.data(), isize);

    // Check that the stream is in a valid state
    if (p_in.fail()) throw gs::IOReadFailure("In BIO::read: input stream failure");

    // Return the object
    return new RegularSpaceGrid(lowValue, step, nbstep);
}

const gs::ClassId &RegularSpaceGridDerivedGeners::wrappedClassId()
{
    static const gs::ClassId wrapId(gs::ClassId::makeId<wrapped_type>());
    return wrapId;
}

