#include <memory>
#include "geners/IOException.hh"
#include "geners/GenericIO.hh"
#include "reflow/core/grids/GeneralSpaceGridDerivedGeners.h"
#include "reflow/core/utils/eigenGeners.h"

using namespace reflow;
using namespace std;


bool GeneralSpaceGridDerivedGeners::write(ostream &p_of, const wrapped_base &p_base,
        const bool p_dumpId) const
{
    // If necessary, write out the class id
    const bool status = p_dumpId ? wrappedClassId().write(p_of) : true;

    // Write the object data out
    if (status)
    {
        const wrapped_type &w = dynamic_cast<const wrapped_type &>(p_base);
        int isize = w.getMeshPerDimension().size();
        gs::write_pod(p_of, isize);
        for (int i = 0; i < isize; ++i)
        {
            int isizeDim = w.getMeshPerDimension()[i]->size();
            gs::write_pod(p_of, isizeDim);
            gs::write_pod_array(p_of,  w.getMeshPerDimension()[i]->data(), isizeDim);
        }
    }

    // Return "true" on success
    return status && !p_of.fail();
}

GeneralSpaceGrid *GeneralSpaceGridDerivedGeners::read(const gs::ClassId &p_id, istream &p_in) const
{
    // Validate the class id. You might want to implement
    // class versioning here.
    wrappedClassId().ensureSameId(p_id);

    // Read in the object data
    int isize = 0;
    gs::read_pod(p_in, &isize);
    vector< shared_ptr< Eigen::ArrayXd > > meshPerDimension(isize);
    for (int i = 0; i < isize; ++i)
    {
        int isizeDim = 0;
        gs::read_pod(p_in, &isizeDim);
        meshPerDimension[i] = make_shared<Eigen::ArrayXd >(isizeDim);
        gs::read_array(p_in, meshPerDimension[i]->data(), isizeDim);
    }

    // Check that the stream is in a valid state
    if (p_in.fail()) throw gs::IOReadFailure("In BIO::read: input stream failure");

    // Return the object
    return new GeneralSpaceGrid(meshPerDimension);
}

const gs::ClassId &GeneralSpaceGridDerivedGeners::wrappedClassId()
{
    static const gs::ClassId wrapId(gs::ClassId::makeId<wrapped_type>());
    return wrapId;
}

