#include "geners/IOException.hh"
#include "geners/GenericIO.hh"
#include "libflow/core/grids/RegularLegendreGridDerivedGeners.h"
#include "libflow/core/utils/eigenGeners.h"

using namespace libflow;
using namespace std;

bool RegularLegendreGridDerivedGeners::write(ostream &p_of, const wrapped_base &p_base,
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
        gs::write_item(p_of, w.getGllPoints());
        gs::write_item(p_of, *w.getFInterpol());
        gs::write_pod_array(p_of, w.getFirstPoints().data(), isize);
        gs::write_pod_array(p_of, w.getLastPoints().data(), isize);


    }

    // Return "true" on success
    return status && !p_of.fail();
}

RegularLegendreGrid *RegularLegendreGridDerivedGeners::read(const gs::ClassId &p_id, istream &p_in) const
{
    // Validate the class id. You might want to implement
    // class versioning here.
    wrappedClassId().ensureSameId(p_id);

    // Read in the object data
    int isize = 0;
    gs::read_pod(p_in, &isize);
    Eigen::ArrayXd lowValue(isize), step(isize);
    Eigen::ArrayXi firstPoints(isize), lastPoints(isize);
    gs::read_pod_array(p_in, lowValue.data(), isize);
    gs::read_pod_array(p_in, step.data(), isize);
    Eigen::ArrayXi nbstep(isize);
    gs::read_pod_array(p_in, nbstep.data(), isize);
    CPP11_auto_ptr< vector< Eigen::ArrayXd > > gllPoints =  gs::read_item< vector< Eigen::ArrayXd > >(p_in);
    CPP11_auto_ptr< vector< Eigen::ArrayXXd > > fInterpol =  gs::read_item< vector< Eigen::ArrayXXd > >(p_in);
    gs::read_pod_array(p_in, firstPoints.data(), isize);
    gs::read_pod_array(p_in, lastPoints.data(), isize);


    // Check that the stream is in a valid state
    if (p_in.fail()) throw gs::IOReadFailure("In BIO::read: input stream failure");

    // Return the object
    return new RegularLegendreGrid(lowValue, step, nbstep, *gllPoints, std::move(fInterpol), firstPoints, lastPoints);
}

const gs::ClassId &RegularLegendreGridDerivedGeners::wrappedClassId()
{
    static const gs::ClassId wrapId(gs::ClassId::makeId<wrapped_type>());
    return wrapId;
}

