#include <memory>
#include "geners/IOException.hh"
#include "geners/GenericIO.hh"
#include "geners/arrayIO.hh"
#include "geners/mapIO.hh"
#include "reflow/core/grids/SparseSpaceGridBoundGeners.h"
#include "reflow/core/utils/eigenGeners.h"
#include "reflow/core/grids/SparseOrderTinyVectorGeners.h"

using namespace reflow;
using namespace std;

bool SparseSpaceGridBoundGeners::write(ostream &p_of, const wrapped_base &p_base,
                                       const bool p_dumpId) const
{
    // If necessary, write out the class id
    const bool status = p_dumpId ? wrappedClassId().write(p_of) : true;

    // Write the object data out
    if (status)
    {
        const wrapped_type &w = dynamic_cast<const wrapped_type &>(p_base);
        gs::write_pod(p_of, w.getNbPoints());
        gs::write_pod(p_of, w.getLevelMax());
        gs::write_item(p_of, w.getLowValues());
        gs::write_item(p_of, w.getSizeDomain());
        gs::write_item(p_of, w.getWeight());
        gs::write_item(p_of, *w.getDataSet());
        gs::write_pod(p_of, w.getDegree());
        gs::write_item(p_of, *w.getSon());
        gs::write_item(p_of, *w.getNeighbourBound());
        gs::write_pod(p_of, w.getIBase());
    }

    // Return "true" on success
    return status && !p_of.fail();
}

SparseSpaceGridBound *SparseSpaceGridBoundGeners::read(const gs::ClassId &p_id, istream &p_in) const
{
    // Validate the class id. You might want to implement
    // class versioning here.
    wrappedClassId().ensureSameId(p_id);


    size_t  nbPoints;
    gs::read_pod(p_in, &nbPoints);
    int levelMax;
    gs::read_pod(p_in, &levelMax);
    unique_ptr< Eigen::ArrayXd > lowValues  = gs::read_item< Eigen::ArrayXd  >(p_in);
    unique_ptr< Eigen::ArrayXd > sizeDomain = gs::read_item< Eigen::ArrayXd  >(p_in);
    unique_ptr< Eigen::ArrayXd > weight  = gs::read_item< Eigen::ArrayXd  >(p_in);
    CPP11_auto_ptr<SparseSet>     pdataSet =  gs::read_item<SparseSet>(p_in);
    shared_ptr<SparseSet> dataSet(std::move(pdataSet));
    size_t degree ;
    gs::read_pod(p_in, &degree);
    CPP11_auto_ptr< Eigen::Array< array<int, 2 >, Eigen::Dynamic, Eigen::Dynamic > >    pson =  gs::read_item< Eigen::Array< array<int, 2 >, Eigen::Dynamic, Eigen::Dynamic > >(p_in);
    shared_ptr<  Eigen::Array< array<int, 2 >, Eigen::Dynamic, Eigen::Dynamic > >  son(std::move(pson));
    CPP11_auto_ptr< Eigen::Array< array<int, 2 >, Eigen::Dynamic, Eigen::Dynamic > >    pneighbourBound =  gs::read_item< Eigen::Array< array<int, 2 >, Eigen::Dynamic, Eigen::Dynamic > >(p_in);
    shared_ptr<  Eigen::Array< array<int, 2 >, Eigen::Dynamic, Eigen::Dynamic > >  neighbourBound(std::move(pneighbourBound));
    int iBase ;
    gs::read_pod(p_in, &iBase);
    // Check that the stream is in a valid state
    if (p_in.fail()) throw gs::IOReadFailure("In BIO::read: input stream failure");

    // Return the object
    return new reflow::SparseSpaceGridBound(*lowValues, *sizeDomain, levelMax, *weight, dataSet, nbPoints, degree, son, neighbourBound, iBase);
}

const gs::ClassId &SparseSpaceGridBoundGeners::wrappedClassId()
{
    static const gs::ClassId wrapId(gs::ClassId::makeId<wrapped_type>());
    return wrapId;
}

