
#ifndef  CONTINUATIONVALUETREEGENERS_H
#define  CONTINUATIONVALUETREEGENERS_H
#include <Eigen/Dense>
#include "geners/GenericIO.hh"
#include "reflow/core/utils/eigenGeners.h"
#include "reflow/tree/ContinuationValueTree.h"
#include "reflow/tree/TreeGeners.h"

/** \file ContinuationValueTreeGeners.h
 * \brief Define non intrusive serialization with random access
*  \author Xavier Warin
 */

/// specialize the ClassIdSpecialization template
/// so that a ClassId object can be associated with the class we want to
/// serialize.  The second argument is the version number.
///@{
gs_specialize_class_id(reflow::ContinuationValueTree, 1)
/// an external class
gs_declare_type_external(reflow::ContinuationValueTree)
///@}

namespace gs
{
//
/// \brief  This is how the specialization of GenericWriter should look like
//
template <class Stream, class State >
struct GenericWriter < Stream, State, reflow::ContinuationValueTree,
           Int2Type<IOTraits<int>::ISEXTERNAL> >
{
    inline static bool process(const reflow::ContinuationValueTree  &p_contTree, Stream &p_os,
                               State *, const bool p_processClassId)
    {
        // If necessary, serialize the class id
        static const ClassId current(ClassId::makeId<reflow::ContinuationValueTree >());
        const bool status = p_processClassId ? ClassId::makeId<reflow::ContinuationValueTree >().write(p_os) : true;
        // Serialize object data if the class id was successfully
        // written out
        if (status)
        {
            std::shared_ptr< reflow::SpaceGrid > ptrGrid = p_contTree.getGrid();
            bool bSharedPtr = (ptrGrid ? true : false);
            write_pod(p_os, bSharedPtr);
            if (bSharedPtr)
                write_item(p_os, *p_contTree.getGrid());
            write_item(p_os, p_contTree.getValues());
        }
        // Return "true" on success, "false" on failure
        return status && !p_os.fail();
    }
};

/// \brief  And this is the specialization of GenericReader
//
template <class Stream, class State  >
struct GenericReader < Stream, State, reflow::ContinuationValueTree, Int2Type<IOTraits<int>::ISEXTERNAL> >
{
    inline static bool readIntoPtr(reflow::ContinuationValueTree  *&ptr, Stream &p_is,
                                   State *p_st, const bool p_processClassId)
    {

        if (p_processClassId)
        {
            static const ClassId current(ClassId::makeId<reflow::ContinuationValueTree>());
            ClassId id(p_is, 1);
            current.ensureSameName(id);
        }

        /* // Deserialize object data. */
        bool bSharedPtr ;
        read_pod(p_is, &bSharedPtr);
        std::unique_ptr<reflow::SpaceGrid> pgrid ;
        if (bSharedPtr)
            pgrid  = read_item<reflow::SpaceGrid>(p_is);
        std::shared_ptr<reflow::SpaceGrid > pgridShared(std::move(pgrid));
        std::unique_ptr<Eigen::ArrayXXd> pvalues = read_item<Eigen::ArrayXXd>(p_is);
        if (p_is.fail())
            // Return "false" on failure
            return false;
        //Build the object from the stored data
        if (ptr)
            *ptr = reflow::ContinuationValueTree();
        else
            ptr = new  reflow::ContinuationValueTree();
        ptr->loadForSimulation(pgridShared,  *pvalues);
        return true;
    }

    inline static bool process(reflow::ContinuationValueTree &s, Stream &is,
                               State *st, const bool p_processClassId)
    {
        // Simply convert reading by reference into reading by pointer
        reflow::ContinuationValueTree *ps = &s;
        return readIntoPtr(ps, is, st, p_processClassId);
    }
};
}



#endif/*  CONTINUATIONVALUETREEGENERS_H */
