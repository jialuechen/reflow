
#ifndef  CONTINUATIONCUTSTREEGENERS_H
#define  CONTINUATIONCUTSTREEGENERS_H
#include <Eigen/Dense>
#include "geners/GenericIO.hh"
#include "geners/vectorIO.hh"
#include "reflow/tree/ContinuationCutsTree.h"
#include "reflow/tree/TreeGeners.h"
#include "reflow/core/grids/SpaceGridGeners.h"
#include "reflow/core/grids/RegularSpaceGridGeners.h"
#include "reflow/core/grids/GeneralSpaceGridGeners.h"
#include "reflow/core/utils/eigenGeners.h"

/** \file ContinuationCutsTreeGeners.h
 * \brief Define non intrusive serialization with random access
*  \author Xavier Warin
 */

/// specialize the ClassIdSpecialization template
/// so that a ClassId object can be associated with the class we want to
/// serialize.  The second argument is the version number.
///@{
gs_specialize_class_id(reflow::ContinuationCutsTree, 1)
/// an external class
gs_declare_type_external(reflow::ContinuationCutsTree)
///@}

namespace gs
{
//
/// \brief  This is how the specialization of GenericWriter should look like
//
template <class Stream, class State >
struct GenericWriter < Stream, State, reflow::ContinuationCutsTree,
           Int2Type<IOTraits<int>::ISEXTERNAL> >
{
    inline static bool process(const reflow::ContinuationCutsTree  &p_cutTree, Stream &p_os,
                               State *, const bool p_processClassId)
    {
        // If necessary, serialize the class id
        static const ClassId current(ClassId::makeId<reflow::ContinuationCutsTree >());
        const bool status = p_processClassId ? ClassId::makeId<reflow::ContinuationCutsTree >().write(p_os) : true;
        // Serialize object data if the class id was successfully
        // written out
        if (status)
        {
            std::shared_ptr< reflow::SpaceGrid > ptrGrid = p_cutTree.getGrid();
            bool bSharedPtr = (ptrGrid ? true : false);
            write_pod(p_os, bSharedPtr);
            if (bSharedPtr)
                write_item(p_os, *p_cutTree.getGrid());
            write_item(p_os, p_cutTree.getValues());
        }
        // Return "true" on success, "false" on failure
        return status && !p_os.fail();
    }
};

/// \brief  And this is the specialization of GenericReader
//
template <class Stream, class State  >
struct GenericReader < Stream, State, reflow::ContinuationCutsTree, Int2Type<IOTraits<int>::ISEXTERNAL> >
{
    inline static bool readIntoPtr(reflow::ContinuationCutsTree  *&ptr, Stream &p_is,
                                   State *p_st, const bool p_processClassId)
    {

        if (p_processClassId)
        {
            static const ClassId current(ClassId::makeId<reflow::ContinuationCutsTree>());
            ClassId id(p_is, 1);
            current.ensureSameName(id);
        }

        // Deserialize object data.
        bool bSharedPtr ;
        read_pod(p_is, &bSharedPtr);
        std::unique_ptr<reflow::SpaceGrid> pgrid ;
        if (bSharedPtr)
            pgrid  = read_item<reflow::SpaceGrid>(p_is);
        std::shared_ptr<reflow::SpaceGrid > pgridShared(std::move(pgrid));
        std::unique_ptr< std::vector< Eigen::ArrayXXd > > cutCoeff = read_item<  std::vector< Eigen::ArrayXXd > >(p_is);

        if (p_is.fail())
            // Return "false" on failure
            return false;
        //Build the object from the stored data
        if (ptr)
        {
            *ptr = reflow::ContinuationCutsTree();
            ptr->loadForSimulation(pgridShared, *cutCoeff) ; // values);
            return true;
        }
        return false;
    }

    inline static bool process(reflow::ContinuationCutsTree &s, Stream &is,
                               State *st, const bool p_processClassId)
    {
        // Simply convert reading by reference into reading by pointer
        reflow::ContinuationCutsTree *ps = &s;
        return readIntoPtr(ps, is, st, p_processClassId);
    }
};
}

#endif/*  CONTINUATIONCUTSTREEGENERS_H */
