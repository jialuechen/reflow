
#ifndef  CONTINUATIONCUTSTREEGENERS_H
#define  CONTINUATIONCUTSTREEGENERS_H
#include <Eigen/Dense>
#include "geners/GenericIO.hh"
#include "geners/vectorIO.hh"
#include "libflow/tree/ContinuationCutsTree.h"
#include "libflow/tree/TreeGeners.h"
#include "libflow/core/grids/SpaceGridGeners.h"
#include "libflow/core/grids/RegularSpaceGridGeners.h"
#include "libflow/core/grids/GeneralSpaceGridGeners.h"
#include "libflow/core/utils/eigenGeners.h"

/** \file ContinuationCutsTreeGeners.h
 * \brief Define non intrusive serialization with random access
*  \author Xavier Warin
 */

/// specialize the ClassIdSpecialization template
/// so that a ClassId object can be associated with the class we want to
/// serialize.  The second argument is the version number.
///@{
gs_specialize_class_id(libflow::ContinuationCutsTree, 1)
/// an external class
gs_declare_type_external(libflow::ContinuationCutsTree)
///@}

namespace gs
{
//
/// \brief  This is how the specialization of GenericWriter should look like
//
template <class Stream, class State >
struct GenericWriter < Stream, State, libflow::ContinuationCutsTree,
           Int2Type<IOTraits<int>::ISEXTERNAL> >
{
    inline static bool process(const libflow::ContinuationCutsTree  &p_cutTree, Stream &p_os,
                               State *, const bool p_processClassId)
    {
        // If necessary, serialize the class id
        static const ClassId current(ClassId::makeId<libflow::ContinuationCutsTree >());
        const bool status = p_processClassId ? ClassId::makeId<libflow::ContinuationCutsTree >().write(p_os) : true;
        // Serialize object data if the class id was successfully
        // written out
        if (status)
        {
            std::shared_ptr< libflow::SpaceGrid > ptrGrid = p_cutTree.getGrid();
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
struct GenericReader < Stream, State, libflow::ContinuationCutsTree, Int2Type<IOTraits<int>::ISEXTERNAL> >
{
    inline static bool readIntoPtr(libflow::ContinuationCutsTree  *&ptr, Stream &p_is,
                                   State *p_st, const bool p_processClassId)
    {

        if (p_processClassId)
        {
            static const ClassId current(ClassId::makeId<libflow::ContinuationCutsTree>());
            ClassId id(p_is, 1);
            current.ensureSameName(id);
        }

        // Deserialize object data.
        bool bSharedPtr ;
        read_pod(p_is, &bSharedPtr);
        std::unique_ptr<libflow::SpaceGrid> pgrid ;
        if (bSharedPtr)
            pgrid  = read_item<libflow::SpaceGrid>(p_is);
        std::shared_ptr<libflow::SpaceGrid > pgridShared(std::move(pgrid));
        std::unique_ptr< std::vector< Eigen::ArrayXXd > > cutCoeff = read_item<  std::vector< Eigen::ArrayXXd > >(p_is);

        if (p_is.fail())
            // Return "false" on failure
            return false;
        //Build the object from the stored data
        if (ptr)
        {
            *ptr = libflow::ContinuationCutsTree();
            ptr->loadForSimulation(pgridShared, *cutCoeff) ; // values);
            return true;
        }
        return false;
    }

    inline static bool process(libflow::ContinuationCutsTree &s, Stream &is,
                               State *st, const bool p_processClassId)
    {
        // Simply convert reading by reference into reading by pointer
        libflow::ContinuationCutsTree *ps = &s;
        return readIntoPtr(ps, is, st, p_processClassId);
    }
};
}

#endif/*  CONTINUATIONCUTSTREEGENERS_H */
