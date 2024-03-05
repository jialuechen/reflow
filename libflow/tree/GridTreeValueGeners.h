
#ifndef  GRIDTREEVALUEGENERS_H
#define  GRIDTREEVALUEGENERS_H
#include <Eigen/Dense>
#include "geners/GenericIO.hh"
#include "libflow/tree/GridTreeValue.h"
#include "libflow/core/grids/SpaceGridGeners.h"
#include "libflow/core/grids/RegularSpaceGridGeners.h"
#include "libflow/core/grids/GeneralSpaceGridGeners.h"
#include "libflow/core/grids/SparseSpaceGridNoBoundGeners.h"
#include "libflow/core/grids/SparseSpaceGridBoundGeners.h"
#include "libflow/core/grids/SparseInterpolatorSpectralGeners.h"
#include "libflow/core/grids/InterpolatorSpectralGeners.h"
#include "libflow/core/grids/LinearInterpolatorSpectralGeners.h"
#include "libflow/core/grids/LegendreInterpolatorSpectralGeners.h"


/** \file GridTreeValueGeners.h
 * \brief Define non intrusive serialization with random access
*  \author Xavier Warin
 */

/// specialize the ClassIdSpecialization template
/// so that a ClassId object can be associated with the class we want to
/// serialize.  The second argument is the version number.
///@{
gs_specialize_class_id(libflow::GridTreeValue, 1)
/// an external class
gs_declare_type_external(libflow::GridTreeValue)
///@}

namespace gs
{
//
/// \brief  This is how the specialization of GenericWriter should look like
//
template <class Stream, class State >
struct GenericWriter < Stream, State, libflow::GridTreeValue,
           Int2Type<IOTraits<int>::ISEXTERNAL> >
{
    inline static bool process(const libflow::GridTreeValue  &p_state, Stream &p_os,
                               State *, const bool p_processClassId)
    {
        // If necessary, serialize the class id
        static const ClassId current(ClassId::makeId<libflow::GridTreeValue >());
        const bool status = p_processClassId ? ClassId::makeId<libflow::GridTreeValue >().write(p_os) : true;
        // Serialize object data if the class id was successfully
        // written out
        if (status)
        {
            std::shared_ptr< libflow::SpaceGrid > ptrGrid = p_state.getGrid();
            bool bSharedPtr = (ptrGrid ? true : false);
            write_pod(p_os, bSharedPtr);
            if (bSharedPtr)
            {
                write_item(p_os, ptrGrid);
                write_item(p_os, p_state.getInterpolators());
            }
        }
        // Return "true" on success, "false" on failure
        return status && !p_os.fail();
    }
};

/// \brief  And this is the specialization of GenericReader
//
template <class Stream, class State  >
struct GenericReader < Stream, State, libflow::GridTreeValue, Int2Type<IOTraits<int>::ISEXTERNAL> >
{
    inline static bool readIntoPtr(libflow::GridTreeValue  *&ptr, Stream &p_is,
                                   State *, const bool p_processClassId)
    {

        if (p_processClassId)
        {
            static const ClassId current(ClassId::makeId<libflow::GridTreeValue>());
            ClassId id(p_is, 1);
            current.ensureSameName(id);
        }

        // Deserialize object data.
        bool bSharedPtr ;
        read_pod(p_is, &bSharedPtr);
        std::shared_ptr<libflow::SpaceGrid > pgridShared;
        CPP11_auto_ptr<std::vector< std::shared_ptr<libflow::InterpolatorSpectral> > > pinterp;
        if (bSharedPtr)
        {
            CPP11_auto_ptr<libflow::SpaceGrid> pgrid = read_item<libflow::SpaceGrid>(p_is);
            pgridShared = std::move(pgrid);
            pinterp = read_item< std::vector< std::shared_ptr<libflow::InterpolatorSpectral> > >(p_is);
            /// now affect grids to interpolator
            for (size_t i = 0 ; i < pinterp->size(); ++i)
                (*pinterp)[i]->setGrid(& *pgridShared);
        }

        if (p_is.fail())
            // Return "false" on failure
            return false;
        //Build the object from the stored data
        if (ptr)
        {
            if (bSharedPtr)
                *ptr = libflow::GridTreeValue(pgridShared,  *pinterp);
            else
                *ptr = libflow::GridTreeValue();
        }
        else
        {
            if (bSharedPtr)
                ptr = new  libflow::GridTreeValue(pgridShared,  *pinterp);
            else
                ptr = new  libflow::GridTreeValue();
        }
        return true;
    }

    inline static bool process(libflow::GridTreeValue &s, Stream &is,
                               State *st, const bool p_processClassId)
    {
        // Simply convert reading by reference into reading by pointer
        libflow::GridTreeValue *ps = &s;
        return readIntoPtr(ps, is, st, p_processClassId);
    }
};
}



#endif/*  GRIDANDREGRESSEDVALUEGENERS_H */
