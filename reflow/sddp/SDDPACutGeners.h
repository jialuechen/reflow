
#ifndef SDDPACUTGENERS_H
#define SDDPACUTGENERS_H
#include <Eigen/Dense>
#include "geners/GenericIO.hh"
#include "reflow/sddp/SDDPACut.h"
#include "reflow/core/utils/eigenGeners.h"

/** \file SDDPACutGeners.h
 *  \brief A cut for SDDP
 *  \author Xavier Warin
 */

/// specialize the ClassIdSpecialization template
/// so that a ClassId object can be associated with the class we want to
/// serialize.  The second argument is the version number.
///@{
gs_specialize_class_id(reflow::SDDPACut, 1)
gs_declare_type_external(reflow::SDDPACut)
///@}
namespace gs
{
//
// This is how the specialization of GenericWriter should look like
//
template <class Stream, class State>
struct GenericWriter < Stream, State, reflow::SDDPACut,   Int2Type<IOTraits<int>::ISEXTERNAL> >
{
    inline static bool process(const reflow::SDDPACut &p_cut, Stream &p_os,
                               State *, const bool p_processClassId)
    {
        // If necessary, serialize the class id
        static const ClassId current(ClassId::makeId<reflow::SDDPACut>());
        const bool status = p_processClassId ? current.write(p_os) : true;
        // Serialize object data if the class id was successfully
        // written out
        if (status)
        {
            write_item(p_os, *p_cut.getCut());
        }
        // Return "true" on success, "false" on failure
        return status && !p_os.fail();
    }
};

// And this is the specialization of GenericReader
//
template <class Stream, class State>
struct GenericReader < Stream, State, reflow::SDDPACut,   Int2Type<IOTraits<int>::ISEXTERNAL> >
{
    inline static bool readIntoPtr(reflow::SDDPACut *&ptr, Stream &p_is,
                                   State *p_st, const bool p_processClassId)
    {
        // Make sure that the serialized class id is consistent with
        // the current one
        static const ClassId current(ClassId::makeId<reflow::SDDPACut>());
        const ClassId &stored = p_processClassId ? ClassId(p_is, 1) : p_st->back();
        current.ensureSameId(stored);

        // Deserialize object data.
        std::shared_ptr< Eigen::ArrayXXd>  cut(std::move(read_item< Eigen::ArrayXXd>(p_is)));

        if (p_is.fail())
            // Return "false" on failure
            return false;
        // Build the object from the stored data
        if (ptr)
            *ptr = reflow::SDDPACut(cut);
        else
            ptr = new  reflow::SDDPACut(cut);
        return true;
    }

    inline static bool process(reflow::SDDPACut &s, Stream &is,
                               State *st, const bool p_processClassId)
    {
        // Simply convert reading by reference into reading by pointer
        reflow::SDDPACut *ps = &s;
        return readIntoPtr(ps, is, st, p_processClassId);
    }
};
}

#endif /* SDDPACUTGENERS_H */
