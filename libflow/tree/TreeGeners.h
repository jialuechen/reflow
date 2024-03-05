
#ifndef TREEGENERS_H
#define TREEGENERS_H
#include <Eigen/Dense>
#include "geners/GenericIO.hh"
#include "geners/vectorIO.hh"
#include "geners/arrayIO.hh"
#include "libflow/tree/Tree.h"

/** \file TreeGeners.h
 * \file Define serialization with geners
*  \author Xavier Warin
*/

/// specialize the ClassIdSpecialization template
/// so that a ClassId object can be associated with the class we want to
/// serialize.  The second argument is the version number.
///@{
gs_specialize_class_id(libflow::Tree, 1)
/// an external class
gs_declare_type_external(libflow::Tree)
///@}

namespace gs
{
//
/// \brief  This is how the specialization of GenericWriter should look like
//
template <class Stream, class State >
struct GenericWriter < Stream, State, libflow::Tree,
           Int2Type<IOTraits<int>::ISEXTERNAL> >
{
    inline static bool process(const libflow::Tree  &p_tree, Stream &p_os,
                               State *, const bool p_processClassId)
    {
        // If necessary, serialize the class id
        static const ClassId current(ClassId::makeId<libflow::Tree >());
        const bool status = p_processClassId ? ClassId::makeId<libflow::Tree >().write(p_os) : true;

        // Serialize object data if the class id was successfully
        // written out
        if (status)
        {
            gs::write_item(p_os, p_tree.getProba());
            gs::write_item(p_os, p_tree.getConnected());
        }
        // Return "true" on success, "false" on failure
        return status && !p_os.fail();
    }
};

/// \brief  And this is the specialization of GenericReader
//
template <class Stream, class State  >
struct GenericReader < Stream, State, libflow::Tree, Int2Type<IOTraits<int>::ISEXTERNAL> >
{
    inline static bool readIntoPtr(libflow::Tree  *&ptr, Stream &p_is,
                                   State *p_st, const bool p_processClassId)
    {

        if (p_processClassId)
        {
            static const ClassId current(ClassId::makeId<libflow::Tree>());
            ClassId id(p_is, 1);
            current.ensureSameName(id);
        }

        // Deserialize object data.
        std::unique_ptr<std::vector< double > > proba =   gs::read_item< std::vector< double >  >(p_is);
        std::unique_ptr<std::vector< std::vector<std::array<int, 2> > > > connected = gs::read_item< std::vector< std::vector<std::array<int, 2> > > >(p_is);
        //Build the object from the stored data
        if (ptr)
        {
            *ptr = libflow::Tree();
            ptr->update(*proba, *connected);
        }
        else
            ptr = new  libflow::Tree(*proba, *connected);
        return true;
    }

    inline static bool process(libflow::Tree &s, Stream &is,
                               State *st, const bool p_processClassId)
    {
        // Simply convert reading by reference into reading by pointer
        libflow::Tree *ps = &s;
        return readIntoPtr(ps, is, st, p_processClassId);
    }
};
}
#endif
