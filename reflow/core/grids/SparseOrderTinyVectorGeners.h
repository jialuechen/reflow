
#ifndef SPARSETINYVECTORGENERS_H
#define SPARSETINYVECTORGENERS_H
#include "geners/GenericIO.hh"

/** \file SparseOrderTinyVectorGeners.h
 * \brief Define non intrusive  serialization with random access
*  \author Xavier Warin
 */

/// specialize the ClassIdSpecialization template
/// so that a ClassId object can be associated with the class we want to
/// serialize.  The second argument is the version number.
///@{
gs_specialize_template_id_T(OrderTinyVector, 1, 1)
/// an external class
gs_declare_template_external_T(OrderTinyVector)
///@}

namespace gs
{
//
/// \brief  This is how the specialization of GenericWriter should look like
//
template <class Stream, class State, typename T>
struct GenericWriter < Stream, State, OrderTinyVector< T >,
           Int2Type<IOTraits<int>::ISEXTERNAL> >
{
    inline static bool process(const OrderTinyVector<T>  &p_comp, Stream &p_os,
                               State *, const bool p_processClassId)
    {
        // If necessary, serialize the class id
        static const ClassId current(ClassId::makeId<OrderTinyVector< T> >());
        const bool status = p_processClassId ? ClassId::makeId< OrderTinyVector< T>  >().write(p_os) : true;
        // Serialize object data if the class id was successfully
        // Return "true" on success, "false" on failure
        return status && !p_os.fail();
    }
};

/// \brief  And this is the specialization of GenericReader
///
template <class Stream, class State, typename T >
struct GenericReader < Stream, State, OrderTinyVector< T>,
           Int2Type<IOTraits<int>::ISEXTERNAL> >
{
    inline static bool readIntoPtr(OrderTinyVector<T>  *&ptr, Stream &p_is,
                                   State *p_st, const bool p_processClassId)
    {
        // Make sure that the serialized class id is consistent with
        // the current one
        static const ClassId current(ClassId::makeId<OrderTinyVector<T> >());
        const ClassId &stored = p_processClassId ? ClassId(p_is, 1) : p_st->back();
        current.ensureSameId(stored);

        // Build the object from the stored data
        if (ptr)
        {
            *ptr = OrderTinyVector<T>();
        }
        else
        {
            ptr = new  OrderTinyVector<T>();
        }
        return true;
    }

    inline static bool process(OrderTinyVector< T>  &s, Stream &is,
                               State *st, const bool p_processClassId)
    {
        // Simply convert reading by reference into reading by pointer
        OrderTinyVector< T >  *ps = &s;
        return readIntoPtr(ps, is, st, p_processClassId);
    }
};
}
#endif /* SPARSEORDERTINYVECTORGENERS */
