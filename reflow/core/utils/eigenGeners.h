#ifndef EIGENGENERS_H_
#define EIGENGENERS_H_
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "geners/GenericIO.hh"
#include "geners/vectorIO.hh"

namespace gs
{
//
// Let the system know that the serialization is performed
// by a user-defined external mechanism
//
template<typename T0, int N1, int N2, int N3, int N4, int N5>
struct IOIsExternal<Eigen::Array<T0, N1, N2, N3, N4, N5> >
{
    enum {value = 1};
};
template<typename T0, int N1, int N2, int N3, int N4, int N5>
struct IOIsExternal<const Eigen::Array<T0, N1, N2, N3, N4, N5> >
{
    enum {value = 1};
};
template<typename T0, int N1, int N2, int N3, int N4, int N5>
struct IOIsExternal<volatile Eigen::Array<T0, N1, N2, N3, N4, N5> >
{
    enum {value = 1};
};
template<typename T0, int N1, int N2, int N3, int N4, int N5>
struct IOIsExternal<const volatile Eigen::Array<T0, N1, N2, N3, N4, N5> >
{
    enum {value = 1};
};

template<typename T0, int N1, int N2, int N3, int N4, int N5>
inline std::string template_class_name_Eigen__Array(
    const std::string &tN, const unsigned nIncl)
{
    return template_class_name_Eigen__Array<T0, N1, N2, N3, N4, N5>(tN.c_str(), nIncl);
}

//
// The following function generates the template class name
// for serialization purposes
//
template<typename T0, int N1, int N2, int N3, int N4, int N5>
inline std::string template_class_name_Eigen__Array(
    const char *templateName, const unsigned nIncl)
{
    assert(templateName);
    std::string name(templateName);
    if (nIncl)
    {
        name += '<';
        {
            const ClassId &id1(ClassIdSpecialization<T0>::classId());
            name += id1.id();
        }
        if (nIncl > 1)
        {
            name += ',';
            std::ostringstream os;
            os << N1 << "(0)";
            name += os.str();
        }
        if (nIncl > 2)
        {
            name += ',';
            std::ostringstream os;
            os << N2 << "(0)";
            name += os.str();
        }
        if (nIncl > 3)
        {
            name += ',';
            std::ostringstream os;
            os << N3 << "(0)";
            name += os.str();
        }
        if (nIncl > 4)
        {
            name += ',';
            std::ostringstream os;
            os << N4 << "(0)";
            name += os.str();
        }
        if (nIncl > 5)
        {
            name += ',';
            std::ostringstream os;
            os << N5 << "(0)";
            name += os.str();
        }
        name += '>';
    }
    return name;
}


//
// Let the system know that the serialization is performed
// by a user-defined external mechanism
//
template<typename T0, int N1, int N2, int N3, int N4, int N5>
struct IOIsExternal<Eigen::Matrix<T0, N1, N2, N3, N4, N5> >
{
    enum {value = 1};
};
template<typename T0, int N1, int N2, int N3, int N4, int N5>
struct IOIsExternal<const Eigen::Matrix<T0, N1, N2, N3, N4, N5> >
{
    enum {value = 1};
};
template<typename T0, int N1, int N2, int N3, int N4, int N5>
struct IOIsExternal<volatile Eigen::Matrix<T0, N1, N2, N3, N4, N5> >
{
    enum {value = 1};
};
template<typename T0, int N1, int N2, int N3, int N4, int N5>
struct IOIsExternal<const volatile Eigen::Matrix<T0, N1, N2, N3, N4, N5> >
{
    enum {value = 1};
};

template<typename T0, int N1, int N2, int N3, int N4, int N5>
inline std::string template_class_name_Eigen__Matrix(
    const std::string &tN, const unsigned nIncl)
{
    return template_class_name_Eigen__Matrix<T0, N1, N2, N3, N4, N5>(tN.c_str(), nIncl);
}

//
// The following function generates the template class name
// for serialization purposes
//
template<typename T0, int N1, int N2, int N3, int N4, int N5>
inline std::string template_class_name_Eigen__Matrix(
    const char *templateName, const unsigned nIncl)
{
    assert(templateName);
    std::string name(templateName);
    if (nIncl)
    {
        name += '<';
        {
            const ClassId &id1(ClassIdSpecialization<T0>::classId());
            name += id1.id();
        }
        if (nIncl > 1)
        {
            name += ',';
            std::ostringstream os;
            os << N1 << "(0)";
            name += os.str();
        }
        if (nIncl > 2)
        {
            name += ',';
            std::ostringstream os;
            os << N2 << "(0)";
            name += os.str();
        }
        if (nIncl > 3)
        {
            name += ',';
            std::ostringstream os;
            os << N3 << "(0)";
            name += os.str();
        }
        if (nIncl > 4)
        {
            name += ',';
            std::ostringstream os;
            os << N4 << "(0)";
            name += os.str();
        }
        if (nIncl > 5)
        {
            name += ',';
            std::ostringstream os;
            os << N5 << "(0)";
            name += os.str();
        }
        name += '>';
    }
    return name;
}


//
// Let the system know that the serialization is performed
// by a user-defined external mechanism
//
template<typename T0, int N1>
struct IOIsExternal<Eigen::SparseMatrix<T0, N1> >
{
    enum {value = 1};
};
template<typename T0, int N1>
struct IOIsExternal<const Eigen::SparseMatrix<T0, N1> >
{
    enum {value = 1};
};
template<typename T0, int N1>
struct IOIsExternal<volatile Eigen::SparseMatrix<T0, N1> >
{
    enum {value = 1};
};
template<typename T0, int N1 >
struct IOIsExternal<const volatile Eigen::SparseMatrix<T0, N1> >
{
    enum {value = 1};
};

template<typename T0, int N1>
inline std::string template_class_name_Eigen__Array(
    const std::string &tN, const unsigned nIncl)
{
    return template_class_name_Eigen__Array<T0, N1>(tN.c_str(), nIncl);
}

template<typename T0, int N1>
inline std::string template_class_name_Eigen__Matrix(
    const std::string &tN, const unsigned nIncl)
{
    return template_class_name_Eigen__Matrix<T0, N1>(tN.c_str(), nIncl);
}


//
// The following function generates the template class name
// for serialization purposes
//
template<typename T0, int N1>
inline std::string template_class_name_Eigen__SparseMatrix(
    const char *templateName, const unsigned nIncl)
{
    assert(templateName);
    std::string name(templateName);
    if (nIncl)
    {
        name += '<';
        {
            const ClassId &id1(ClassIdSpecialization<T0>::classId());
            name += id1.id();
        }
        if (nIncl > 1)
        {
            name += ',';
            std::ostringstream os;
            os << N1 << "(0)";
            name += os.str();
        }
        name += '>';
    }
    return name;
}
}


#define gs_specialize_template_hlp_Eigen__Array(qualifyer, name, version, MAX) /**/ \
    template<typename T0,int N1,int N2,int N3,int N4,int N5> \
    struct ClassIdSpecialization<qualifyer name <T0,N1,N2,N3,N4,N5> > \
    {inline static ClassId classId(const bool isPtr=false) \
    {return ClassId(template_class_name_Eigen__Array<T0,N1,N2,N3,N4,N5>(#name,MAX), version, isPtr);}};

#define gs_specialize_template_id_Eigen__Array(name, version, MAX) /**/ \
namespace gs { \
    gs_specialize_template_hlp_Eigen__Array(GENERS_EMPTY_TYPE_QUALIFYER_, name, version, MAX) \
    gs_specialize_template_hlp_Eigen__Array(const, name, version, MAX) \
    gs_specialize_template_hlp_Eigen__Array(volatile, name, version, MAX) \
    gs_specialize_template_hlp_Eigen__Array(const volatile, name, version, MAX) \
}

#define gs_specialize_template_hlp_Eigen__Matrix(qualifyer, name, version, MAX) /**/ \
    template<typename T0,int N1,int N2,int N3,int N4,int N5> \
    struct ClassIdSpecialization<qualifyer name <T0,N1,N2,N3,N4,N5> > \
    {inline static ClassId classId(const bool isPtr=false) \
    {return ClassId(template_class_name_Eigen__Matrix<T0,N1,N2,N3,N4,N5>(#name,MAX), version, isPtr);}};

#define gs_specialize_template_id_Eigen__Matrix(name, version, MAX) /**/ \
namespace gs { \
    gs_specialize_template_hlp_Eigen__Matrix(GENERS_EMPTY_TYPE_QUALIFYER_, name, version, MAX) \
    gs_specialize_template_hlp_Eigen__Matrix(const, name, version, MAX) \
    gs_specialize_template_hlp_Eigen__Matrix(volatile, name, version, MAX) \
    gs_specialize_template_hlp_Eigen__Matrix(const volatile, name, version, MAX) \
}
#define gs_specialize_template_hlp_Eigen__SparseMatrix(qualifyer, name, version, MAX) /**/ \
    template<typename T0,int N1> \
    struct ClassIdSpecialization<qualifyer name <T0,N1> > \
    {inline static ClassId classId(const bool isPtr=false) \
    {return ClassId(template_class_name_Eigen__SparseMatrix<T0,N1>(#name,MAX), version, isPtr);}};

#define gs_specialize_template_id_Eigen__SparseMatrix(name, version, MAX) /**/ \
namespace gs { \
    gs_specialize_template_hlp_Eigen__SparseMatrix(GENERS_EMPTY_TYPE_QUALIFYER_, name, version, MAX) \
    gs_specialize_template_hlp_Eigen__SparseMatrix(const, name, version, MAX) \
    gs_specialize_template_hlp_Eigen__SparseMatrix(volatile, name, version, MAX) \
    gs_specialize_template_hlp_Eigen__SparseMatrix(const volatile, name, version, MAX) \
}

//
// Actual specialization of the template class id
//
gs_specialize_template_id_Eigen__Array(Eigen::Array, 1, 3)
//
// Actual specialization of the template class id
//
gs_specialize_template_id_Eigen__Matrix(Eigen::Matrix, 1, 3)
//
// Actual specialization of the template class id
//
gs_specialize_template_id_Eigen__SparseMatrix(Eigen::SparseMatrix, 1, 1)

//
// Specialize the behavior of the two template classes at the heart of
// the serialization facility: gs::GenericWriter and gs::GenericReader
//
namespace gs
{
template <class Stream, class State, typename T0, int N1, int N2, int N3, int N4, int N5>
struct GenericWriter<Stream, State, Eigen::Array<T0, N1, N2, N3, N4, N5>,
           Int2Type<IOTraits<int>::ISEXTERNAL> >
{
    inline static bool process(const Eigen::Array<T0, N1, N2, N3, N4, N5> &p_array, Stream &p_os,
                               State *, const bool processClassId)
    {
        // If necessary, serialize the class id
        static const ClassId current(ClassId::makeId<Eigen::Array<T0, N1, N2, N3, N4, N5> >());
        bool status = processClassId ? current.write(p_os) : true;

        // Serialize object data if the class id was successfully written out
        if (status)
        {
            int isizeRows = p_array.rows();
            int isizeCols = p_array.cols();
            write_pod(p_os, isizeRows);
            write_pod(p_os, isizeCols);
            write_pod_array(p_os, p_array. data(), isizeRows * isizeCols);
        }

        // Return "true" on success, "false" on failure
        return status && !p_os.fail();
    }
};

template <class Stream, class State, typename T0, int N1, int N2, int N3, int N4, int N5>
struct GenericReader<Stream, State, Eigen::Array<T0, N1, N2, N3, N4, N5>,
           Int2Type<IOTraits<int>::ISEXTERNAL> >
{
    inline static bool readIntoPtr(Eigen::Array<T0, N1, N2, N3, N4, N5> *&ptr, Stream &p_is,
                                   State *st, const bool processClassId)
    {
        // Make sure that the serialized class id is consistent with
        // the current one
        static const ClassId current(ClassId::makeId<Eigen::Array<T0, N1, N2, N3, N4, N5> >());
        const ClassId &stored = processClassId ? ClassId(p_is, 1) : st->back();

        // Check that the name is consistent. Do not check for the
        // consistency of the complete id because we want to be able
        // to read different versions of parameter classes.
        current.ensureSameName(stored);

        int isizeRows = 0;
        read_pod(p_is, &isizeRows);
        int isizeCols = 0;
        read_pod(p_is, &isizeCols);
        int isizeLoc = isizeRows * isizeCols;
        Eigen::Array< T0, N1, N2, N3, N4, N5> values(isizeRows, isizeCols);
        read_pod_array(p_is, values.data(), isizeLoc);

        if (ptr == 0)
            ptr = new Eigen::Array<T0, N1, N2, N3, N4, N5>(values);
        else
            *ptr = values;
        return true;
    }

    inline static bool process(Eigen::Array<T0, N1, N2, N3, N4, N5> &s, Stream &p_is,
                               State *st, const bool processClassId)
    {
        // Simply convert reading by reference into reading by pointer
        Eigen::Array<T0, N1, N2, N3, N4, N5> *ps = &s;
        return readIntoPtr(ps, p_is, st, processClassId);
    }
};

template <class Stream, class State, typename T0, int N1, int N2, int N3, int N4, int N5>
struct GenericWriter<Stream, State, Eigen::Matrix<T0, N1, N2, N3, N4, N5>,
           Int2Type<IOTraits<int>::ISEXTERNAL> >
{
    inline static bool process(const Eigen::Matrix<T0, N1, N2, N3, N4, N5> &p_array, Stream &p_os,
                               State *, const bool processClassId)
    {
        // If necessary, serialize the class id
        static const ClassId current(ClassId::makeId<Eigen::Matrix<T0, N1, N2, N3, N4, N5> >());
        bool status = processClassId ? current.write(p_os) : true;

        // Serialize object data if the class id was successfully written out
        if (status)
        {
            int isizeRows = p_array.rows();
            int isizeCols = p_array.cols();
            write_pod(p_os, isizeRows);
            write_pod(p_os, isizeCols);
            write_pod_array(p_os, p_array. data(), isizeRows * isizeCols);
        }

        // Return "true" on success, "false" on failure
        return status && !p_os.fail();
    }
};

template <class Stream, class State, typename T0, int N1, int N2, int N3, int N4, int N5>
struct GenericReader<Stream, State, Eigen::Matrix<T0, N1, N2, N3, N4, N5>,
           Int2Type<IOTraits<int>::ISEXTERNAL> >
{
    inline static bool readIntoPtr(Eigen::Matrix<T0, N1, N2, N3, N4, N5> *&ptr, Stream &p_is,
                                   State *st, const bool processClassId)
    {
        // Make sure that the serialized class id is consistent with
        // the current one
        static const ClassId current(ClassId::makeId<Eigen::Matrix<T0, N1, N2, N3, N4, N5> >());
        const ClassId &stored = processClassId ? ClassId(p_is, 1) : st->back();

        // Check that the name is consistent. Do not check for the
        // consistency of the complete id because we want to be able
        // to read different versions of parameter classes.
        current.ensureSameName(stored);

        int isizeRows = 0;
        read_pod(p_is, &isizeRows);
        int isizeCols = 0;
        read_pod(p_is, &isizeCols);
        int isizeLoc = isizeRows * isizeCols;
        Eigen::Matrix< T0, N1, N2, N3, N4, N5> values(isizeRows, isizeCols);
        read_pod_array(p_is, values.data(), isizeLoc);

        if (ptr == 0)
            ptr = new Eigen::Matrix<T0, N1, N2, N3, N4, N5>(values);
        else
            *ptr = values;
        return true;
    }

    inline static bool process(Eigen::Matrix<T0, N1, N2, N3, N4, N5> &s, Stream &p_is,
                               State *st, const bool processClassId)
    {
        // Simply convert reading by reference into reading by pointer
        Eigen::Matrix<T0, N1, N2, N3, N4, N5> *ps = &s;
        return readIntoPtr(ps, p_is, st, processClassId);
    }
};

template <class Stream, class State, typename T0, int N1>
struct GenericWriter<Stream, State, Eigen::SparseMatrix<T0, N1>,
           Int2Type<IOTraits<int>::ISEXTERNAL> >
{
    inline static bool process(const Eigen::SparseMatrix<T0, N1> &p_matrix, Stream &p_os,
                               State *, const bool processClassId)
    {
        // If necessary, serialize the class id
        static const ClassId current(ClassId::makeId<Eigen::SparseMatrix<T0, N1> >());
        bool status = processClassId ? current.write(p_os) : true;

        // Serialize object data if the class id was successfully written out
        if (status)
        {
            int innerSize = p_matrix.innerSize();
            int outerSize = p_matrix.outerSize();
            std::vector<int> row, col;
            std::vector<T0> value;
            for (int i = 0; i < outerSize; ++i)
            {
                for (typename Eigen::SparseMatrix<T0, N1>::InnerIterator it(p_matrix, i); it; ++it)
                {
                    row.push_back(it.row());
                    col.push_back(it.col());
                    value.push_back(it.value());
                }
            }
            write_pod(p_os, innerSize);
            write_pod(p_os, outerSize);
            write_item(p_os, row);
            write_item(p_os, col);
            write_item(p_os, value);
        }

        // Return "true" on success, "false" on failure
        return status && !p_os.fail();
    }
};

template <class Stream, class State, typename T0, int N1>
struct GenericReader<Stream, State, Eigen::SparseMatrix<T0, N1>,
           Int2Type<IOTraits<int>::ISEXTERNAL> >
{
    inline static bool readIntoPtr(Eigen::SparseMatrix<T0, N1> *&ptr, Stream &p_is,
                                   State *st, const bool processClassId)
    {
        // Make sure that the serialized class id is consistent with
        // the current one
        static const ClassId current(ClassId::makeId<Eigen::SparseMatrix<T0, N1> >());
        const ClassId &stored = processClassId ? ClassId(p_is, 1) : st->back();

        // Check that the name is consistent. Do not check for the
        // consistency of the complete id because we want to be able
        // to read different versions of parameter classes.
        current.ensureSameName(stored);

        int innerSize = 0;
        read_pod(p_is, &innerSize);
        int outerSize = 0;
        read_pod(p_is, &outerSize);
        int rows = (N1 == 1) ? outerSize : innerSize;
        int cols = (N1 == 1) ? innerSize : outerSize;
        Eigen::SparseMatrix< T0, N1> mat(rows, cols);
        std::unique_ptr< std::vector<int> > row = read_item<std::vector<int> >(p_is);
        std::unique_ptr< std::vector<int> > col = read_item<std::vector<int> >(p_is);
        std::unique_ptr< std::vector<T0> > value = read_item<std::vector< T0> >(p_is);
        typedef typename Eigen::Triplet<T0> T;
        std::vector<T> triplets;
        triplets.reserve(row->size());
        for (int is = 0 ; is < row->size(); ++is)
            triplets.push_back(T((*row)[is], (*col)[is], (*value)[is]));
        mat.setFromTriplets(triplets.begin(), triplets.end());
        if (ptr == 0)
            ptr = new Eigen::SparseMatrix<T0, N1>(mat);
        else
            *ptr = mat;
        return true;
    }

    inline static bool process(Eigen::SparseMatrix<T0, N1> &s, Stream &p_is,
                               State *st, const bool processClassId)
    {
        // Simply convert reading by reference into reading by pointer
        Eigen::SparseMatrix<T0, N1> *ps = &s;
        return readIntoPtr(ps, p_is, st, processClassId);
    }
};
}

#endif /* EIGENGENERS_H_ */
