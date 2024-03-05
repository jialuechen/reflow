
#ifndef EIGENSERIALIZATION_H
#define EIGENSERIALIZATION_H
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <Eigen/Dense>

/** \file eigenSerialization.h
 *  \brief Boost serialization of eigen object of type matrix_type (matrix or array)
 *  \author Xavier Warin
 */

namespace boost
{
namespace serialization
{


/// \brief save the object
/// \param ar archive used
/// \param g object to load
template <   class Archive,
             class S,
             int Rows_,
             int Cols_,
             int Ops_,
             int MaxRows_,
             int MaxCols_ >
inline void save(Archive &ar,
                 const Eigen::Array<S, Rows_, Cols_, Ops_, MaxRows_, MaxCols_> &g,
                 const unsigned int)
{
    int rows = g.rows();
    int cols = g.cols();
    ar &rows;
    ar &cols;
    ar &boost::serialization::make_array(g.data(), rows * cols);
}

/// \brief load the object
/// \param ar archive used
/// \param g object to load
template <   class Archive,
             class S,
             int Rows_,
             int Cols_,
             int Ops_,
             int MaxRows_,
             int MaxCols_ >
inline void load(
    Archive &ar,
    Eigen::Array<S, Rows_, Cols_, Ops_, MaxRows_, MaxCols_> &g,
    const unsigned int)
{
    int rows, cols;
    ar &rows;
    ar &cols;
    g.resize(rows, cols);
    ar &boost::serialization::make_array(g.data(), rows * cols);
}

/// \brief serialize the object Eigen
/// \param ar archive used
/// \param g object to load
/// \param version useless
template <   class Archive,
             class S,
             int Rows_,
             int Cols_,
             int Ops_,
             int MaxRows_,
             int MaxCols_ >
inline void serialize(
    Archive &ar,
    Eigen::Array< S, Rows_, Cols_, Ops_, MaxRows_, MaxCols_> &g,
    const unsigned int version)
{
    split_free(ar, g, version);
}


/// \brief save the object
/// \param ar archive used
/// \param g object to load
template <   class Archive,
             class S,
             int Rows_,
             int Cols_,
             int Ops_,
             int MaxRows_,
             int MaxCols_ >
inline void save(Archive &ar,
                 const Eigen::Matrix<S, Rows_, Cols_, Ops_, MaxRows_, MaxCols_> &g,
                 const unsigned int)
{
    int rows = g.rows();
    int cols = g.cols();
    ar &rows;
    ar &cols;
    ar &boost::serialization::make_array(g.data(), rows * cols);
}

/// \brief load the object
/// \param ar archive used
/// \param g object to load
template <   class Archive,
             class S,
             int Rows_,
             int Cols_,
             int Ops_,
             int MaxRows_,
             int MaxCols_ >
inline void load(
    Archive &ar,
    Eigen::Matrix<S, Rows_, Cols_, Ops_, MaxRows_, MaxCols_> &g,
    const unsigned int)
{
    int rows, cols;
    ar &rows;
    ar &cols;
    g.resize(rows, cols);
    ar &boost::serialization::make_array(g.data(), rows * cols);
}

/// \brief serialize the object Eigen
/// \param ar archive used
/// \param g object to load
/// \param version useless
template <   class Archive,
             class S,
             int Rows_,
             int Cols_,
             int Ops_,
             int MaxRows_,
             int MaxCols_ >
inline void serialize(
    Archive &ar,
    Eigen::Matrix< S, Rows_, Cols_, Ops_, MaxRows_, MaxCols_> &g,
    const unsigned int version)
{
    split_free(ar, g, version);
}

} // namespace serialization
} // namespace boost



#endif /* EIGENSERIALIZATION_H */
