
#ifndef SPARSEGRIDTYPES_H
#define SPARSEGRIDTYPES_H
#include <map>
#include <Eigen/Dense>

/* \file sparseGridTypes.h
 *  \brief defines some complicated types to lighten the source code for sparse grids.
 *         - The sparse grid is defined by levels in each dimension.
 *         -  For each (multi-dimensional level), a set of points are given :
 *            Each point is defines as follows
 *              -# the position of the point  in each dimension at the given level is given
 *              -# a number permitting to identify the point in the global grid  is given
 *  \author Xavier Warin
 */
namespace Eigen
{
/// \brief Defines an Eigen array of char
typedef Array<char, Eigen::Dynamic, 1>  ArrayXc;
/// \brief Defines an Eigen array of unsigned int
typedef Array<unsigned int, Eigen::Dynamic, 1>  ArrayXui;
}

/// \class OrderTinyVector sparseGridTypes.h
/// Permit to sort sparse grid levels and sparse grids points inside a level in the map structure
template< typename T>
class OrderTinyVector
{
public:
    /// \brief comparison operator
    bool operator()(const   Eigen::Array<T, Eigen::Dynamic, 1>   &firstVector, const Eigen::Array<T, Eigen::Dynamic, 1>   &secondVector) const
    {
        for (int i = 0 ; i < firstVector.size(); ++i)
        {
            if (firstVector(i) < secondVector(i))
            {
                return true;
            }
            else
            {
                if (firstVector(i) > secondVector(i))
                    return false ;
            }
        }
        return false;
    }
};


/// \brief At given level defines the set of point belonging to this level : the map is indexed by multidimensional position of the point
#define    SparseLevel std::map<  Eigen::Array<unsigned int,Eigen::Dynamic,1> , size_t, OrderTinyVector<unsigned int  > >
/// \brief the whole sparse grid structure : the map is indexed by multidimensional levels
#define     SparseSet std::map<  Eigen::Array<char,Eigen::Dynamic,1 > , std::map<   Eigen::Array<unsigned int,Eigen::Dynamic,1> , size_t, OrderTinyVector< unsigned int >  > ,OrderTinyVector< char>  >
/// \brief defines a point in a sparse grid : level, index, number
#define SparsePoint std::tuple< Eigen::Array<char, Eigen::Dynamic, 1> , Eigen::Array<unsigned int, Eigen::Dynamic, 1>, int >

/// \class OrderLevel sparseGridTypes.h
/// Permit to store iterator on levels
class OrderLevel
{
public:
    /// \brief comparison operator
    bool operator()(const SparseSet::const_iterator   &m_firstLevel, const  SparseSet::const_iterator    &m_secondLevel) const
    {
        for (int i = 0 ; i < m_firstLevel->first.size(); ++i)
        {
            if (m_firstLevel->first(i) <  m_secondLevel->first(i))
            {
                return true;
            }
            else
            {
                if (m_firstLevel->first(i) >  m_secondLevel->first(i))
                    return false ;
            }
        }
        return false;
    }
};

#endif /* SPARSEGRIDTYPES.H */
