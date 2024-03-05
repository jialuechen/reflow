// Copyright (C) 2021 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#ifndef FULLREGULARINTGRIDITERATOR_H
#define FULLREGULARINTGRIDITERATOR_H
#include <Eigen/Dense>
#include <libflow/core/grids/FullGridIterator.h>



/** \file  FullRegularIntGridIterator.h
 *  \brief defines an iterator on points of a full grid of int values with a step of one
 *  \author  xavier Warin
 */

namespace libflow
{
/// \class FullRegularIntGridIterator  FullRegularIntGridIterator.h
///    Iterator on a given grid
class FullRegularIntGridIterator :  public FullGridIterator
{

    Eigen::ArrayXi m_lowValues ; ///< minimal value of the mesh in each direction

public :

    /// \brief Default constructor
    FullRegularIntGridIterator() {}

    /// \brief Constructor
    /// \param p_lowValues  in each dimension minimal value of the grid
    /// \param p_sizeDim    Size of the grid in each dimension (number of points)
    FullRegularIntGridIterator(const Eigen::ArrayXi &p_lowValues,
                               const  Eigen::ArrayXi  &p_sizeDim) : FullGridIterator(p_sizeDim), m_lowValues(p_lowValues)
    {}

    /// \brief Constructor with jump
    ///  Permits to iterate jumping some values (parallel mode)
    /// \param p_lowValues  in each dimension minimal value of the grid
    /// \param p_sizeDim    Size of the grid in each dimension (number of points)
    /// \param p_jump       offset for the iterator
    FullRegularIntGridIterator(const Eigen::ArrayXi &p_lowValues, const  Eigen::ArrayXi  &p_sizeDim, const int &p_jump) :
        FullGridIterator(p_sizeDim, p_jump), m_lowValues(p_lowValues)
    {}

    /// \brief get current  coordinates
    Eigen::ArrayXd getCoordinate() const
    {
        Eigen::ArrayXd ret =  m_lowValues.cast<double>() +  m_coord.cast<double>() ;
        return ret;
    }

    /// \brief get current integer coordinates
    Eigen::ArrayXi getIntCoordinate() const
    {
        return m_lowValues + m_coord ;
    }
};
}
#endif /* FULLREGULARINTGRIDITERATOR_H */

