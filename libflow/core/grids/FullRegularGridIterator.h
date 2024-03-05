// Copyright (C) 2016 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#ifndef FULLREGULARGRIDITERATOR_H
#define FULLREGULARGRIDITERATOR_H
#include <Eigen/Dense>
#include <libflow/core/grids/FullGridIterator.h>

/**  \file FullRegularGridIterator.h
 *   \brief Defines an iterator on the points of a full grid with regular mesh
 *   \author Xavier Warin
 */
namespace libflow
{

/// \class  FullRegularGridIterator  FullRegularGridIterator.h
///    Iterator on a given grid
class FullRegularGridIterator : public FullGridIterator
{

    Eigen::ArrayXd m_lowValues ; ///< minimal value of the mesh in each direction
    Eigen::ArrayXd m_step; ///< Step in each direction

public :

    /// \brief Default constructor
    FullRegularGridIterator() {}

    /// \brief Constructor
    /// \param p_lowValues  in each dimension minimal value of the grid
    /// \param p_step       in each dimension the step size
    /// \param p_sizeDim    Size of the grid in each dimension (number of points)
    FullRegularGridIterator(const Eigen::ArrayXd &p_lowValues, const Eigen::ArrayXd &p_step,
                            const  Eigen::ArrayXi  &p_sizeDim) : FullGridIterator(p_sizeDim), m_lowValues(p_lowValues), m_step(p_step)
    {}

    /// \brief Constructor with jump
    ///  Permits to iterate jumping some values (parallel mode)
    /// \param p_lowValues  in each dimension minimal value of the grid
    /// \param p_step       in each dimension the step size
    /// \param p_sizeDim    Size of the grid in each dimension (number of points)
    /// \param p_jump       offset for the iterator
    FullRegularGridIterator(const Eigen::ArrayXd &p_lowValues, const Eigen::ArrayXd &p_step, const  Eigen::ArrayXi  &p_sizeDim, const int &p_jump) :
        FullGridIterator(p_sizeDim, p_jump), m_lowValues(p_lowValues), m_step(p_step)
    {}

    /// \brief get current integer coordinates
    Eigen::ArrayXd getCoordinate() const
    {
        Eigen::ArrayXd ret =  m_lowValues +  m_coord.cast<double>() * m_step;
        return ret;
    }
};
}
#endif /* FULLREGULARGRIDITERATOR_H */
