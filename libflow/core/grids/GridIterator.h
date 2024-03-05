
#ifndef GRIDITERATOR_H
#define GRIDITERATOR_H
#include <Eigen/Dense>

/**  \file GridIterator.h
 *   \brief Defines an iterator on the points of a grid
 *   \author Xavier Warin
 */
namespace libflow
{

/// \class  GridIterator  GridIterator.h
///    Iterator on a given grid
class GridIterator
{

public :

    /// \brief Constructor
    GridIterator() {}

    /// \brief Destructor
    virtual ~GridIterator() {}

    /// \brief get current  coordinates
    virtual  Eigen::ArrayXd  getCoordinate() const = 0 ;

    /// \brief Check if the iterator is valid
    virtual  bool isValid(void) const = 0;

    /// \brief iterate on point
    virtual void next() = 0;

    /// \brief iterate jumping some point
    /// \param p_incr  increment in the jump
    virtual void nextInc(const int &p_incr) = 0;

    /// \brief get counter  : the integer associated the current point
    virtual int getCount() const = 0;

    /// \brief Permits to  jump to a given place given the number of processors (permits to use MPI and openmp)
    /// \param  p_rank    processor rank
    /// \param  p_nbProc  number of processor
    /// \param  p_jump    increment jump for iterator
    virtual void jumpToAndInc(const int &p_rank, const int &p_nbProc, const int &p_jump) = 0;

    /// \brief return relative position
    virtual int  getRelativePosition() const = 0 ;

    /// \brief return number of points treated
    virtual  int getNbPointRelative() const = 0 ;

    /// \brief Reset the interpolator
    virtual void reset() = 0 ;

};
}
#endif /* GRIDITERATOR_H */
