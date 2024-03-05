
#ifndef FULLGRIDITERATOR_H
#define FULLGRIDITERATOR_H
#include <Eigen/Dense>
#include <libflow/core/grids/GridIterator.h>

/**  \file FullGridIterator.h
 *   \brief Defines an iterator on the points of a full grid
 *   \author Xavier Warin
 */
namespace libflow
{

/// \class  FullGridIterator  FullGridIterator.h
///    Iterator on a given grid
class FullGridIterator : public GridIterator
{
protected :
    bool m_bValid ; ///< if the iterator is valid
    Eigen::ArrayXi m_sizeDim ; ///< number of point per dimension
    Eigen::ArrayXi m_coord ; ///< coordinate in  the grid
    int m_count ; ///< counter of iteration
    int m_firstPosIter ; ///< first point to treat
    int m_lastPosIter ; ///< last point not to treat (default is size of m_dataSet)
    int m_jumpInit ; ///< initial jump

public :

    /// \brief Default constructor
    FullGridIterator(): m_bValid(true), m_count(0) {}

    /// \brief Constructor
    /// \param p_sizeDim    Size of the grid in each dimension
    FullGridIterator(const  Eigen::ArrayXi  &p_sizeDim) : m_bValid(true), m_sizeDim(p_sizeDim), m_coord(Eigen::ArrayXi::Zero(p_sizeDim.size())), m_count(0), m_firstPosIter(0),
        m_lastPosIter(m_sizeDim.prod()), m_jumpInit(0)
    {
    }

    /// \brief Constructor with jump
    ///  Permits to iterate jumping some values (parallel mode)
    /// \param p_sizeDim    Size of the grid in each dimension
    /// \param p_jump       offset for the iterator
    FullGridIterator(const  Eigen::ArrayXi  &p_sizeDim, const int &p_jump) : m_bValid(true), m_sizeDim(p_sizeDim), m_coord(Eigen::ArrayXi::Zero(p_sizeDim.size())),
        m_count(0), m_firstPosIter(0), m_lastPosIter(m_sizeDim.prod()), m_jumpInit(p_jump)
    {
        int ipos = 0;
        while ((m_bValid) && (ipos++ < p_jump))
            next();
    }

    /// \brief Reset the interpolator
    void reset()
    {
        m_bValid = true;
        m_count = 0;
        m_firstPosIter = 0;
        m_lastPosIter = m_sizeDim.prod();
        int ipos = 0;
        while ((m_bValid) && (ipos++ < m_jumpInit))
            next();

    }

    /// \brief Permits to  jump to a given place given the number of processors (permits to use MPI and openmp)
    /// \param  p_rank    processor rank
    /// \param  p_nbProc  number of processor
    /// \param  p_jump    increment jump for iterator
    void jumpToAndInc(const int &p_rank, const int &p_nbProc, const int &p_jump)
    {
        int nbPoints = m_sizeDim.prod() ;
        int npointPProc = (int)(nbPoints / p_nbProc);
        int nRestPoint = nbPoints % p_nbProc;
        m_firstPosIter = p_rank * npointPProc + (p_rank < nRestPoint ? p_rank : nRestPoint);
        m_lastPosIter = m_firstPosIter + npointPProc + (p_rank < nRestPoint ? 1 : 0);
        m_count = m_firstPosIter + p_jump;
        if (m_count >= m_lastPosIter)
            m_bValid = false;
        else
        {
            int idec = m_count;
            int idiv = m_sizeDim.prod();
            for (int idim = m_sizeDim.size() - 1; idim >= 0; --idim)
            {
                idiv /= m_sizeDim(idim);
                m_coord(idim) = idec / idiv;
                idec  = idec % idiv;
            }
        }
    }

    /// \brief Check if the iterator is valid
    bool isValid(void) const
    {
        return m_bValid ;
    }

    /// \brief iterate on point
    void next()
    {
        int iDim = 0;
        m_count++;
        if (m_count >= m_lastPosIter)
            m_bValid = false;
        else
        {
            while (iDim < m_sizeDim.size())
            {
                if (m_coord(iDim) < m_sizeDim(iDim) - 1)
                {
                    m_coord(iDim) += 1;
                    return;
                }
                else
                {

                    m_coord(iDim) = 0;
                    iDim += 1;
                }
            }
            m_bValid = false;
        }
    }

    /// \brief iterate jumping some point
    /// \param p_incr  incement in the jump
    void nextInc(const int &p_incr)
    {
        int ipos = 0 ;
        while ((m_bValid) && (ipos++ < p_incr))
            next();
    }
    /// \brief get counter
    int getCount() const
    {
        return m_count;
    }
    /// \brief return relative position
    inline int getRelativePosition() const
    {
        return m_count -  m_firstPosIter;
    }

    /// \brief return number of points treated
    inline int getNbPointRelative() const
    {
        return  m_lastPosIter -  m_firstPosIter;
    }

};
}
#endif /* GRIDITERATOR_H */
