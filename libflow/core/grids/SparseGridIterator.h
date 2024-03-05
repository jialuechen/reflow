
#ifndef SPARSEGRIDITERATOR_H
#define SPARSEGRIDITERATOR_H
#include <Eigen/Dense>
#include <libflow/core/grids/GridIterator.h>
#include "libflow/core/sparse/sparseGridTypes.h"
#include "libflow/core/sparse/sparseGridCommon.h"

/**  \file SparseGridIterator.h
 *   \brief Defines an iterator on the points of a  sparse  grid
 *   \author Xavier Warin
 */
namespace libflow
{

/// \class  SparseGridIterator  SparseGridIterator.h
///    Iterator on a given grid
class SparseGridIterator : public GridIterator
{

protected :

    std::shared_ptr<SparseSet>  m_dataSet; ///< Data structure for sparse grid
    SparseSet::const_iterator m_iterLevelFirst ; ///< Iterator for first level to be treated
    SparseSet::const_iterator m_iterLevelLast  ; ///< Iterator for first level not treated
    SparseSet::const_iterator m_iterLevel ; ///< Iterator for the  level of the grid
    SparseLevel::const_iterator m_iterPosition ; ///< Iterator for the position of the point in the grid
    int m_posIter ; ///< integer for position in the iterator
    int m_firstPosIter ; ///< first point to treat
    int m_lastPosIter ; ///< last point not to treat (default is size of m_dataSet)
    bool m_bValid ; ///< true if the iterator is valid
    int m_jumpInit ; ///< initial jump

    /// \brief number of points
    inline int  getPointNumber() const
    {
        int nbPt = 0;
        for (SparseSet::const_iterator iterLevel =  m_iterLevelFirst; iterLevel !=  m_iterLevelLast; ++iterLevel)
            nbPt += iterLevel->second.size();
        return nbPt ;
    }

    /// \brief Increment iterators
    /// \param p_jump   increment of the iterator
    inline void incrementIterator(const int &p_jump)
    {
        int iJump = 0;
        SparseSet::const_iterator iterLevel;
        SparseLevel::const_iterator iterPosition;
        int bJumpLevel = false;
        for (iterLevel = m_iterLevel; iterLevel != m_iterLevelLast; ++iterLevel, bJumpLevel = true)
        {
            if (bJumpLevel)
                m_iterPosition = iterLevel->second.begin();
            for (iterPosition = m_iterPosition;  iterPosition != iterLevel->second.end(); ++iterPosition)
            {
                iJump += 1;
                if (iJump > p_jump)
                {
                    break;
                }
                else
                    m_posIter += 1 ;

            }
            if (iJump > p_jump)
                break;
        }
        m_iterLevel = iterLevel;
        m_iterPosition = iterPosition;

        if (m_posIter >= m_lastPosIter)
        {
            m_bValid = false;
        }
    }

public :

    /// \brief default constructor
    SparseGridIterator() {}

    /// \brief Constructor
    /// \param p_dataSet    data structure for mesh
    SparseGridIterator(const  std::shared_ptr<SparseSet>   &p_dataSet) : m_dataSet(p_dataSet),  m_iterLevelFirst(m_dataSet->begin()), m_iterLevelLast(m_dataSet->end()),
        m_iterLevel(m_dataSet->begin()), m_iterPosition(m_iterLevel->second.begin()), m_posIter(0),
        m_firstPosIter(0), m_lastPosIter(getPointNumber()),
        m_bValid(true), m_jumpInit(0)
    {}

    /// \brief Constructor with jump
    ///  Permits to iterate jumping some values (parallel mode)
    /// \param p_dataSet    data structure for mesh
    /// \param p_jump              increment jump for iterator
    SparseGridIterator(const  std::shared_ptr<SparseSet>   &p_dataSet, const int &p_jump) :
        m_dataSet(p_dataSet), m_iterLevelFirst(m_dataSet->begin()), m_iterLevelLast(m_dataSet->end()), m_iterLevel(m_dataSet->begin()),
        m_iterPosition(m_iterLevel->second.begin()), m_posIter(0), m_firstPosIter(0), m_lastPosIter(getPointNumber()), m_bValid(true), m_jumpInit(p_jump)
    {
        incrementIterator(p_jump);
    }

    /// \brief Constructor for iterator only for the points of a given multi level
    /// \param p_dataSet    data structure for mesh
    /// \param p_iterLevel  iterator on a multi level in the sparse grid
    SparseGridIterator(const  std::shared_ptr<SparseSet>   &p_dataSet, const SparseSet::const_iterator &p_iterLevel) :
        m_dataSet(p_dataSet), m_iterLevelFirst(p_iterLevel), m_iterLevelLast(p_iterLevel), m_iterLevel(p_iterLevel), m_iterPosition(p_iterLevel->second.begin()), m_posIter(0),
        m_firstPosIter(0), m_lastPosIter(p_iterLevel->second.size()),
        m_bValid(true), m_jumpInit(0)
    {
        m_iterLevelLast++;
    }

    /// \brief Constructor with jumpfor iterator only for the points of a given multi level
    ///  Permits to iterate jumping some values (parallel mode)
    /// \param p_dataSet    data structure for mesh
    /// \param p_iterLevel  iterator on a multi level in the sparse grid
    /// \param p_jump       increment jump for iterator
    SparseGridIterator(const  std::shared_ptr<SparseSet>   &p_dataSet, const SparseSet::const_iterator &p_iterLevel, const int &p_jump) :
        m_dataSet(p_dataSet),  m_iterLevelFirst(p_iterLevel), m_iterLevelLast(p_iterLevel),
        m_iterLevel(p_iterLevel), m_iterPosition(p_iterLevel->second.begin()), m_posIter(0), m_firstPosIter(0), m_lastPosIter(p_iterLevel->second.size()), m_bValid(true),
        m_jumpInit(p_jump)
    {
        m_iterLevelLast++;
        incrementIterator(p_jump);
    }

    /// \brief reset interpolator
    void reset()
    {
        m_iterLevel = m_iterLevelFirst;
        m_iterPosition = m_iterLevelFirst->second.begin();
        m_posIter = 0;
        m_bValid = true;
        incrementIterator(m_jumpInit);
    }

    /// \brief Permits to  jump to a given place given the number of processors (permits to use MPI and openmp)
    /// \param  p_rank    processor rank
    /// \param  p_nbProc  number of processor
    /// \param  p_jump    increment jump for iterator
    void jumpToAndInc(const int &p_rank, const int &p_nbProc, const int &p_jump)
    {
        int nbPoints = m_lastPosIter ;
        int npointPProc = (int)(nbPoints / p_nbProc);
        int nRestPoint = nbPoints % p_nbProc;
        m_firstPosIter = p_rank * npointPProc + (p_rank < nRestPoint ? p_rank : nRestPoint);
        m_lastPosIter = m_firstPosIter + npointPProc + (p_rank < nRestPoint ? 1 : 0);
        incrementIterator(m_firstPosIter + p_jump);
    }

    /// \brief Check if the iterator is valid
    bool isValid(void) const
    {
        return m_bValid ;
    }
    /// \brief iterate on point
    void next()
    {
        incrementIterator(1);
    }
    /// \brief iterate jumping some point
    /// \param p_incr  increment in the jump
    void nextInc(const int &p_incr)
    {
        incrementIterator(p_incr);
    }

    /// \brief get counter
    int getCount() const
    {
        return m_iterPosition->second;
    }

    /// \brief return relative position
    inline int getRelativePosition() const
    {
        return m_posIter -  m_firstPosIter;
    }

    /// \brief return number of points treated
    inline int getNbPointRelative() const
    {
        return  m_lastPosIter -  m_firstPosIter;
    }

};
}
#endif /* SPARSEGRIDITERATOR_H */
