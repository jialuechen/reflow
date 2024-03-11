
#ifndef SPARSEGRIDBOUNDITERATOR_H
#define SPARSEGRIDBOUNDITERATOR_H
#include <Eigen/Dense>
#include <libflow/core/grids/SparseGridIterator.h>
#include "libflow/core/sparse/GetCoordinateBound.h"
#include "libflow/core/sparse/sparseGridBound.h"

namespace libflow
{

/// \class  SparseGridBoundIterator  SparseGridBoundIterator.h
///    Iterator on a given grid with boundary points
class SparseGridBoundIterator : public SparseGridIterator
{

    ///\brief  minimal value of the mesh in each direction
    Eigen::ArrayXd  m_lowValues ;

    /// \brief Size of the resolution domain in each dimension
    Eigen::ArrayXd  m_sizeDomain;


public :

    /// \brief default constructor
    SparseGridBoundIterator(): SparseGridIterator() {}

    /// \brief Constructor
    /// \param p_dataSet    data structure for mesh
    /// \param p_lowValues   coordinates of bottom, left etc.. point of the domain
    /// \param p_sizeDomain  domain size in each dimension  such that the points lie in \f$ [ lowValues[0], lowValues[0] + sizeDomain[0]] \times ... \times  [ lowValues[NDIM], lowValues[NDIM] + sizeDomain[0]] \f$
    SparseGridBoundIterator(const  std::shared_ptr<SparseSet>   &p_dataSet, const Eigen::ArrayXd   &p_lowValues, const Eigen::ArrayXd &p_sizeDomain) : SparseGridIterator(p_dataSet),  m_lowValues(p_lowValues), m_sizeDomain(p_sizeDomain) {}

    /// \brief Constructor with jump
    ///  Permits to iterate jumping some values (parallel mode)
    /// \param p_dataSet    data structure for mesh
    /// \param p_lowValues   coordinates of bottom, left etc.. point of the domain
    /// \param p_sizeDomain  domain size in each dimension  such that the points lie in \f$ [ lowValues[0], lowValues[0] + sizeDomain[0]] \times ... \times  [ lowValues[NDIM], lowValues[NDIM] + sizeDomain[0]] \f$
    /// \param p_jump              increment jump for iterator
    SparseGridBoundIterator(const  std::shared_ptr<SparseSet>   &p_dataSet, const Eigen::ArrayXd   &p_lowValues, const Eigen::ArrayXd &p_sizeDomain, const int &p_jump) :  SparseGridIterator(p_dataSet, p_jump),  m_lowValues(p_lowValues), m_sizeDomain(p_sizeDomain) {}

    /// \brief Constructor only iterating on points of a given level
    /// \param p_dataSet     data structure for mesh
    /// \param p_iterLevel  iterator on a multi level in the sparse grid
    /// \param p_lowValues   coordinates of bottom, left etc.. point of the domain
    /// \param p_sizeDomain  domain size in each dimension  such that the points lie in \f$ [ lowValues[0], lowValues[0] + sizeDomain[0]] \times ... \times  [ lowValues[NDIM], lowValues[NDIM] + sizeDomain[NDIM]]  \f$
    SparseGridBoundIterator(const  std::shared_ptr<SparseSet>   &p_dataSet,  const SparseSet::const_iterator &p_iterLevel,
                            const Eigen::ArrayXd   &p_lowValues, const Eigen::ArrayXd &p_sizeDomain) :
        SparseGridIterator(p_dataSet, p_iterLevel),  m_lowValues(p_lowValues), m_sizeDomain(p_sizeDomain) {}

    /// \brief Constructor with jump only iterating on points of a given level
    ///  Permits to iterate  jumping some values (parallel mode)
    /// \param p_dataSet     data structure for mesh
    /// \param p_iterLevel  iterator on a multi level in the sparse grid
    /// \param p_lowValues   coordinates of bottom, left etc.. point of the domain
    /// \param p_sizeDomain  domain size in each dimension  such that the points lie in \f$ [ lowValues[0], lowValues[0] + sizeDomain[0]] \times ... \times  [ lowValues[NDIM], lowValues[NDIM] +  sizeDomain[NDIM]] \f$
    /// \param p_jump        increment jump for iterator
    SparseGridBoundIterator(const  std::shared_ptr<SparseSet>   &p_dataSet, const SparseSet::const_iterator &p_iterLevel,
                            const Eigen::ArrayXd   &p_lowValues, const Eigen::ArrayXd &p_sizeDomain, const int &p_jump) :
        SparseGridIterator(p_dataSet, p_iterLevel, p_jump),  m_lowValues(p_lowValues), m_sizeDomain(p_sizeDomain) {}


    /// \brief get current integer coordinates
    Eigen::ArrayXd getCoordinate() const
    {
        return m_lowValues + GetCoordinateBound()(m_iterLevel->first, m_iterPosition->first) * m_sizeDomain;
    }
};
}
#endif /* SPARSEGRIDBOUNDITERATOR_H */
