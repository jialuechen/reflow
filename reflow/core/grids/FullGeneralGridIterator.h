
#ifndef FULLGENERALGRIDITERATOR_H
#define FULLGENERALGRIDITERATOR_H
#include <Eigen/Dense>
#include <reflow/core/grids/FullGridIterator.h>

/**  \file FullGeneralGridIterator.h
 *   \brief Defines an iterator on the points of a full grid with general mesh
 *   \author Xavier Warin
 */
namespace reflow
{

/// \class  FullGeneralGridIterator  FullGeneralGridIterator.h
///    Iterator on a given grid
class FullGeneralGridIterator : public FullGridIterator
{

    std::vector<std::shared_ptr<Eigen::ArrayXd> > m_meshPerDimension;

public :

    /// \brief Default constructor
    FullGeneralGridIterator() {}

    /// \brief Constructor
    /// \param p_meshPerDimension  mesh in each dimension
    /// \param p_sizeDim    Size of the grid in each dimension
    FullGeneralGridIterator(const  std::vector<std::shared_ptr<Eigen::ArrayXd> >   &p_meshPerDimension,
                            const  Eigen::ArrayXi  &p_sizeDim) : FullGridIterator(p_sizeDim), m_meshPerDimension(p_meshPerDimension)
    {}

    /// \brief Constructor with jump
    ///  Permits to iterate jumping some values (parallel mode)
    /// \param p_meshPerDimension  mesh in each dimension
    /// \param p_sizeDim    Size of the grid in each dimension
    /// \param p_jump       offset for the iterator
    FullGeneralGridIterator(const  std::vector<std::shared_ptr<Eigen::ArrayXd> >   &p_meshPerDimension, const  Eigen::ArrayXi  &p_sizeDim, const int &p_jump) :
        FullGridIterator(p_sizeDim, p_jump), m_meshPerDimension(p_meshPerDimension)
    {}

    /// \brief get current integer coordinates
    inline Eigen::ArrayXd getCoordinate() const
    {
        Eigen::ArrayXd ret(m_meshPerDimension.size());
        for (size_t id = 0; id < m_meshPerDimension.size(); ++id)
            ret(id) = (*m_meshPerDimension[id])(m_coord(id));
        return ret;
    }
};
}
#endif /* FULLGENERALGRIDITERATOR_H */
