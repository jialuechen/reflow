
#ifndef FULLLEGENDREGRIDITERATOR_H
#define FULLLEGENDREGRIDITERATOR_H
#include <vector>
#include <Eigen/Dense>
#include <libflow/core/grids/FullGridIterator.h>

/**  \file FullLegendreGridIterator.h
 *   \brief Defines an iterator on the points of a full grid with legendre mesh
 *   \author Xavier Warin
 */
namespace libflow
{

/// \class  FullLegendreGridIterator  FullLegendreGridIterator.h
///    Iterator on a given grid
class FullLegendreGridIterator : public FullGridIterator
{

    Eigen::ArrayXd m_lowValues ; ///< minimal value of the mesh in each direction
    Eigen::ArrayXd m_step; ///< Step in each direction
    std::vector< Eigen::ArrayXd >  m_gllPoints ; ///< Gauss Legendre Lobatto points
    Eigen::ArrayXi m_firstPoints ; /// in each dimension the number of the first collocation point inside the grid
    Eigen::ArrayXi m_lastPoints ; /// in each direction the number of the last collocation point present in the grid in the last mesh (in the given direction)

public :

    /// \brief Default constructor
    FullLegendreGridIterator() {}

    /// \brief Constructor
    /// \param p_lowValues  in each dimension minimal value of the grid
    /// \param p_step       in each dimension the step size
    /// \param p_sizeDim    Number of mesh in each direction
    /// \param p_poly       in each direction degree for Gauss Lobatto polynomial
    /// \param p_gllPoints       in each direction Gauss Legendre Lobatto points
    /// \param p_firstPoints     first collocation in each dimension belonging to the grid
    /// \param p_lastPoints       last collocation in each dimension belonging to the grid
    FullLegendreGridIterator(const Eigen::ArrayXd &p_lowValues, const Eigen::ArrayXd &p_step,
                             const  Eigen::ArrayXi  &p_sizeDim,   const Eigen::ArrayXi &p_poly,
                             const   std::vector< Eigen::ArrayXd >   &p_gllPoints,
                             const Eigen::ArrayXi &p_firstPoints,  const Eigen::ArrayXi &p_lastPoints) :
        FullGridIterator(p_sizeDim * p_poly + 1 - p_firstPoints - p_poly + p_lastPoints), m_lowValues(p_lowValues), m_step(p_step), m_gllPoints(p_gllPoints),
        m_firstPoints(p_firstPoints), m_lastPoints(p_lastPoints)

    {}

    /// \brief Constructor with jump
    ///  Permits to iterate jumping some values (parallel mode)
    /// \param p_lowValues  in each dimension minimal value of the grid
    /// \param p_step       in each dimension the step size
    /// \param p_sizeDim    number of mesh in each direction
    /// \param p_poly       in each direction degree for Gauss Lobatto polynomial
    /// \param p_gllPoints       in each direction Gauss Legendre Lobatto points
    /// \param p_firstPoints     first collocation in each dimension belonging to the grid
    /// \param p_lastPoints       last collocation in each dimension belonging to the grid
    /// \param p_jump       offset for the iterator
    FullLegendreGridIterator(const Eigen::ArrayXd &p_lowValues, const Eigen::ArrayXd &p_step, const  Eigen::ArrayXi  &p_sizeDim,  const Eigen::ArrayXi &p_poly,
                             const   std::vector< Eigen::ArrayXd >   &p_gllPoints, const Eigen::ArrayXi &p_firstPoints,  const Eigen::ArrayXi &p_lastPoints, const int &p_jump) :
        FullGridIterator(p_sizeDim * p_poly + 1 - p_firstPoints - p_poly + p_lastPoints, p_jump), m_lowValues(p_lowValues), m_step(p_step), m_gllPoints(p_gllPoints),
        m_firstPoints(p_firstPoints), m_lastPoints(p_lastPoints)

    {}

    /// \brief get current integer coordinates
    Eigen::ArrayXd getCoordinate() const
    {
        Eigen::ArrayXd ret(m_coord.size());
        for (int i = 0; i < m_coord.size(); ++i)
        {
            int nPoint = m_gllPoints[i].size() - 1;
            int coordRec =  m_coord(i) + m_firstPoints(i);
            int imesh = coordRec / nPoint;
            int ipoint = coordRec % nPoint;
            ret(i) =  m_lowValues(i) +  m_step(i) * (imesh + 0.5 * (1 + m_gllPoints[i](ipoint)));
        }
        return ret;
    }
};
}
#endif /* FULLLEGENDREGRIDITERATOR_H */
