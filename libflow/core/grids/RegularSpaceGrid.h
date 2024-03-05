
#ifndef REGULARSPACEGRID_H
#define REGULARSPACEGRID_H
#include <memory>
#include <boost/lexical_cast.hpp>
#include <Eigen/Dense>
#include "libflow/core/utils/comparisonUtils.h"
#include "libflow/core/grids/RegularGrid.h"
#include "libflow/core/grids/FullRegularGridIterator.h"
#include "libflow/core/grids/LinearInterpolator.h"
#include "libflow/core/grids/LinearInterpolatorSpectral.h"


/** \file RegularSpaceGrid.h
 *  \brief Defines a \f$n\f$ dimensional grid with equal space step
 *  \author Xavier Warin
 */
namespace libflow
{

/// \class RegularSpaceGrid RegularSpaceGrid.h
/// Defines regular grids with same mesh size
/// Two points in each direction on each mesh
class RegularSpaceGrid : public RegularGrid
{
private :

    Eigen::ArrayXi m_utPointToGlobal ; ///< helper to easily go from integer coordinate to global one in the mesh
    size_t m_nbPoints ; ///< number of points in the mesh

public :

    /// \brief Default constructor
    RegularSpaceGrid(): RegularGrid(), m_nbPoints(0) {}


    /// \brief Constructor
    /// \param p_lowValues  in each dimension minimal value of the grid
    /// \param p_step       in each dimension the step size
    /// \param p_nbStep     in each dimension the number of steps
    RegularSpaceGrid(const Eigen::ArrayXd &p_lowValues, const Eigen::ArrayXd &p_step, const  Eigen::ArrayXi &p_nbStep):
        RegularGrid(p_lowValues, p_step, p_nbStep), m_utPointToGlobal(p_lowValues.size())
    {
        if (p_lowValues.size() > 0)
        {
            m_utPointToGlobal(0) = 1;
            for (int id = 1; id < m_utPointToGlobal.size(); ++id)
                m_utPointToGlobal(id) = m_utPointToGlobal(id - 1) * m_dimensions(id - 1);
            m_nbPoints = m_utPointToGlobal(m_utPointToGlobal.size() - 1) * m_dimensions(m_utPointToGlobal.size() - 1);
        }
    }


    /// \name get position coordinate of a point and generic information
    ///@{
    /// lower coordinate of a point (iCoord) in the grid such that each coordinates lies in the mesh defines by \f$ iCoord \f$  and \f$ iCoord+1 \f$
    Eigen::ArrayXi lowerPositionCoord(const Eigen::Ref<const Eigen::ArrayXd > &p_point) const
    {
#ifndef NOCHECK_GRID
        assert(isInside(p_point)) ;
#endif
        Eigen::ArrayXi intCoord(p_point.size());
        for (int i = 0; i < p_point.size(); ++i)
        {
            intCoord(i) = std::max(std::min(roundIntAbove((p_point(i) - m_lowValues(i)) / m_step(i)), m_nbStep(i) - 1), 0);
        }
        return intCoord;
    }
    /// upper coordinate of a point (iCoord) in the grid such that each coordinates lies in the mesh defines by \f$ iCoord-1 \f$  and \f$ iCoord \f$
    Eigen::ArrayXi upperPositionCoord(const Eigen::Ref<const Eigen::ArrayXd > &p_point) const
    {
#ifndef NOCHECK_GRID
        assert(isInside(p_point)) ;
#endif
        Eigen::ArrayXi intCoord(p_point.size());
        for (int i = 0; i < p_point.size(); ++i)
        {
            intCoord(i) = std::max(std::min(roundIntAbove((p_point(i) - m_lowValues(i)) / m_step(i)) + 1, m_nbStep(i)), 0);
        }
        return intCoord;
    }

    ///  transform integer coordinates  to real coordinates
    Eigen::ArrayXd  getCoordinateFromIntCoord(const Eigen::ArrayXi &p_icoord) const
    {
        Eigen::ArrayXd ret =  m_lowValues +  p_icoord.cast<double>() * m_step;
        return ret;
    }
    ///@}

    /// \brief get sub grid
    /// \param  p_mesh for each dimension give the first point of the mesh and the first outside of the domain
    std::shared_ptr<FullGrid> getSubGrid(const Eigen::Array< std::array<int, 2>, Eigen::Dynamic, 1> &p_mesh) const
    {
        if (p_mesh.size() == 0)
        {
            return   std::make_shared<RegularSpaceGrid>();
        }
        Eigen::ArrayXd lowValues(p_mesh.size()) ;
        Eigen::ArrayXi  nbStep(p_mesh.size()) ;
        for (int id = 0; id < p_mesh.size(); ++id)
        {
            nbStep(id) = p_mesh(id)[1] - p_mesh(id)[0] - 1;
            lowValues(id) = m_lowValues(id) + p_mesh(id)[0] * m_step(id);
        }
        return std::make_shared<RegularSpaceGrid>(lowValues, m_step, nbStep);
    }

    /// \brief Coordinate in each direction to global (integers)
    /// \param p_iCoord  coordinate (in integer)
    int intCoordPerDimToGlobal(const Eigen::ArrayXi &p_iCoord) const
    {
        int iret = p_iCoord(0);
        for (int id = 1 ; id < p_iCoord.size(); ++id)
        {
            iret += p_iCoord(id) * m_utPointToGlobal(id);
        }
        return iret;
    }

    /// \brief get back iterator associated to the grid (multi thread)
    /// \param   p_iThread  Thread number  (for multi thread purpose)
    std::shared_ptr< GridIterator> getGridIteratorInc(const int   &p_iThread) const
    {
        return std::make_shared<FullRegularGridIterator>(m_lowValues, m_step, m_dimensions, p_iThread) ;
    }

    /// \brief get back iterator associated to the grid
    std::shared_ptr< GridIterator> getGridIterator() const
    {
        return std::make_shared<FullRegularGridIterator>(m_lowValues, m_step, m_dimensions) ;
    }

    /// \brief  Get back interpolator at a point Interpolate on the grid : here it is a linear interpolator
    /// \param  p_coord   coordinate of the point for interpolation
    /// \return interpolator at the point coordinates  on the grid
    std::shared_ptr<Interpolator> createInterpolator(const Eigen::ArrayXd &p_coord) const
    {
        return 	std::make_shared<LinearInterpolator>(this, p_coord) ;

    }
    /// \brief Get back a spectral operator associated to a whole function
    /// \param p_values   Function value at the grids points
    /// \return  the whole interpolated  value function
    std::shared_ptr<InterpolatorSpectral> createInterpolatorSpectral(const Eigen::ArrayXd &p_values) const
    {
        return std::make_shared<LinearInterpolatorSpectral>(this, p_values) ;
    }

    /// \brief Get back the number of points on the meshing
    inline size_t getNbPoints() const
    {
        return m_nbPoints;
    }

};

}

#endif /* REGULARSPACEGRID_H */
