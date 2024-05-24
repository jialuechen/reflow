
#ifndef GENERALSPACEGRID_H
#define GENERALSPACEGRID_H
#include <iostream>
#include <memory>
#include <vector>
#include <array>
#include <boost/lexical_cast.hpp>
#include <Eigen/Dense>
#include "reflow/core/utils/comparisonUtils.h"
#include "reflow/core/grids/FullGrid.h"
#include "reflow/core/grids/FullGeneralGridIterator.h"
#include "reflow/core/grids/LinearInterpolator.h"
#include "reflow/core/grids/LinearInterpolatorSpectral.h"


namespace reflow
{

/// \class GeneralSpaceGrid GeneralSpaceGrid.h
/// Non regular full grid definition
class GeneralSpaceGrid : public FullGrid
{
private :

    std::vector<std::shared_ptr<Eigen::ArrayXd> > m_meshPerDimension;
    Eigen::ArrayXi m_utPointToGlobal ; ///< helper to easily go from integer coordinate to global one in the mesh
    size_t  m_nbPoints ; ///< number of points in the mesH
    Eigen::ArrayXi m_dimensions ; ///< store the dimension of the global grid


public :

/// \brief Default constructor
    GeneralSpaceGrid(): m_nbPoints(0) {}

/// \brief Constructor
/// \param p_meshPerDimension  mesh in each dimension
    GeneralSpaceGrid(const  std::vector<std::shared_ptr<Eigen::ArrayXd> >   &p_meshPerDimension):
        m_meshPerDimension(p_meshPerDimension), m_utPointToGlobal(p_meshPerDimension.size()), m_dimensions(p_meshPerDimension.size())
    {
        if (p_meshPerDimension.size() > 0)
        {
            m_utPointToGlobal(0) = 1;
            for (int id = 1; id < m_utPointToGlobal.size(); ++id)
                m_utPointToGlobal(id) = m_utPointToGlobal(id - 1) * p_meshPerDimension[id - 1]->size();
            m_nbPoints = m_utPointToGlobal(m_utPointToGlobal.size() - 1) * p_meshPerDimension[m_utPointToGlobal.size() - 1]->size();
            for (size_t i = 0; i < m_meshPerDimension.size(); ++i)
                m_dimensions(i) = m_meshPerDimension[i]->size();
        }
        else
        {
            m_nbPoints = 0;
        }
    }


    /// \brief Check equality between two grids
    inline bool operator==(const GeneralSpaceGrid &p_reg) const
    {
        if (m_meshPerDimension.size() != p_reg.getMeshPerDimension().size())
            return  false;
        for (size_t i = 0; i < m_meshPerDimension.size() ; ++i)
            for (int j = 0; j < m_meshPerDimension[i]->size() ; ++j)
            {
                if (!almostEqual((*m_meshPerDimension[i])(j), (*p_reg.getMeshPerDimension()[i])(j), 10))
                {
                    return  false;
                }
            }
        return true;
    }
    /// \name get back value
    ///@{
    const std::vector<std::shared_ptr<Eigen::ArrayXd> > &getMeshPerDimension() const
    {
        return  m_meshPerDimension ;
    }
    inline size_t getNbPoints() const
    {
        return m_nbPoints;
    }
    ///@}

    /// \name get position coordinate of a point and generical information
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
            int ipos = 1 ;
            while ((p_point(i) > (*m_meshPerDimension[i])(ipos)) && (ipos < (m_meshPerDimension[i]->size() - 1))) ipos++;
            intCoord(i) = ipos - 1;
        }
        return intCoord;
    }
    /// upper coordinate of a point (iCoord) in the grid such that each coordinates lies in the mesh defines by \f$ iCoord-1 \f$  and \f$ iCoord+1 \f$
    Eigen::ArrayXi upperPositionCoord(const Eigen::Ref<const Eigen::ArrayXd >   &p_point) const
    {
#ifndef NOCHECK_GRID
        assert(isInside(p_point)) ;
#endif
        Eigen::ArrayXi intCoord(p_point.size());
        for (int i = 0; i < p_point.size(); ++i)
        {
            int ipos = 1 ;
            while ((p_point(i) > (*m_meshPerDimension[i])(ipos)) && (ipos < m_meshPerDimension[i]->size())) ipos++;
            intCoord(i) = ipos;
        }
        return intCoord;
    }
    /// size of mesh given a lower coordinate
    Eigen::ArrayXd getMeshSize(const Eigen::Ref<const Eigen::ArrayXi > &p_icoord) const
    {
        Eigen::ArrayXd mesh(p_icoord.size());
        for (size_t i = 0; i < m_meshPerDimension.size(); ++i)
        {
            assert(m_meshPerDimension[i]->size() > p_icoord(i) + 1);
            mesh(i) = (*m_meshPerDimension[i])(p_icoord(i) + 1) - (*m_meshPerDimension[i])(p_icoord(i));
        }
        return mesh;
    }

    /// dimension of the grid
    inline int getDimension() const
    {
        return  m_meshPerDimension.size();
    }

    /// number of steps in each dimension
    inline const Eigen::ArrayXi &getDimensions() const
    {
        return m_dimensions ;
    }
    ///@}

    /// \brief transform integer coordinates  to real coordinates
    Eigen::ArrayXd  getCoordinateFromIntCoord(const Eigen::ArrayXi &p_icoord) const
    {
        Eigen::ArrayXd ret(m_meshPerDimension.size());
        for (size_t id = 0; id < m_meshPerDimension.size(); ++id)
            ret(id) = (*m_meshPerDimension[id])(p_icoord(id));
        return ret;
    }

    /// \brief Coordinate in each direction to global
    int intCoordPerDimToGlobal(const Eigen::ArrayXi &p_iCoord) const
    {
        int iret = p_iCoord(0);
        for (int id = 1 ; id < p_iCoord.size(); ++id)
        {
            iret += p_iCoord(id) * m_utPointToGlobal(id);
        }
        return iret;
    }

    /// \brief get sub grid
    /// \param  p_mesh for each dimension give the first point of the mesh and the first outside of the domain
    std::shared_ptr<FullGrid> getSubGrid(const Eigen::Array< std::array<int, 2>, Eigen::Dynamic, 1> &p_mesh) const
    {
        if (p_mesh.size() == 0)
            return  std::make_shared<GeneralSpaceGrid>();
        std::vector<std::shared_ptr<Eigen::ArrayXd> > localMesh(p_mesh.size());
        for (int id = 0; id < p_mesh.size(); ++id)
        {
            int iSize = p_mesh(id)[1] - p_mesh(id)[0];
            localMesh[id] = std::make_shared<Eigen::ArrayXd>(iSize);
            *localMesh[id] = m_meshPerDimension[id]->segment(p_mesh(id)[0], iSize);
        }
        return std::make_shared<GeneralSpaceGrid>(localMesh);
    }


    /// \brief test if the point is inside the domain
    /// \param p_point point to test
    /// \return true if the point is inside the open domain
    bool isStrictlyInside(const Eigen::ArrayXd &p_point) const
    {
        if (m_meshPerDimension.size() == 0)
            return false;
        for (int id = 0; id < p_point.size(); ++id)
        {
            if (p_point(id) <= (*m_meshPerDimension[id])[0] + std::fabs((*m_meshPerDimension[id])[0])*std::numeric_limits<double>::epsilon())
                return false;
            if (p_point(id) >= (*m_meshPerDimension[id])[m_meshPerDimension[id]->size() - 1] - std::fabs((*m_meshPerDimension[id])[m_meshPerDimension[id]->size() - 1])*
                    std::numeric_limits<double>::epsilon())
                return false;
        }
        return true ;
    }

    /// \brief test if the point is  inside the domain (boundaries included)
    /// \param p_point point to test
    /// \return true if the point is inside the closed domain
    bool isInside(const Eigen::ArrayXd &p_point) const
    {
        if (m_meshPerDimension.size() == 0)
            return false;
        for (int id = 0; id < p_point.size(); ++id)
        {
            double errNum = std::max(std::fabs((*m_meshPerDimension[id])[0]), std::fabs((*m_meshPerDimension[id])[m_meshPerDimension[id]->size() - 1]));
            if (p_point(id) < (*m_meshPerDimension[id])[0] - errNum * std::numeric_limits<double>::epsilon())
                return false;
            if (p_point(id) > (*m_meshPerDimension[id])[m_meshPerDimension[id]->size() - 1] + errNum * std::numeric_limits<double>::epsilon())
                return false;
        }
        return true ;

    }

    /// \brief get back bounds  associated to the grid
    /// \return to the grid in each dimension give the extreme values (min, max)
    std::vector <std::array< double, 2>  > getExtremeValues() const
    {
        std::vector<  std::array< double, 2> > retGrid(m_meshPerDimension.size());
        for (size_t i = 0; i <  m_meshPerDimension.size(); ++i)
        {
            retGrid[i][0] = (*m_meshPerDimension[i])(0);
            retGrid[i][1] = (*m_meshPerDimension[i])(m_meshPerDimension[i]->size() - 1);
        }
        return retGrid;
    }
    /// \brief truncate a point that it stays inside the domain
    /// \param p_point  point to truncate
    inline void truncatePoint(Eigen::ArrayXd &p_point) const
    {
        for (size_t i = 0; i <  m_meshPerDimension.size(); ++i)
            p_point(i) = std::min(std::max(p_point(i), (*m_meshPerDimension[i])(0)), (*m_meshPerDimension[i])(m_meshPerDimension[i]->size() - 1)) ;
    }

    /// \brief get back iterator associated to the grid (multi thread)
    /// \param   p_iThread  Thread number  (for multi thread purpose)
    std::shared_ptr< GridIterator> getGridIteratorInc(const int &p_iThread) const
    {
        return std::make_shared< FullGeneralGridIterator>(m_meshPerDimension, m_dimensions, p_iThread) ;
    }

    /// \brief get back iterator associated to the grid
    std::shared_ptr< GridIterator> getGridIterator() const
    {
        return std::make_shared< FullGeneralGridIterator>(m_meshPerDimension, m_dimensions) ;
    }
    /// \brief  Get back interpolator at a point Interpolate on the grid : here it is a linear interpolator
    /// \param  p_coord   coordinate of the point for interpolation
    /// \return interpolator at the point coordinate on the grid
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

    /// \brief check comptability of the mesh with the number of points in each direction
    /// \param  p_nbPoints   number of points
    bool  checkMeshAndPointCompatibility(const int &p_nbPoints) const
    {
        return (p_nbPoints == static_cast<int>(m_nbPoints));
    }
};
}
#endif /* GENERALSPACEGRID_H */
