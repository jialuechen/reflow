
#ifndef REGULARSPACEINTGRID_H
#define REGULARSPACEINTGRID_H
#include <memory>
#include <iostream>
#include <boost/lexical_cast.hpp>
#include <Eigen/Dense>
#include "libflow/core/grids/FullRegularIntGridIterator.h"


/** \file RegularSpaceIntGrid.h
 *  \brief Defines a \f$n\f$ dimensional grid  of  integer with step of one
 *  \author Xavier Warin
 */

namespace libflow
{

/// \class RegularSpaceIntGrid RegularSpaceIntGrid.h
/// Defines regular grids of integer with step of one
class RegularSpaceIntGrid
{
private :
    Eigen::ArrayXi m_lowValues ; ///< minimal value of the mesh in each direction
    Eigen::ArrayXi m_nbStep ; ///< Number of steps in each dimension
    Eigen::ArrayXi m_dimensions ; ///< store the dimension of the global grid
    Eigen::ArrayXi m_utPointToGlobal ; ///< helper to easily go from integer coordinate to global one in the mesh
    size_t m_nbPoints ; ///< number of points in the mesh

public :

    /// \brief Default constructor
    RegularSpaceIntGrid(): m_lowValues(),  m_nbStep(), m_dimensions(), m_nbPoints(0) {}


    /// \brief Constructor
    /// \param p_lowValues  in each dimension minimal value of the grid
    /// \param p_nbStep     in each dimension the number of steps
    RegularSpaceIntGrid(const Eigen::ArrayXi &p_lowValues, const  Eigen::ArrayXi &p_nbStep):
        m_lowValues(p_lowValues),  m_nbStep(p_nbStep), m_dimensions(p_lowValues.size()), m_utPointToGlobal(p_lowValues.size())
    {
        if (p_lowValues.size() > 0)
        {
            m_dimensions = m_nbStep + 1;
            m_utPointToGlobal(0) = 1;
            for (int id = 1; id < m_utPointToGlobal.size(); ++id)
                m_utPointToGlobal(id) = m_utPointToGlobal(id - 1) * m_dimensions(id - 1);
            m_nbPoints = m_utPointToGlobal(m_utPointToGlobal.size() - 1) * m_dimensions(m_utPointToGlobal.size() - 1);
        }
    }


    /// \brief get sub grid
    /// \param  p_mesh for each dimension give the first point of the mesh and the first outside of the domain
    std::shared_ptr<RegularSpaceIntGrid> getSubGrid(const Eigen::Array< std::array<int, 2>, Eigen::Dynamic, 1> &p_mesh) const
    {
        if (p_mesh.size() == 0)
        {
            return   std::make_shared<RegularSpaceIntGrid>();
        }
        Eigen::ArrayXi   lowValues(p_mesh.size()) ;
        Eigen::ArrayXi   nbStep(p_mesh.size()) ;
        for (int id = 0; id < p_mesh.size(); ++id)
        {
            nbStep(id) = p_mesh(id)[1] - p_mesh(id)[0] - 1;
            lowValues(id) = m_lowValues(id) + p_mesh(id)[0];
        }
        return std::make_shared<RegularSpaceIntGrid>(lowValues, nbStep);
    }


    /// \brief  Get position in 1D aray associate to some coordinates
    /// \param   p_iCoord  global coordinate (in integer)
    /// \return  local position in 1D array
    int globCoordPerDimToLocal(const Eigen::ArrayXi &p_iCoord) const
    {
        int iret = p_iCoord(0) - m_lowValues(0) ;
        for (int id = 1 ; id < p_iCoord.size(); ++id)
        {
            iret += (p_iCoord(id) - m_lowValues(id)) * m_utPointToGlobal(id);
        }
        return iret;
    }

    /// \brief get back iterator associated to the grid (multi thread)
    /// \param   p_iThread  Thread number  (for multi thread purpose)
    FullRegularIntGridIterator getGridIteratorInc(const int   &p_iThread) const
    {
        return FullRegularIntGridIterator(m_lowValues, m_dimensions, p_iThread) ;
    }

    /// \brief get back iterator associated to the grid
    FullRegularIntGridIterator getGridIterator() const
    {
        return FullRegularIntGridIterator(m_lowValues, m_dimensions) ;
    }


    /// \brief Get back the number of points on the meshing
    inline size_t getNbPoints() const
    {
        return m_nbPoints;
    }

    ///  dimension of the grid
    inline int getDimension() const
    {
        return   m_lowValues.size();
    }
    ///  number of steps in each dimension
    inline const Eigen::ArrayXi   &getDimensions() const
    {
        return m_dimensions ;
    }
    // get back low value in given direction
    inline int getLowValueDim(const int &p_idim) const
    {
        return m_lowValues(p_idim);
    }

    // get back max value in given direction
    inline int getMaxValueDim(const int &p_idim) const
    {
        return m_lowValues(p_idim) + m_nbStep(p_idim);
    }
    /// Get back size in one dimension
    /// \param p_idim dimension of interest
    inline int getSizeInDim(const int &p_idim) const
    {
        return m_nbStep(p_idim) + 1;
    }



};

}

#endif /* REGULARSPACEINTGRID_H */
