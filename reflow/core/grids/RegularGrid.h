
#ifndef REGULARGRID_H
#define REGULARGRID_H
#include <iosfwd>
#include <iostream>
#include <Eigen/Dense>
#include "reflow/core/grids/FullGrid.h"

/** \file RegularGrid.h
 *  \brief Defines a \f$n\f$ dimensional grid with equal space step
 *         Base class for classes with a given number of points on each mesh
 *  \author Xavier Warin
 */
namespace reflow
{

/// \class RegularGrid RegularGrid.h
/// Defines regular grids with same mesh size
/// Base class for grids with some points inside each mesh
class RegularGrid : public FullGrid
{
protected :
    Eigen::ArrayXd m_lowValues ; ///< minimal value of the mesh in each direction
    Eigen::ArrayXd m_step; ///< Step in each direction
    Eigen::ArrayXi m_nbStep ; ///< Number of steps in each dimension
    Eigen::ArrayXi m_dimensions ; ///< store the dimension of the global grid

public :

    /// \brief Default constructor
    RegularGrid(): m_lowValues(), m_step(), m_nbStep(), m_dimensions() {}


    /// \brief Constructor
    /// \param p_lowValues  in each dimension minimal value of the grid
    /// \param p_step       in each dimension the step size
    /// \param p_nbStep     in each dimension the number of steps
    RegularGrid(const Eigen::ArrayXd &p_lowValues, const Eigen::ArrayXd &p_step, const  Eigen::ArrayXi &p_nbStep):
        m_lowValues(p_lowValues), m_step(p_step), m_nbStep(p_nbStep), m_dimensions(p_lowValues.size())
    {
        if (p_lowValues.size() > 0)
        {
            m_dimensions = m_nbStep + 1;
        }
    }

    /// \brief Check equality between two grids
    inline bool operator==(const RegularGrid &p_reg) const
    {
        if (m_lowValues.size() != p_reg.getLowValues().size())
            return false;
        for (int i = 0; i < m_lowValues.size(); ++i)
        {
            if (!almostEqual(m_lowValues(i), p_reg.getLowValues()(i), 10))
                return false;
            if (!almostEqual(m_step(i), p_reg.getStep()(i), 10))
                return false;
            if (m_nbStep(i) !=  p_reg.getNbStep()(i))
                return false;
        }
        return true ;
    }


    /// \name get back value
    ///@{
    const Eigen::ArrayXd &getLowValues() const
    {
        return  m_lowValues ;
    }
    const Eigen::ArrayXd &getStep() const
    {
        return  m_step ;
    }

    const Eigen::ArrayXi &getNbStep() const
    {
        return m_nbStep;
    }
    ///@}

    /// \brief get number of steps in one direction
    /// \param p_id dimension concerned
    inline int getNbStep(const int &p_id) const
    {
        return m_nbStep(p_id);
    }

    /// \brief get the number of mesh
    inline int getNbMeshes() const
    {
        return m_nbStep.prod();
    }

    /// size of mesh given a lower coordinate
    Eigen::ArrayXd getMeshSize(const Eigen::Ref<const Eigen::ArrayXi > &) const
    {
        return m_step;
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


    /// \brief get back bounds  associated to the grid
    /// \return to the grid in each dimension give the extreme values (min, max)
    std::vector <std::array< double, 2>  > getExtremeValues() const
    {
        std::vector<  std::array< double, 2> > retGrid(m_lowValues.size());
        for (int i = 0; i <  m_lowValues.size(); ++i)
        {
            retGrid[i][0] = m_lowValues(i);
            retGrid[i][1] = m_lowValues(i) + m_nbStep(i) * m_step(i);
        }
        return retGrid;
    }

    /// \brief test if the point is strictly inside the domain
    /// \param p_point point to test
    /// \return true if the point is inside the open domain
    bool isStrictlyInside(const Eigen::ArrayXd &p_point) const
    {
        if (m_lowValues.size() == 0)
            return false ;
        for (int id = 0; id < p_point.size(); ++id)
        {
            if (p_point(id) <=  m_lowValues(id) +   std::fabs(m_lowValues(id))*std::numeric_limits<double>::epsilon())
                return false;
            double upperPoint = m_lowValues(id) + m_step(id) * m_nbStep(id);
            if (p_point(id) >= (upperPoint - std::fabs(upperPoint)*std::numeric_limits<double>::epsilon()))
                return false;
        }
        return true ;
    }

    /// \brief test if the point is  inside the domain (boundaries included)
    /// \param p_point point to test
    /// \return true if the point is inside the closed domain
    bool isInside(const Eigen::ArrayXd &p_point) const
    {
        if (m_lowValues.size() == 0)
            return false ;
        for (int id = 0; id < p_point.size(); ++id)
        {
            if (p_point(id) <  m_lowValues(id) - std::fabs(m_lowValues(id))*std::numeric_limits<double>::epsilon())
            {
                return false;
            }
            double upperPoint = m_lowValues(id) + m_step(id) * m_nbStep(id);
            if (p_point(id) > (upperPoint + std::max(std::fabs(upperPoint), std::fabs(m_lowValues(id)))*std::numeric_limits<double>::epsilon()*m_step(id)))
            {
                return false;
            }
        }
        return true ;
    }

    /// \brief truncate a point that it stays inside the domain
    /// \param p_point  point to truncate
    void truncatePoint(Eigen::ArrayXd &p_point) const
    {
        for (int id = 0 ; id < p_point.size(); ++id)
            p_point(id) = std::max(m_lowValues(id), std::min(m_lowValues(id) + m_step(id) * m_nbStep(id), p_point(id)));
    }

    /// \brief  To print object for debug (don't use operator << due to geners)
    void print() const
    {
        std::cout << " Low values " <<  m_lowValues << std::endl ;
        std::cout << " Step       "   <<  m_step  << std::endl ;
        std::cout << "   nbStep " << m_nbStep  << std::endl ;
    }

    /// \brief check comptability of the mesh with the number of points in each direction
    /// \param  p_nbPoints   number of points
    bool  checkMeshAndPointCompatibility(const int &p_nbPoints) const
    {
        return (p_nbPoints == (m_nbStep + 1).prod());
    }
};

}

#endif /* REGULARGRID_H */
