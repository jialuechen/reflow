
#ifndef LEGENDREINTERPOLATORSPECTRAL_H
#define LEGENDREINTERPOLATORSPECTRAL_H
#include <Eigen/Dense>
#include <libflow/core/grids/RegularLegendreGrid.h>
#include <libflow/core/grids/InterpolatorSpectral.h>

/** \file LegendreInterpolatorSpectral.h
 *  \brief Defines a legendre interpolator on a full grid
 * \author Xavier Warin
 */
namespace libflow
{

/// \class LegendreInterpolatorSpectral LegendreInterpolatorSpectral.h
/// As in LegendreInterpolator class,it permits to interpolate a function.
/// This version is effective if one function is to be interpolated at different points.
/// In this case a spectral representation of the function is stored and used for interpolation
class LegendreInterpolatorSpectral : public InterpolatorSpectral
{

private :

    const RegularLegendreGrid *m_grid ;  ///< grid used
    Eigen::ArrayXXd m_spectral ; ///< spectral representation of the function (nb of basis function, number of mesh)
    Eigen::ArrayXXi  m_funcBaseExp ; ///< helper to get the degree of a polynomial : for each multidimensional function basis gives in all dimensions the degree of the polynomial 1D basis
    Eigen::ArrayXd m_min ; ///<  on each mesh minimal of the values on the mesh
    Eigen::ArrayXd m_max ; ///<  on each mesh maximal of the values on the mesh

public :

    /** \brief Constructor taking in values on the grid
     *  \param p_grid   is the grid used to interpolate
     *  \param p_values   Function value at the grids points
     */
    LegendreInterpolatorSpectral(const RegularLegendreGrid   *p_grid, const Eigen::ArrayXd &p_values) ;


    /** \brief Constructor taking the values of the grids , but without affecting the grid
     *  Convinient for serialization : affectation of the pointer should be done after
     * \param p_spectral    spectral values associated to the interpolator
     * \param p_funcBaseExp  helper
     * \param p_min        on each mesh minimal of the values on the mesh
     * \param p_max        on each mesh maximal of the values on the mesh
     */
    LegendreInterpolatorSpectral(const Eigen::ArrayXXd &p_spectral, const Eigen::ArrayXXi &p_funcBaseExp,
                                 const Eigen::ArrayXd &p_min, const Eigen::ArrayXd &p_max) : m_spectral(p_spectral), m_funcBaseExp(p_funcBaseExp), m_min(p_min), m_max(p_max) {};

    /** \brief Affect the grid : use for deserialization
     * \param p_grid  the grid to affect
     */
    void setGrid(const SpaceGrid *p_grid)
    {
        m_grid = static_cast< const RegularLegendreGrid *>(p_grid);
    }

    /** \brief Get back grid associated to operator
     */
    const libflow::SpaceGrid *getGrid()
    {
        return static_cast< const SpaceGrid *>(m_grid);
    }

    /**  \brief  interpolate
     *  \param  p_point  coordinates of the point for interpolation
     *  \return interpolated value
     */
    inline double apply(const Eigen::ArrayXd &p_point) const
    {
        // legendre functions
        std::shared_ptr< std::array< std::function< double(const double &) >, 11 > >  legendre = m_grid->getLegendre();
        Eigen::ArrayXd xCoord(p_point.size());
        // mesh number
        int imeshLoc = 0;
        int idec = 1;
        for (int id = 0; id < p_point.size(); ++id)
        {
            int icoordMin = 0 ;
            m_grid->rescalepoint(p_point(id), id, xCoord(id), icoordMin);
            int  coordmesh = icoordMin / m_grid->getPoly(id);
            imeshLoc += idec * coordmesh;
            idec *= m_grid->getNbStep(id);
        }
        double vFunction = 0. ;
        for (int ib = 0 ; ib < m_funcBaseExp.cols(); ++ib)
        {
            double funcVal = 1. ;
            for (int id = 0 ; id < m_funcBaseExp.rows()  ; ++id)
                funcVal *= (*legendre)[m_funcBaseExp(id, ib)](xCoord(id));
            vFunction += m_spectral(ib, imeshLoc) * funcVal;
        }
        // to avoid oscillations
        vFunction = std::min(std::max(vFunction, m_min(imeshLoc)), m_max(imeshLoc));
        return vFunction;
    }
    /** \brief Get back the spectral values */
    const Eigen::ArrayXXd &getSpectral() const
    {
        return m_spectral ;
    }

    /** \brief Get back helper */
    const Eigen::ArrayXXi &getFuncBaseExp() const
    {
        return m_funcBaseExp ;
    }

    /** \brief Get back m_min */
    const Eigen::ArrayXd &getMin() const
    {
        return m_min ;
    }


    /** \brief Get back m_max */
    const Eigen::ArrayXd &getMax() const
    {
        return m_max ;
    }


};
}
#endif
