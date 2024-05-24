
#ifndef LINEARINTERPOLATORSPECTRAL_H
#define LINEARINTERPOLATORSPECTRAL_H
#include <Eigen/Dense>
#include "reflow/core/grids/FullGrid.h"
#include "reflow/core/grids/InterpolatorSpectral.h"

/** \file LinearInterpolatorSpectral.h
 *  \brief Defines an interpolator for a linear grid   : here is a global interpolator, storing the representation of the  function
 *         to interpolate.
 * \author Xavier Warin
 */
namespace reflow
{

/// \class LinearInterpolatorSpectral LinearInterpolatorSpectral.h
/// Linear interpolation object with  points on the grid
/// Templated are the basis functions
class LinearInterpolatorSpectral : public InterpolatorSpectral
{

    const FullGrid   *m_grid ; //< grid used
    Eigen::ArrayXd m_values ; //< Fonction values on the  grid

public :

    /** \brief Constructor taking in values on the grid
     *  \param p_grid      is the linear  grid used to interpolate
     *  \param p_values    Fonction values on the  grid
     */
    LinearInterpolatorSpectral(const FullGrid *p_grid, const Eigen::ArrayXd &p_values) : m_grid(p_grid), m_values(p_values)
    {}

    /** \brief Constructor taking the values of the grids , but without affecting the grid
    *  Convenient for serialization : affectation of the pointer should be done after
    * \param p_values values associated to the interpolator
    */
    LinearInterpolatorSpectral(const Eigen::ArrayXd &p_values): m_values(p_values) {}  ;

    /** \brief Affect the grid : use for deserialization
     * \param p_grid  the grid to affect
     */
    void setGrid(const SpaceGrid *p_grid)
    {
        m_grid = static_cast< const FullGrid *>(p_grid);
    }

    /** \brief Get back grid associated to operator
     */
    const reflow::SpaceGrid *getGrid()
    {
        return static_cast< const SpaceGrid *>(m_grid);
    }


    /**  \brief  interpolate
     *  \param  p_point  coordinates of the point for interpolation
     *  \return interpolated value
     */
    inline double apply(const Eigen::ArrayXd &p_point) const
    {
        return m_grid->createInterpolator(p_point)->apply(m_values);
    }

    /// \brief get back function values
    const Eigen::ArrayXd &getValues() const
    {
        return m_values;
    }
};
}
#endif /* LINEARINTERPOLATORSPECTRAL_H */
