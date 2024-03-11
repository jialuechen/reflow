
#ifndef SPARSEINTERPOLATORSPECTRAL_H
#define SPARSEINTERPOLATORSPECTRAL_H
#include <Eigen/Dense>
#include "libflow/core/grids/SparseSpaceGrid.h"
#include "libflow/core/grids/InterpolatorSpectral.h"

namespace libflow
{

/// \class SparseInterpolatorSpectral SparseInterpolatorSpectral.h
/// Sparse interpolation object with any points
/// Templated are the basis functions
class SparseInterpolatorSpectral : public InterpolatorSpectral
{

    const SparseSpaceGrid *m_grid ;  //< grid used
    Eigen::ArrayXd m_hierar ; //< Hierarchized value of the function  to interpolate

public :

    /** \brief Constructor taking in values on the grid
     *  \param p_grid      is the sparse  grid used to interpolate
     *  \param p_values    Function values on the sparse grid
     */
    SparseInterpolatorSpectral(const SparseSpaceGrid *p_grid, const Eigen::ArrayXd &p_values) : m_grid(p_grid), m_hierar(p_values)
    {
        // store hierarchized values
        p_grid->toHierarchize(m_hierar);
    }

    /** \brief Constructor convenient for deserialization
     * \param p_hierar  Hierarchical values
     */
    SparseInterpolatorSpectral(const Eigen::ArrayXd &p_hierar): m_hierar(p_hierar) {}

    /** \brief Affect the grid : use for deserialization
     * \param p_grid  the grid to affect
     */
    void setGrid(const SpaceGrid *p_grid)
    {
        m_grid = static_cast< const SparseSpaceGrid *>(p_grid);
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
        return m_grid->createInterpolator(p_point)->apply(m_hierar);
    }

    /**  \brief Get back hierarchical values
     */
    const Eigen::ArrayXd   &getHierar() const
    {
        return m_hierar;
    }
};
}
#endif /* SPARSEINTERPOLATORSPECTRAL_H */
