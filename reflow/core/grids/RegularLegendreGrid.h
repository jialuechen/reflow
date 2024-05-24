
#ifndef REGULARLEGENDREGRID_H
#define REGULARLEGENDREGRID_H
#include <vector>
#include <functional>
#include <array>
#include <math.h>
#include <boost/lexical_cast.hpp>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include "reflow/core/utils/comparisonUtils.h"
#include "reflow/core/grids/RegularGrid.h"
#include "reflow/core/grids/FullLegendreGridIterator.h"
#include "reflow/core/grids/Interpolator.h"
#include "reflow/core/utils/AnalyticLegendre.h"

/** \file RegularLegendreGrid.h
 * \brief Defines a  \f$n\f$ dimensional grid with equal space step.
 *        On each mesh,  in each direction  Gauss Legendre Lobatto  points are added
 *        The number of points on each mesh  is depending in the dimension.
 *         This grid is associated to an interpolator
 *         \f{eqnarray*}{
 *          I_{M}(f) & =&  \sum_{k=0}^{M} \tilde f_k L_k(x) ,  \\
 *          \tilde f_k &  =  & \frac{1}{\kappa_k} \sum_{i=0}^{M} \rho_i f(\eta_i) L_k( \eta_i) , \\
 *          \kappa_k & = &  \sum_{i=0}^{M} L_k(\eta_i)^2 \rho_i ,
 *         \f}
 *         where the functions  \f$L_N\f$  satisfy the recurrence
 *         \f{eqnarray*}{
 *        (N+1) L_{N+1}(x) & = &  (2N+1) x L_N(x) -N L_{N-1}(x), \\
 *         L_0 & = & 1 , \quad   L_1 = x ,
 *         \f}
 *         \f$\eta_1 = -1, \eta_{M+1} =1\f$, the \f$\eta_i\f$ \f$(i=2,...,M)\f$  are the  zeros of \f$L^{'}_{M}\f$  and the eigenvalues of the matrix \f$P\f$
 *       \f{eqnarray*}{
 *            P &=& \left ( \begin{array}{lllll}
 *          0  & \gamma_1 & ...& 0 & 0 \\
 *           \gamma_1 & 0 & ...   & 0 & 0 \\
 *            ...      & ... & ... & ... & ... \\
 *             0  & 0 & ... & 0 & \gamma_{M-2} \\
 *            0  & 0 & ... & \gamma_{M-2} & 0
 *         \end{array}
 *   \right),  \\
 *   \gamma_n & = & \frac{1}{2} \sqrt{\frac{n(n+2)}{(n+\frac{1}{2})(n+\frac{3}{2})}}  , 1 \le n \le M-2 ,
 *  \f}
 *   and the weights satisfies
 *    \f[
 *     \rho_{i}   =  \frac{2.}{(M+1)M L_{M}^2(\eta_i)} ,  1 \le i \le M+1.
 *    \f]
 *  The grid may be a whole grid or a truncated grid : in each direction the first points \f$\eta_i \f$ on the first mesh or the last points in the last mesh may be
 *  missing : this may arrive when a sub grid of a grid is taken.
 *  \author Xavier Warin
 */
namespace reflow
{


/// \class RegularLegendreGrid 	RegularLegendreGrid.h
/// Defines a regular grid with same mesh but adding Gauss Legendre Lobatto points

class RegularLegendreGrid : public RegularGrid
{
private :

    Eigen::ArrayXi m_poly ; ///< Polynomial degree for Gauss Lobatto points
    std::vector< Eigen::ArrayXd >   m_gllPoints ; ///< Gauss Legendre Lobatto points
    int m_nbPoints; ///< Total number of points
    std::shared_ptr< std::array< std::function< double(const double &) >, 11 > > m_legendre;   ///< store legendre polynomial
    /// \brief store for each dimension \f$ id,  0 \le k \le  M, 0 \le i \le M \f$ , \f$ \frac{1}{\kappa_k} \rho_i  L_k( \eta_i) \f$
    std::shared_ptr< std::vector< Eigen::ArrayXXd >  > m_fInterpol ;
    Eigen::ArrayXi m_firstPoints ; /// in each dimension the number of the first collocation points inside the grid
    Eigen::ArrayXi m_lastPoints ; /// in each direction the number of the last collocation points present in the grid in the last mesh (in the given direction)

public :

    /// \brief Default constructor
    RegularLegendreGrid(): m_nbPoints(0) {}

    /// \brief Main constructor
    /// \param p_lowValues  in each dimension minimal value of the grid
    /// \param p_step       in each dimension the step size
    /// \param p_nbStep     in each dimension the number of steps
    /// \param p_poly       in each direction degree for Gauss Lobatto polynomial
    RegularLegendreGrid(const Eigen::ArrayXd &p_lowValues, const Eigen::ArrayXd &p_step, const  Eigen::ArrayXi &p_nbStep, const Eigen::ArrayXi &p_poly);

    /// \brief Main constructor
    /// \param p_lowValues       in each dimension minimal value of the grid
    /// \param p_step            in each dimension the step size
    /// \param p_nbStep          in each dimension the number of steps
    /// \param p_gllPoints       in each direction Gauss Legendre Lobatto points
    /// \param p_fInterpol       store in each dimension the weights for interpolation
    /// \param p_firstPoints     first collocation in each dimension belonging to the grid
    /// \param p_lastPoints       last collocation in each dimension belonging to the grid
    RegularLegendreGrid(const Eigen::ArrayXd &p_lowValues, const Eigen::ArrayXd &p_step, const  Eigen::ArrayXi &p_nbStep,  const   std::vector< Eigen::ArrayXd >   &p_gllPoints,
                        std::shared_ptr< std::vector< Eigen::ArrayXXd >  > p_fInterpol, const Eigen::ArrayXi &p_firstPoints,  const Eigen::ArrayXi &p_lastPoints);

    /// \name get position coordinate of a point and generic information
    ///@{
    /// lower coordinate of a point (iCoord) in the grid such that each coordinates lies in the mesh defines by \f$ iCoord \f$  and \f$ iCoord+ npoly \f$
    Eigen::ArrayXi lowerPositionCoord(const Eigen::Ref<const Eigen::ArrayXd > &p_point) const;


    /// upper coordinate of a point (iCoord) in the grid such that each coordinates lies in the mesh defines by \f$ iCoord-npoly \f$  and \f$ iCoord \f$
    Eigen::ArrayXi upperPositionCoord(const Eigen::Ref<const Eigen::ArrayXd > &p_point) const;

    ///  transform integer coordinates  to real coordinates
    Eigen::ArrayXd  getCoordinateFromIntCoord(const Eigen::ArrayXi &p_icoord) const;
    ///@}

    /// \brief get sub grid
    /// \param  p_mesh for each dimension give the first point of the mesh and the first outside of the domain
    std::shared_ptr<FullGrid> getSubGrid(const Eigen::Array< std::array<int, 2>, Eigen::Dynamic, 1> &p_mesh) const;

    /// \brief Coordinate in each direction to global (integers)
    /// \param p_iCoord  coordinate (in integer)
    int intCoordPerDimToGlobal(const Eigen::ArrayXi &p_iCoord) const
    {
        int iret = p_iCoord(0);
        int idec = 1;
        for (int id = 1 ; id < p_iCoord.size(); ++id)
        {
            idec *= m_dimensions(id - 1);
            iret += p_iCoord(id) * idec;
        }
        assert(iret < m_nbPoints);
        return iret;
    }

    /// \brief get back iterator associated to the grid (multi thread)
    /// \param   p_iThread  Thread number  (for multi thread purpose)
    std::shared_ptr< GridIterator> getGridIteratorInc(const int &p_iThread) const;


    /// \brief get back iterator associated to the grid
    std::shared_ptr< GridIterator> getGridIterator() const ;

    /// \brief  Get back interpolator at a point Interpolate on the grid : here it is a linear interpolator
    /// \param  p_coord   coordinate of the point for interpolation
    /// \return interpolator at the point coord on the grid
    std::shared_ptr<Interpolator> createInterpolator(const Eigen::ArrayXd &p_coord) const;

    /// \brief Get back a spectral operator associated to a whole function
    /// \param p_values   Function value at the grids points
    /// \return  the whole interpolated  value function
    std::shared_ptr<InterpolatorSpectral> createInterpolatorSpectral(const Eigen::ArrayXd &p_values) const ;


    /// \brief Get back the number of points on the meshing
    inline size_t getNbPoints() const
    {
        return m_nbPoints;
    }

    /// \brief Get back Gauss Lobatto Points
    inline  const std::vector< Eigen::ArrayXd > &getGllPoints() const
    {
        return m_gllPoints;
    }

    /// \brief Get back the degrees of the polynomials in each direction
    inline  const Eigen::ArrayXi &getPoly() const
    {
        return m_poly;
    }
    /// \brief Get back the degree of the polynomial approximation in a given dimension :
    /// \param p_idim  given dimension
    inline int getPoly(const int &p_idim) const
    {
        return m_poly(p_idim);
    }

    /// \brief get back fInterpol
    inline  std::shared_ptr< std::vector< Eigen::ArrayXXd >  >  getFInterpol() const
    {
        return m_fInterpol ;
    }

    /// \brief get back Legendre polynomials
    inline  std::shared_ptr< std::array< std::function< double(const double &) >, 11 > >  getLegendre() const
    {
        return m_legendre ;
    }

    /// \brief get back first collocation point for first mesh in each direction
    inline  const Eigen::ArrayXi &getFirstPoints() const
    {
        return m_firstPoints;
    }

    /// \brief get back last collocation point for first mesh in each direction
    inline  const Eigen::ArrayXi &getLastPoints() const
    {
        return m_lastPoints;
    }

    /// \brief check comptability of the mesh with the number of points in each direction
    /// \param  p_nbPoints   number of points
    bool checkMeshAndPointCompatibility(const int &p_nbPoints) const
    {
        return (p_nbPoints == m_nbPoints);
    }

    /// \brief Rescale a point in one dimension
    /// \param  p_point  coordinate in a given dimension
    /// \param  p_idim   working dimension
    /// \param  p_coordLoc local coordinate in [-1,1] (return)
    /// \param  p_iCoord local integer coordinate (return)
    void rescalepoint(const double  &p_point, const int &p_idim,  double &p_coordLoc, int &p_iCoord) const;
};
}
#endif
