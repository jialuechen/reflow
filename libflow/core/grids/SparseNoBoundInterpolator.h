
#ifndef SPARSENOBOUNDINTERPOLATOR_H
#define SPARSENOBOUNDINTERPOLATOR_H
#include <vector>
#include <Eigen/Dense>
#include "libflow/core/sparse/sparseGridNoBound.h"

namespace libflow
{

/// \class SparseNoBoundInterpolator SparseNoBoundInterpolator.h
/// Sparse interpolation object with boundary points
/// Templated are the basis functions
template<  class basisFunctionCenter, class basisFunctionLeft,  class basisFunctionRight>
class SparseNoBoundInterpolator : public Interpolator
{
private :

    std::shared_ptr< Eigen::Array< std::array<int, 2 >, Eigen::Dynamic, Eigen::Dynamic > > m_son ; ///<  Store the sons of all points
    int m_iBase ; ///< number of the base node in the data structure
    Eigen::ArrayXd m_point ; ///< Point used for interpolation

public :

    /** Default constructor
     */
    SparseNoBoundInterpolator() {}

    /** \brief Constructor
     *  \param p_son                     sons of all points in all directions
     *  \param p_iBase                   Number of the point associated to the base of the data structure
     *  \param p_point                   is the coordinate of the points used for interpolation
     */
    SparseNoBoundInterpolator(const std::shared_ptr< Eigen::Array< std::array<int, 2 >, Eigen::Dynamic, Eigen::Dynamic > > &p_son,
                              const int &p_iBase, const Eigen::ArrayXd &p_point): m_son(p_son), m_iBase(p_iBase), m_point(p_point) { }

    /** \brief  interpolate
     *  \param  p_dataValues   Values of the data on the grid
     *  \return interpolated value
     */
    inline double apply(const Eigen::Ref< const Eigen::ArrayXd > &p_dataValues) const
    {
        return globalEvaluationWithSonNoBound< basisFunctionCenter, basisFunctionLeft, basisFunctionRight, double, Eigen::ArrayXd>(m_point, m_iBase, *m_son, p_dataValues);
    }

    /**  \brief  interpolate and use vectorization
     *  \param  p_dataValues   Values of the data on the grid. Interpolation is achieved for all values in the first dimension
     *  \return interpolated value
     */
    Eigen::ArrayXd applyVec(const Eigen::ArrayXXd &p_dataValues) const
    {
        return  globalEvaluationWithSonNoBound< basisFunctionCenter, basisFunctionLeft, basisFunctionRight, Eigen::ArrayXd, Eigen::ArrayXXd >(m_point, m_iBase, *m_son, p_dataValues);
    }

    /** \brief  Same as above but avoids copy for Numpy eigen mapping due to storage conventions
     *  \param  p_dataValues   Values of the data on the grid. Interpolation is achieved for all values in the first dimension
     *  \return interpolated value
     */
    inline Eigen::ArrayXd applyVecPy(Eigen::Ref< Eigen::ArrayXXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> >  p_dataValues) const override
    {
        return  globalEvaluationWithSonNoBound< basisFunctionCenter, basisFunctionLeft, basisFunctionRight, Eigen::ArrayXd, Eigen::ArrayXXd >(m_point, m_iBase, *m_son, p_dataValues);
    }
};
}
#endif /* SPARSENOBOUNDINTERPOLATOR */
