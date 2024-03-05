
#ifndef SPARSEGRIDNOBOUND_H
#define SPARSEGRIDNOBOUND_H
#include <iostream>
#include <functional>
#include <Eigen/Dense>
#include "libflow/core/sparse/sparseGridTypes.h"
#include "libflow/core/sparse/sparseGridUtils.h"
#include "libflow/core/sparse/sparseGridCommon.h"

/** \file sparseGridNoBound.h
 *  \brief Regroup some functions use in sparse grid when no boundary points are present
 *  \author Xavier Warin
 */

namespace libflow
{

/// \brief Recursive construction for Data structure
/// \param p_levelCurrent                      Current index of levels of the point
/// \param p_positionCurrent                   Current position of the point ,at the given level
/// \param p_iterDataStructureCurrent          iterator in the data structure
/// \param p_idim                              Current dimension
/// \param p_dataSet                           Data structure with all the points
/// \param p_ipoint                            Point number
void recursiveSparseConstructionNoBound(Eigen::ArrayXc &p_levelCurrent,
                                        Eigen::ArrayXui &p_positionCurrent,
                                        SparseSet::iterator &p_iterDataStructureCurrent,
                                        const unsigned short  int &p_idim,
                                        SparseSet &p_dataSet,
                                        size_t &p_ipoint);

/// \brief Initial  construction for Data structure
///        The level  \f$ (l_i)_{i=1,NDIM} \f$ kept satisfies
///       \f$ \sum_{i=1}^{NDIM} l_i \alpha_i \le \f$  levelMax \f$ + NDIM -1 \f$
/// \param p_levelMax          Max level for the sparse grid
/// \param p_alpha             weight used for anisotropic sparse grids
/// \param p_dataSet           Data structure with all the points
/// \param p_ipoint            Point number
void initialSparseConstructionNoBound(const unsigned int &p_levelMax,
                                      const Eigen::ArrayXd &p_alpha,
                                      SparseSet   &p_dataSet,
                                      size_t     &p_ipoint);

/// \brief Initial  construction for Data structure
///        The level  \f$ (l_i)_{i=1,NDIM} \f$ kept satisfies
///       \f$  l_i \alpha_i \le \f$  levelMax
/// \param p_levelMax          Max level for the full  grid
/// \param p_alpha             weight used for anisotropic full grids
/// \param p_dataSet           Data structure with all the points
/// \param p_ipoint            Point number
void initialFullConstructionNoBound(const unsigned int &p_levelMax,
                                    const Eigen::ArrayXd &p_alpha,
                                    SparseSet   &p_dataSet,
                                    size_t     &p_ipoint);

/// \brief Explore dimension for hierarchization dehierarchization  iterating from root in a given direction (1D call)  when no boundary points
/// \param p_levelCurrent                 Current level of the point
/// \param p_positionCurrent              Current position  of the point
/// \param p_iterLevel                    Iterator on current level
/// \param p_idim                         Current dimension where Hierarchization is achieved
/// \param p_vecOtherDim                  Vector of dimensions different from p_idim
/// \param p_idimRemain                   Number of dim to explore
/// \param p_dataSet                      Data structure with all the points
/// \param p_source                       function value to transform (either Hierarchized or Dehierarchized)
/// \param p_output                            result
template< class HierDehier, class T, class TT>
void recursiveExploration1DNoBound(Eigen::ArrayXc &p_levelCurrent,
                                   Eigen::ArrayXui &p_positionCurrent,
                                   const SparseSet::const_iterator &p_iterLevel,
                                   const unsigned int &p_idim,
                                   const SparseSet &p_dataSet,
                                   const Eigen::ArrayXui &p_vecOtherDim,
                                   const unsigned int   &p_idimRemain,
                                   const TT &p_source,
                                   TT &p_output)
{
    if (p_iterLevel == p_dataSet.end())
        return ;

    // achieve 1D Hierarchization for the current node and given direction
    HierDehier().template operator()<T, TT>(p_levelCurrent, p_positionCurrent, p_iterLevel, p_idim, p_dataSet, p_source, p_output);

    // recursive if current is  not a leaf
    for (size_t idd = 0 ; idd < p_idimRemain; ++idd)
    {
        // dimension to dive into
        int idimDive = p_vecOtherDim(idd);

        unsigned int oldPosition = p_positionCurrent(idimDive);
        char oldLevel = p_levelCurrent(idimDive);
        // child level
        p_levelCurrent(idimDive) = oldLevel + 1 ;
        const typename SparseSet::const_iterator  iterLevelChild = p_dataSet.find(p_levelCurrent);

        // left
        p_positionCurrent(idimDive) = 2 * oldPosition;
        // recursive (de)hierarchization from this point
        recursiveExploration1DNoBound<HierDehier, T, TT>(p_levelCurrent, p_positionCurrent, iterLevelChild, p_idim, p_dataSet, p_vecOtherDim, idd + 1, p_source, p_output);
        // right
        p_positionCurrent(idimDive) = 2 * oldPosition + 1;
        // recursive (de)hierarchization from this point
        recursiveExploration1DNoBound<HierDehier, T, TT>(p_levelCurrent, p_positionCurrent, iterLevelChild, p_idim, p_dataSet, p_vecOtherDim, idd + 1, p_source, p_output);

        p_levelCurrent(idimDive) = oldLevel;
        p_positionCurrent(idimDive) = oldPosition;
    }
}


/// \brief global hierarchization or dehierarchization when no boundary points
/// \param p_dataSet      Data structure with all the points
/// \param p_idim         dimension of the problem
/// \param p_output       values changed from nodal to hierarchical or vice versa
template<  class HierDehier, class T, class TT >
void ExplorationNoBound(const  SparseSet &p_dataSet, const int &p_idim,  TT &p_output)
{
    // get root
    Eigen::ArrayXc  rootLevel(p_idim) ;
    Eigen::ArrayXui rootPosition(p_idim);
    HierDehier().get_root(rootLevel, rootPosition);
    SparseSet::const_iterator iterRoot = p_dataSet.find(rootLevel);
    Eigen::ArrayXui  vecOtherDim(p_idim);
    for (unsigned int  id = 0 ; id < static_cast<unsigned int>(p_idim); ++id)
    {
        int ipos  = 0 ;
        for (unsigned short  idd = 0 ; idd < static_cast<unsigned short>(p_idim); ++idd)
            if (idd != id)
                vecOtherDim(ipos++) = idd ;
        recursiveExploration1DNoBound<HierDehier, T, TT> (rootLevel, rootPosition, iterRoot, id, p_dataSet, vecOtherDim, p_idim - 1, p_output, p_output) ;
    }
}


/// \brief  Create basis function that can be used in interpolation
///         During interpolations, it permits to avoid to check the level and the position to decide which function to use
///  \param p_levelMax       level max associated to the sparse grid
///  \param p_weight         anisotropy of the sparse grid
///  \param p_degree         degree of the sparse grid
///  \param p_functionScal   for each position, for  the level used, defines the  basis function
void createBasisFunctionNoBound(const int &p_levelMax, const Eigen::ArrayXd &p_weight, const int &p_degree, std::vector< std::vector< std::function< double(const double &) > > > &p_functionScal);


/// \brief Evaluation of a  function by interpolation (generic for Linear, Quadratic and Cubic) (without  boundary  points)
/// We suppose here that the son have been calculated for each node (to accelerate resolution)
/// Templates are here to define interpolation functions
/// \param p_iPoint                 Point number
/// \param p_xMiddle                Position in [0,1] of current node in each dimension
/// \param p_dx                     Semi mesh size
/// \param p_x                      evaluation point
/// \param p_idimMin                minimal dimension search (to avoid to go twice at same node)
/// \param p_funcVal                Function basis values at current node for all dimensions
/// \param p_son                    Son array (first dimension is the node number, second is the dimension , 0 in array corresponds to left, 1 to right)
/// \param p_hierarValues             Array of Hierarchical values
template< class basisFunctionCenter, class basisFunctionLeft, class  basisFunctionRight,  class T, class TT >
T recursiveEvaluationWithSonNoBound(const int &p_iPoint,
                                    Eigen::ArrayXd &p_xMiddle,
                                    Eigen::ArrayXd &p_dx,
                                    const Eigen::ArrayXd &p_x,
                                    const unsigned short int &p_idimMin,
                                    Eigen::ArrayXd &p_funcVal,
                                    const Eigen::Array< std::array<int, 2 >, Eigen::Dynamic, Eigen::Dynamic >    &p_son,
                                    const TT &p_hierarValues)
{

    T res =  DoubleOrArray()(p_hierarValues, p_iPoint) * p_funcVal.prod();

    // iterate on dimension
    for (int idim = 0 ; idim < p_idimMin ;  ++idim)
    {
        // utilitarian
        double olfFuncVal = p_funcVal(idim);
        double oldXMiddle = p_xMiddle(idim);
        double oldDx = p_dx(idim);
        double dxModified = 0.5 * p_dx(idim) ;
        p_dx(idim) = dxModified;
        // semi size mesh
        if (p_x(idim) <= p_xMiddle(idim))
        {
            if (p_son(p_iPoint, idim)[0] >= 0)
            {
                // go left
                p_xMiddle(idim) -= dxModified;
                // first case if boundary left
                if (almostEqual<double>(oldXMiddle, oldDx, 10))
                    p_funcVal(idim) = 2 * LinearHatValue(0., 1. / oldDx)(p_x(idim));
                else if (almostEqual<double>(oldXMiddle, 1 - oldDx, 10))
                    // Level 3 (starting from 1)
                    p_funcVal(idim) = basisFunctionCenter(p_xMiddle(idim), 1. / dxModified)(p_x(idim));
                else
                    // level > 3)
                    p_funcVal(idim) = basisFunctionLeft(p_xMiddle(idim), 1. / dxModified)(p_x(idim));
                // add contribution
                res += recursiveEvaluationWithSonNoBound< basisFunctionCenter, basisFunctionLeft, basisFunctionRight, T, TT >(p_son(p_iPoint, idim)[0], p_xMiddle, p_dx,
                        p_x, idim + 1, p_funcVal, p_son, p_hierarValues);
            }
        }
        else
        {
            if (p_son(p_iPoint, idim)[1] >= 0)
            {
                // go right
                p_xMiddle(idim) += dxModified;
                if (almostEqual<double>(oldXMiddle, 1 - oldDx, 10))
                    p_funcVal(idim) = 2 * LinearHatValue(1., 1. / oldDx)(p_x(idim));
                else if (almostEqual<double>(oldXMiddle, oldDx, 10))
                    p_funcVal(idim) = basisFunctionCenter(p_xMiddle(idim), 1. / dxModified)(p_x(idim));
                else
                    p_funcVal(idim) = basisFunctionRight(p_xMiddle(idim), 1. / dxModified)(p_x(idim));
                // add contribution
                res += recursiveEvaluationWithSonNoBound< basisFunctionCenter, basisFunctionLeft, basisFunctionRight, T, TT>(p_son(p_iPoint, idim)[1], p_xMiddle, p_dx, p_x,
                        idim + 1, p_funcVal, p_son, p_hierarValues);
            }
        }
        p_funcVal(idim) = olfFuncVal;
        p_xMiddle(idim) = oldXMiddle;
        p_dx(idim) = oldDx;

    }
    return res ;
}



///  \brief Generic evaluation with bounds
///  \param p_x                        evaluation point coordinates
///  \param p_iBase                    Number of the base point of the structure
///  \param p_son                      Son array (first dimension is the node number, second is the dimension , 0 in array corresponds to left, 1 to right)
///  \param p_hierarValues             Array of Hierarchical values
template< class basisFunctionCenter, class basisFunctionLeft, class  basisFunctionRight, class T, class TT >
T globalEvaluationWithSonNoBound(const Eigen::ArrayXd   &p_x,
                                 const int &p_iBase,
                                 const Eigen::Array< std::array<int, 2 >, Eigen::Dynamic, Eigen::Dynamic >    &p_son,
                                 const TT &p_hierarValues)
{
    Eigen::ArrayXd dx = Eigen::ArrayXd::Constant(p_x.size(), 0.5);
    Eigen::ArrayXd  xMiddle = Eigen::ArrayXd::Constant(p_x.size(), 0.5);
    Eigen::ArrayXd  funcVal =  Eigen::ArrayXd::Constant(p_x.size(), 1.);

    return recursiveEvaluationWithSonNoBound< basisFunctionCenter, basisFunctionLeft, basisFunctionRight, T, TT >
           (p_iBase, xMiddle, dx, p_x, p_x.size(), funcVal,  p_son, p_hierarValues) ;
}



///  \brief Pre calculate the son of the point in all dimension
///  \param p_dataSet         Data structure
///  \param p_idim            Dimension of the problem
///  \param p_nbPoint         Number of points in data structure
///  \param p_son             Son array (nb points,NDIM,Left/Right)
///  \return Base point value
int  sonEvaluationNoBound(const SparseSet   &p_dataSet, const int &p_idim,
                          const int   &p_nbPoint,
                          Eigen::Array< std::array<int, 2 >, Eigen::Dynamic, Eigen::Dynamic > &p_son);

///@}
}
#endif /* SPARSEGRIDNOBOUND.H */
