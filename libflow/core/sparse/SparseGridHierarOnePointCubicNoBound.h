
#ifndef SPARSEGRIDHIERARONEPOINTCUBICNOBOUND_H
#define SPARSEGRIDHIERARONEPOINTCUBICNOBOUND_H
#include <Eigen/Dense>
#include "libflow/core/sparse/sparseGridTypes.h"
#include "libflow/core/sparse/sparseGridUtils.h"
#include "libflow/core/sparse/SparseGridHierarOnePointNoBound.h"

/** \file SparseGridHierarOnePointCubicNoBound.h
 * \brief Class to Hierarchize a single point with data structure with boundary points
 * \author Xavier Warin
 */

namespace libflow
{

/// \class SparseGridHierarOnePointCubicNoBound SparseGridHierarOnePointCubicNoBound.h
/// It permits to hierarchize a single point performing Hierarchization in the successive
/// dimensions : cubic case
template< class T, class TT>
class SparseGridHierarOnePointCubicNoBound : public SparseGridHierarOnePointNoBound
{
    ///  \brief Recursive hierarchization
    /// \param p_levelCurrent     index level for the current point
    /// \param p_positionCurrent   position of the current point
    /// \param p_iterLevel         iterator on the level of the current point in the data structure
    /// \param p_idim              current working dimension
    /// \param p_dataSet           sparse grid data structure
    /// \param p_nodalValues       Array of nodal values depending on point number. Depending on TT, can be a two dimension array. Then each column corresponds to a point.
    /// \param p_bQuad             if true use quadratic approximation, otherwise linear
    /// \param p_bCubic            if true use cubic approximation
    T  recursive1DHierarchization(Eigen::ArrayXc &p_levelCurrent,
                                  Eigen::ArrayXui &p_positionCurrent,
                                  const  SparseSet::const_iterator &p_iterLevel,
                                  const  int &p_idim,
                                  const SparseSet &p_dataSet,
                                  const TT   &p_nodalValues,
                                  const bool &p_bQuad,
                                  const bool &p_bCubic)
    {
        // position
        SparseLevel::const_iterator iterPosition = p_iterLevel->second.find(p_positionCurrent);
        if (p_idim == -1)
        {
            // position
            int iposPoint = iterPosition->second;
            return  DoubleOrArray()(p_nodalValues, iposPoint);
        }
        else
        {
            T  ret = recursive1DHierarchization(p_levelCurrent, p_positionCurrent,  p_iterLevel, p_idim - 1, p_dataSet, p_nodalValues, true, true);
            if (p_levelCurrent(p_idim) > 1)
            {
                // store current configuration
                char currentLevel = p_levelCurrent(p_idim);
                unsigned int currentPosition = p_positionCurrent(p_idim);
                // switch to direct father
                GetDirectFatherNoBound()(p_levelCurrent(p_idim), p_positionCurrent(p_idim));
                // store direct father characteristics
                char fatherLevel = p_levelCurrent(p_idim);
                unsigned int fatherPosition = p_positionCurrent(p_idim);
                // find level iterator
                SparseSet::const_iterator  iterLevelDirectFather = p_dataSet.find(p_levelCurrent);
                T hierarDirect = recursive1DHierarchization(p_levelCurrent, p_positionCurrent, iterLevelDirectFather, p_idim - 1, p_dataSet, p_nodalValues, true, true);
                if (currentLevel == 2)
                    ret -= hierarDirect;
                else
                {
                    // first treat boundary points
                    if ((currentPosition == 0) || (currentPosition == lastNode[currentLevel - 1]))
                    {
                        // extrapolation needed : get grand father
                        GetDirectFatherNoBound()(p_levelCurrent(p_idim), p_positionCurrent(p_idim));
                        SparseSet::const_iterator  iterLevelDirectGrandFather = p_dataSet.find(p_levelCurrent);
                        T hierarGrandDirect = recursive1DHierarchization(p_levelCurrent, p_positionCurrent, iterLevelDirectGrandFather, p_idim - 1, p_dataSet, p_nodalValues, true, true);
                        T  hierarNonDirect = 2 * hierarDirect - hierarGrandDirect ;
                        ret -=  0.5 * (hierarDirect + hierarNonDirect);
                    }
                    else
                    {
                        // first linear approximation
                        // non direct father  level and position
                        p_levelCurrent(p_idim) = currentLevel;
                        p_positionCurrent(p_idim) = currentPosition;
                        GetNonDirectFatherNoBound()(p_levelCurrent(p_idim), p_positionCurrent(p_idim));
                        // find level iterator
                        SparseSet::const_iterator  iterLevelNonDirectFather = p_dataSet.find(p_levelCurrent);
                        T hierarNonDirect = recursive1DHierarchization(p_levelCurrent, p_positionCurrent, iterLevelNonDirectFather, p_idim - 1, p_dataSet, p_nodalValues, true, true);
                        ret -=  0.5 * (hierarDirect + hierarNonDirect);
                        if ((currentPosition > 1) && (currentPosition < lastNode[currentLevel - 1] - 1))
                        {
                            if (p_bQuad)
                            {
                                // hierarchize in the  same dimension the direct father
                                p_levelCurrent(p_idim) = fatherLevel;
                                p_positionCurrent(p_idim) = fatherPosition;
                                T hierarDirectFather = recursive1DHierarchization(p_levelCurrent, p_positionCurrent, iterLevelDirectFather, p_idim, p_dataSet, p_nodalValues, false, false);
                                ret -= 0.25 * hierarDirectFather;
                                if ((currentPosition > 3) && (currentPosition < lastNode[currentLevel - 1] - 3))
                                    if (p_bCubic)
                                    {
                                        int iBaseType = iNodeToFunc[currentPosition % 4];
                                        // quadratic  father hierarchization
                                        T hierarDirectQuadFather = recursive1DHierarchization(p_levelCurrent, p_positionCurrent, iterLevelDirectFather, p_idim, p_dataSet, p_nodalValues, true, false);
                                        ret +=  hierarDirectQuadFather * weightQuadraticParent[iBaseType];
                                    }
                            }
                        }
                    }
                }
                // go back to initial state
                p_levelCurrent(p_idim) = currentLevel;
                p_positionCurrent(p_idim) = currentPosition;
            }
            return ret ;
        }
    }

public :

    /// \brief  Hierarchize a single point (existing in the data structure)
    /// \param p_levelCurrent     index level for the current point
    /// \param p_positionCurrent   position of the current point
    /// \param p_dataSet           sparse grid data structure
    /// \param p_nodalValues       Array of nodal values depending on point number. Depending on TT, can be a two dimension array. Then each column corresponds to a point.
    T operator()(const Eigen::ArrayXc &p_levelCurrent,
                 const Eigen::ArrayXui &p_positionCurrent,
                 const SparseSet &p_dataSet,
                 const TT   &p_nodalValues)
    {
        // get iterator for the level in the data structure
        SparseSet::const_iterator  iterLevel = p_dataSet.find(p_levelCurrent);
        // copy level and position
        Eigen::ArrayXc  levelCurrent = p_levelCurrent;
        Eigen::ArrayXui positionCurrent = p_positionCurrent;
        return recursive1DHierarchization(levelCurrent, positionCurrent, iterLevel, p_levelCurrent.size() - 1, p_dataSet, p_nodalValues, true, true);
    }
};
}
#endif /* SPARSEGRIDHIERARONEPOINTCUBICNOBOUND_H */
