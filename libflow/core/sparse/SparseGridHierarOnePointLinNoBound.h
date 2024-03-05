
#ifndef SPARSEGRIDHIERARONEPOINTLINNOBOUND_H
#define SPARSEGRIDHIERARONEPOINTLINNOBOUND_H
#include <Eigen/Dense>
#include "libflow/core/sparse/sparseGridTypes.h"
#include "libflow/core/sparse/sparseGridUtils.h"
#include "libflow/core/sparse/SparseGridHierarOnePointNoBound.h"

/** \file SparseGridHierarOnePointLinNoBound.h
 * \brief Class to Hierarchize a single point with data structure without  boundary points
 * \author Xavier Warin
 */

namespace libflow
{

/// \class SparseGridHierarOnePointLinNoBound SparseGridHierarOnePointLinNoBound.h
/// It permits to hierarchize a single point performing Hierarchization in the successive
/// dimensions : linear case, no boundary points, approximation of boundary conditions
template< class T, class TT>
class SparseGridHierarOnePointLinNoBound : public SparseGridHierarOnePointNoBound
{

///  \brief Recursive hierarchization
    /// \param p_levelCurrent     index level for the current point
    /// \param p_positionCurrent   position of the current point
    /// \param p_iterLevel         iterator on the level of the current point in the data structure
    /// \param p_idim              current working dimension
    /// \param p_dataSet           sparse grid data structure
    /// \param p_nodalValues       Array of nodal values depending on point number. Depending on TT, can be a two dimension array. Then each column corresponds to a point.
    T  recursive1DHierarchization(Eigen::ArrayXc &p_levelCurrent,
                                  Eigen::ArrayXui &p_positionCurrent,
                                  const  SparseSet::const_iterator &p_iterLevel,
                                  const  int &p_idim,
                                  const  SparseSet &p_dataSet,
                                  const  TT   &p_nodalValues)
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
            T  ret = recursive1DHierarchization(p_levelCurrent, p_positionCurrent,  p_iterLevel, p_idim - 1, p_dataSet, p_nodalValues);
            if (p_levelCurrent(p_idim) > 1)
            {
                // store current configuration
                char currentLevel = p_levelCurrent(p_idim);
                unsigned int currentPosition = p_positionCurrent(p_idim);
                // switch to direct father
                GetDirectFatherNoBound()(p_levelCurrent(p_idim), p_positionCurrent(p_idim));
                // find level iterator
                SparseSet::const_iterator  iterLevelDirectFather = p_dataSet.find(p_levelCurrent);
                T hierarDirect = recursive1DHierarchization(p_levelCurrent, p_positionCurrent, iterLevelDirectFather, p_idim - 1, p_dataSet, p_nodalValues);
                if (currentLevel == 2)
                    ret -= hierarDirect;
                else
                {
                    // not boundary point
                    if ((currentPosition != 0) && (currentPosition != lastNode[currentLevel - 1]))
                    {
                        // non direct father  level and position
                        p_levelCurrent(p_idim) = currentLevel;
                        p_positionCurrent(p_idim) = currentPosition;
                        GetNonDirectFatherNoBound()(p_levelCurrent(p_idim), p_positionCurrent(p_idim));
                        // find level iterator
                        SparseSet::const_iterator  iterLevelNonDirectFather = p_dataSet.find(p_levelCurrent);
                        T hierarNonDirect = recursive1DHierarchization(p_levelCurrent, p_positionCurrent, iterLevelNonDirectFather, p_idim - 1, p_dataSet, p_nodalValues);
                        ret -=  0.5 * (hierarDirect + hierarNonDirect);
                    }
                    else
                    {
                        // use extrapolation with grandfather
                        GetDirectFatherNoBound()(p_levelCurrent(p_idim), p_positionCurrent(p_idim));
                        SparseSet::const_iterator  iterLevelDirectGrandFather = p_dataSet.find(p_levelCurrent);
                        T hierarGrandDirect = recursive1DHierarchization(p_levelCurrent, p_positionCurrent, iterLevelDirectGrandFather, p_idim - 1, p_dataSet, p_nodalValues);
                        T  hierarNonDirect = 2 * hierarDirect - hierarGrandDirect ;
                        ret -=  0.5 * (hierarDirect + hierarNonDirect);
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
        // get iterator for the level in the data Structure
        SparseSet::const_iterator  iterLevel = p_dataSet.find(p_levelCurrent);
        // copy level and position
        Eigen::ArrayXc  levelCurrent = p_levelCurrent;
        Eigen::ArrayXui positionCurrent = p_positionCurrent;
        return recursive1DHierarchization(levelCurrent, positionCurrent, iterLevel, p_levelCurrent.size() - 1, p_dataSet, p_nodalValues);
    }
};
}
#endif /* SPARSEGRIDHIERARONEPOINTLINNOBOUND_H */
