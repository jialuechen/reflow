
#ifndef  SPARSEGRIDCOMMON_H
#define  SPARSEGRIDCOMMON_H
#include  <Eigen/Dense>
#include "libflow/core/sparse/sparseGridTypes.h"
#include "libflow/core/sparse/sparseGridUtils.h"
#include "libflow/core/utils/comparisonUtils.h"


/** \file sparseGridCommon.h
 * \brief Regroup some functions used in sparse grids whether you use classical sparse grids with boundaries or
 *        simplified one suppressing boundaries.
 * \author Xavier Warin
 */

namespace libflow
{
/// \defgroup sparseCommon   sparse grid common functions
/// \brief Regroup some functions used in different sparse grids approximation
/// @{

/// \brief Create data structure of levels par sparse grids
///        The level  \f$ (l_i)_{i=1,NDIM} \f$ kept satisfies
///        \f$ \sum_{i=1}^{NDIM} l_i \alpha_i \le \f$  levelMax
/// \param   p_levelCurrent      current multilevel
/// \param   p_idimMin           minimal dimension treated
/// \param   p_levelMax          max level
/// \param   p_alpha             weight used for anisotropic sparse grids
/// \param   p_dataSet           data_structure
/// \param   p_levelCalc         utilitarian to store the current sum  \f$ \sum_{i=1}^{NDIM} l_i \alpha_i \f$
void createLevelsSparse(const Eigen::ArrayXc &p_levelCurrent,
                        const int   &p_idimMin,
                        const int   &p_levelMax,
                        const Eigen::ArrayXd &p_alpha,
                        SparseSet    &p_dataSet,
                        double   p_levelCalc);

/// \brief Create data_structure of levels for full grid
///        The levels  \f$ (l_i)_{i=1,NDIM} \f$ satisfies :
///         \f$ l_i  \alpha_i \le \f$ levelMax
///
/// \param   p_levelCurrent      current multilevel
/// \param   p_idimMin           minimal dimension treated
/// \param   p_levelMax          max level
/// \param   p_alpha             weight used for anisotropic sparse grids
/// \param   p_dataSet           data_structure
void createLevelsFull(const Eigen::ArrayXc &p_levelCurrent,
                      const int   &p_idimMin,
                      const int   &p_levelMax,
                      const Eigen::ArrayXd &p_alpha,
                      SparseSet    &p_dataSet);

/// \brief 1 dimensional recursive construction of the grid for 1D
/// \param p_levelCurrent       Current index of the point
/// \param p_positionCurrent    Current level of the point
/// \param p_dataSet            Data structure with all the points (all level already exists)
/// \param p_ipoint             Point number
void sparse1DConstruction(Eigen::ArrayXc &p_levelCurrent,
                          Eigen::ArrayXui &p_positionCurrent,
                          SparseSet &p_dataSet,
                          size_t &p_ipoint);

/// \brief  Create a a new data structure from a first one  with  holes in levels
///         and modify hierarchized values accordingly
/// \param p_dataSet          first data set with "holes "in levels
/// \param p_hierarchized    hierarchized values
/// \param p_valuesFunction  an array storing the nodal values (modified on the new struture)
/// \param p_newDataSet      new data set
/// \param p_newHierarchized  array of hierarchized values
/// \param p_newValuesFunction array of nodal values
template< class T >
void modifyDataSetAndHierachized(const SparseSet    &p_dataSet,
                                 const T &p_hierarchized,
                                 const T &p_valuesFunction,
                                 SparseSet &p_newDataSet,
                                 T   &p_newHierarchized,
                                 T &p_newValuesFunction)
{
    DoubleOrArray().resize(p_newHierarchized, p_hierarchized.size());
    DoubleOrArray().resize(p_newValuesFunction, p_hierarchized.size());
    int nCount = 0;
    for (auto level :  p_dataSet)
    {
        SparseLevel levelLoc;
        for (auto position : level.second)
        {
            levelLoc[position.first] = nCount;
            DoubleOrArray().affect(p_newHierarchized, nCount, DoubleOrArray()(p_hierarchized, position.second));
            DoubleOrArray().affect(p_newValuesFunction, nCount++, DoubleOrArray()(p_valuesFunction, position.second));
        }
        p_newDataSet[level.first] = levelLoc;
    }
    DoubleOrArray().resize(p_newHierarchized, nCount);
    DoubleOrArray().resize(p_newValuesFunction, nCount);
}


///@}
}
#endif /* SPARSEGRIDCOMMON.H */
