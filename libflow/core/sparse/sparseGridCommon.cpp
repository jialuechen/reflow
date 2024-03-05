// Copyright (C) 2016 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#include  <Eigen/Dense>
#include "libflow/core/sparse/sparseGridTypes.h"
#include "libflow/core/utils/comparisonUtils.h"


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
                        double   p_levelCalc)
{
    if (isLesserOrEqual(p_levelCalc, static_cast<double>(p_levelMax)))
    {
        // create an empty structure for current level
        SparseLevel emptyMap;
        p_dataSet[p_levelCurrent] = emptyMap;
        for (int id = p_idimMin ; id < p_alpha.size() ; ++id)
        {
            Eigen::ArrayXc nextLevel(p_levelCurrent);
            nextLevel[id] += 1;
            createLevelsSparse(nextLevel, id, p_levelMax, p_alpha, p_dataSet, p_levelCalc + p_alpha[id]);
        }
    }
}


/// \brief Create data_struture of levels for full grid
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
                      SparseSet    &p_dataSet)
{
    // great an empty structure for current level
    SparseLevel emptyMap;
    p_dataSet[p_levelCurrent] = emptyMap;
    for (int id = p_idimMin ; id < p_alpha.size() ; ++id)
    {
        Eigen::ArrayXc nextLevel(p_levelCurrent);
        nextLevel(id) += 1;
        if (isLesserOrEqual(nextLevel[id]*p_alpha[id], static_cast<double>(p_levelMax)))
            createLevelsFull(nextLevel, id, p_levelMax, p_alpha, p_dataSet);
    }
}


/// \brief 1 dimensional recursive construction of the grid for 1D
/// \param p_levelCurrent       Current index of the point
/// \param p_positionCurrent    Current level of the point
/// \param p_dataSet            Data structure with all the points (all level already exists)
/// \param p_ipoint             Point number
void sparse1DConstruction(Eigen::ArrayXc &p_levelCurrent,
                          Eigen::ArrayXui &p_positionCurrent,
                          SparseSet &p_dataSet,
                          size_t &p_ipoint)
{
    char oldLevel = p_levelCurrent(0);
    p_levelCurrent(0) += 1;
    SparseSet::iterator iterdataLevel = p_dataSet.find(p_levelCurrent);
    // if following level exists
    if (iterdataLevel != p_dataSet.end())
    {
        unsigned int oldPosition = p_positionCurrent(0);

        // left right
        p_positionCurrent(0) *= 2;
        iterdataLevel->second[p_positionCurrent] = p_ipoint++;
        sparse1DConstruction(p_levelCurrent, p_positionCurrent, p_dataSet, p_ipoint);
        // right
        p_positionCurrent(0) += 1;
        iterdataLevel->second[p_positionCurrent] = p_ipoint++;
        sparse1DConstruction(p_levelCurrent, p_positionCurrent, p_dataSet, p_ipoint);

        p_positionCurrent(0) = oldPosition;
    }
    p_levelCurrent(0) = oldLevel;
}

}
