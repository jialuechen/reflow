// Copyright (C) 2016 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#ifndef SPARSEGRIDLINBOUND_H
#define SPARSEGRIDLINBOUND_H
#include <Eigen/Dense>
#include "libflow/core/sparse/sparseGridTypes.h"
#include "libflow/core/sparse/SparseGridHierarDehierarBound.h"


/** \file SparseGridLinBound.h
 * \brief Regroup hierarchization and dehierarchization for linear sparse grids
 * \author Xavier Warin
 */
namespace libflow
{
/// \defgroup linearSparseBound Linear Hierarchization and Deheriarchization with boundary points
/// \brief Regroup function used in hierarchization and dehierarchization for sparse grids with  points on the boundary and a linear approximation per mesh
///@{



/// \class  Hierar1DLinBound   SparseGridLinBound.h
/// Hierarchization
class Hierar1DLinBound : public HierarDehierarBound
{

protected :

    /// \brief Hierarchization in given dimension in 1D
    /// \param p_levelCurrent            current index of the point
    /// \param p_positionCurrent         current level of the point
    /// \param p_iterLevel               iterator on level
    /// \param p_idim                    current dimension
    /// \param p_leftParentNodalValue    Left parent nodal value
    /// \param p_rightParentNodalValue   Right parent nodal value
    /// \param p_dataSet                 Data structure with all the points
    /// \param p_nodalValues               Nodal values
    /// \param p_hierarValues              Hierarchical values
    template< class T, class TT>
    void recursive1DHierarchization(Eigen::ArrayXc &p_levelCurrent,
                                    Eigen::ArrayXui &p_positionCurrent,
                                    const  SparseSet::const_iterator &p_iterLevel,
                                    const unsigned int &p_idim,
                                    const T &p_leftParentNodalValue,
                                    const T &p_rightParentNodalValue,
                                    const SparseSet &p_dataSet,
                                    const TT   &p_nodalValues,
                                    TT &p_hierarValues)
    {
        if (p_iterLevel == p_dataSet.end())
            return ;
        // position
        SparseLevel::const_iterator iterPosition = p_iterLevel->second.find(p_positionCurrent);
        if (iterPosition == p_iterLevel->second.end())
            return ;
        // position
        int iposPoint = iterPosition->second;
        T valueMidle =  DoubleOrArray()(p_nodalValues, iposPoint);
        // hierarchization
        DoubleOrArray().affect(p_hierarValues, iposPoint, valueMidle  - 0.5 * (p_leftParentNodalValue + p_rightParentNodalValue));
        char oldLevel = p_levelCurrent(p_idim);
        unsigned int oldPosition = p_positionCurrent(p_idim);
        // child level
        p_levelCurrent(p_idim) += 1;
        SparseSet::const_iterator  iterLevelChild = p_dataSet.find(p_levelCurrent);
        // left
        p_positionCurrent(p_idim) = 2 * oldPosition;
        recursive1DHierarchization<T, TT>(p_levelCurrent, p_positionCurrent,  iterLevelChild, p_idim, p_leftParentNodalValue, valueMidle, p_dataSet, p_nodalValues, p_hierarValues);
        // right
        p_positionCurrent(p_idim) += 1;
        recursive1DHierarchization<T, TT>(p_levelCurrent, p_positionCurrent, iterLevelChild, p_idim,  valueMidle, p_rightParentNodalValue, p_dataSet, p_nodalValues, p_hierarValues);

        p_positionCurrent(p_idim) = oldPosition;
        p_levelCurrent(p_idim) = oldLevel;
    }

    /// \brief Hierarchization in given dimension in 1D
    /// \param  p_levelCurrent                Current level of the point
    /// \param p_positionCurrent              Current position of the point
    /// \param p_iterLevel                    Iterator  on level
    /// \param p_idim                         Current dimension
    /// \param p_leftParentNodalValue         Left parent nodal value
    /// \param p_rightParentNodalValue        Right parent nodal value
    /// \param p_dataSet                      Data structure with all the points
    /// \param p_nodalValues                  Nodal values
    /// \param p_hierarValues                 Hierarchical values
    template< class T, class TT>
    void HierarchizationFirstLevel(Eigen::ArrayXc   &p_levelCurrent,
                                   Eigen::ArrayXui &p_positionCurrent,
                                   const  SparseSet::const_iterator &p_iterLevel,
                                   const unsigned int &p_idim,
                                   const T &p_leftParentNodalValue,
                                   const T &p_rightParentNodalValue,
                                   const SparseSet &p_dataSet,
                                   const TT &p_nodalValues,
                                   TT &p_hierarValues)
    {
        if (p_iterLevel == p_dataSet.end())
            return ;
        // position
        SparseLevel::const_iterator iterPosition = p_iterLevel->second.find(p_positionCurrent);
        if (iterPosition == p_iterLevel->second.end())
            return ;
        // position
        int iposPoint = iterPosition->second;
        T valueMidle = DoubleOrArray()(p_nodalValues, iposPoint);
        // hierarchization
        DoubleOrArray().affect(p_hierarValues, iposPoint, valueMidle  - 0.5 * (p_leftParentNodalValue + p_rightParentNodalValue));

        char oldLevel = p_levelCurrent(p_idim);
        unsigned int oldPosition = p_positionCurrent(p_idim);

        // child level
        p_levelCurrent(p_idim) += 1;
        SparseSet::const_iterator  iterLevelChild = p_dataSet.find(p_levelCurrent);
        // left
        p_positionCurrent(p_idim) = 0;
        recursive1DHierarchization<T, TT>(p_levelCurrent, p_positionCurrent, iterLevelChild, p_idim, p_leftParentNodalValue, valueMidle, p_dataSet, p_nodalValues, p_hierarValues);
        // right
        p_positionCurrent(p_idim) = 1;
        recursive1DHierarchization<T, TT>(p_levelCurrent, p_positionCurrent,  iterLevelChild, p_idim,  valueMidle, p_rightParentNodalValue, p_dataSet, p_nodalValues, p_hierarValues);

        p_positionCurrent(p_idim) = oldPosition;
        p_levelCurrent(p_idim) = oldLevel;

    }
public :

    // default
    Hierar1DLinBound() {}

    // operator for 1D Hierarchization
    /// \brief Hierarchization in given dimension in 1D
    /// \param p_levelCurrent            Current level of the point
    /// \param p_positionCurrent         Current position  of the point in the current level
    /// \param p_iterLevel               Iterator on current level
    /// \param p_idim                    Current dimension
    /// \param p_dataSet                 Data structure with all the points
    /// \param p_nodalValues             Nodal values
    /// \param p_hierarValues            Hierarchical values
    template< class T, class TT>
    void operator()(Eigen::ArrayXc &p_levelCurrent,
                    Eigen::ArrayXui &p_positionCurrent,
                    const  SparseSet::const_iterator &p_iterLevel,
                    const unsigned int     &p_idim,
                    const SparseSet       &p_dataSet,
                    const TT &p_nodalValues,
                    TT        &p_hierarValues)
    {
        // left and right value
        T leftValue  ;
        T rightValue  ;
        Eigen::ArrayXui  leftBound(p_positionCurrent) ;
        leftBound(p_idim) = 0;
        SparseLevel::const_iterator iterLeft = p_iterLevel->second.find(leftBound);
        if (iterLeft != p_iterLevel->second.end())
            leftValue = DoubleOrArray()(p_nodalValues, iterLeft->second);
        Eigen::ArrayXui  rightBound(p_positionCurrent);
        rightBound(p_idim) = 2;
        SparseLevel::const_iterator iterRight = p_iterLevel->second.find(rightBound);
        if (iterRight != p_iterLevel->second.end())
            rightValue = DoubleOrArray()(p_nodalValues, iterRight->second);
        // First Level Hierarchization
        HierarchizationFirstLevel<T, TT>(p_levelCurrent, p_positionCurrent, p_iterLevel, p_idim, leftValue, rightValue, p_dataSet, p_nodalValues, p_hierarValues);

    }

};


/// \class  Dehierar1DLinBound   SparseGridLinBound.h
/// Dehierarchization
class Dehierar1DLinBound : public HierarDehierarBound
{
protected :
    /// \brief Dehierarchization 1D in given dimension
    /// \param p_levelCurrent            Current level of the point
    /// \param p_positionCurrent         Current position of the point in the current level
    /// \param p_iterLevel               Iterator on current level
    /// \param p_idim                    Current dimension
    /// \param p_leftParentHierarValue   Left parent nodal value
    /// \param p_rightParentHierarValue    Left parent nodal value
    /// \param p_dataSet                 Data structure with all the points
    /// \param p_hierarValues            Hierarchical values
    /// \param p_nodalValues             Nodal values
    template< class T, class TT>
    void recursive1DDehierarchization(Eigen::ArrayXc &p_levelCurrent,
                                      Eigen::ArrayXui &p_positionCurrent,
                                      const  SparseSet::const_iterator &p_iterLevel,
                                      const unsigned int &p_idim,
                                      const T &p_leftParentHierarValue,
                                      const T &p_rightParentHierarValue,
                                      const SparseSet &p_dataSet,
                                      const TT &p_hierarValues,
                                      TT &p_nodalValues)
    {
        if (p_iterLevel == p_dataSet.end())
            return ;
        // position
        SparseLevel::const_iterator iterPosition = p_iterLevel->second.find(p_positionCurrent);
        if (iterPosition == p_iterLevel->second.end())
            return ;
        // position
        int iposPoint = iterPosition->second;
        T valueMidle = DoubleOrArray()(p_hierarValues, iposPoint);

        // do dehierarchization
        valueMidle += 0.5 * (p_leftParentHierarValue + p_rightParentHierarValue);
        DoubleOrArray().affect(p_nodalValues, iposPoint, valueMidle);
        char oldLevel = p_levelCurrent(p_idim);
        unsigned int oldPosition = p_positionCurrent(p_idim);
        // child level
        p_levelCurrent(p_idim) += 1;
        SparseSet::const_iterator  iterLevelChild = p_dataSet.find(p_levelCurrent);
        // left
        p_positionCurrent(p_idim) = 2 * oldPosition;
        recursive1DDehierarchization<T, TT>(p_levelCurrent, p_positionCurrent, iterLevelChild, p_idim, p_leftParentHierarValue, valueMidle, p_dataSet, p_hierarValues, p_nodalValues);
        // right
        p_positionCurrent(p_idim) += 1;
        recursive1DDehierarchization<T, TT>(p_levelCurrent, p_positionCurrent,  iterLevelChild, p_idim,  valueMidle, p_rightParentHierarValue,  p_dataSet, p_hierarValues, p_nodalValues);

        p_positionCurrent(p_idim) = oldPosition;
        p_levelCurrent(p_idim) = oldLevel;
    }

    /// \brief Dehierarchization 1D in given dimension
    /// \param  p_levelCurrent              Current index of the point
    /// \param p_positionCurrent            Current level of the point
    /// \param p_iterLevel                  Iteration on level
    /// \param p_idim                       Current dimension
    /// \param p_leftParentHierarValue      Left parent nodal value
    /// \param p_rightParentHierarValue     Right parent nodal value
    /// \param p_dataSet                    Data structure with all the points
    /// \param p_hierarValues               Hierarchical values
    /// \param p_nodalValues                Nodal values
    template< class T, class TT>
    void dehierarchizationFirstLevel(Eigen::ArrayXc   &p_levelCurrent,
                                     Eigen::ArrayXui &p_positionCurrent,
                                     const  SparseSet::const_iterator &p_iterLevel,
                                     const unsigned int &p_idim,
                                     const T &p_leftParentHierarValue,
                                     const T &p_rightParentHierarValue,
                                     const SparseSet &p_dataSet,
                                     const TT &p_hierarValues,
                                     TT &p_nodalValues)
    {
        if (p_iterLevel == p_dataSet.end())
            return ;
        // position
        SparseLevel::const_iterator iterPosition = p_iterLevel->second.find(p_positionCurrent);
        if (iterPosition == p_iterLevel->second.end())
            return ;
        // position
        int iposPoint = iterPosition->second;
        T valueMidle =  DoubleOrArray()(p_hierarValues, iposPoint);

        // do dehierarchization
        valueMidle += 0.5 * (p_leftParentHierarValue + p_rightParentHierarValue);
        DoubleOrArray().affect(p_nodalValues, iposPoint, valueMidle);

        char oldLevel = p_levelCurrent(p_idim);
        unsigned int oldPosition = p_positionCurrent(p_idim);

        // child level
        p_levelCurrent(p_idim) += 1;
        SparseSet::const_iterator  iterLevelChild = p_dataSet.find(p_levelCurrent);
        // left
        p_positionCurrent(p_idim) = 0;
        recursive1DDehierarchization<T, TT>(p_levelCurrent, p_positionCurrent,  iterLevelChild, p_idim, p_leftParentHierarValue, valueMidle, p_dataSet, p_hierarValues, p_nodalValues);
        // right
        p_positionCurrent(p_idim) = 1;
        recursive1DDehierarchization<T, TT>(p_levelCurrent, p_positionCurrent, iterLevelChild, p_idim,  valueMidle, p_rightParentHierarValue,  p_dataSet, p_hierarValues, p_nodalValues);

        p_positionCurrent(p_idim) = oldPosition;
        p_levelCurrent(p_idim) = oldLevel;
    }

public :

    Dehierar1DLinBound() {}

    /// \brief Dehierarchization 1D in given dimension
    /// \param p_levelCurrent       Current level of the point
    /// \param p_positionCurrent    Current  position of the point
    /// \param p_iterLevel          Iterator on current level
    /// \param p_idim               Current dimension
    /// \param p_dataSet            Data structure with all the points
    /// \param p_hierarValues       Hierarchical values
    /// \param p_nodalValues        Nodal values
    template< class T, class TT>
    void operator()(Eigen::ArrayXc &p_levelCurrent,
                    Eigen::ArrayXui &p_positionCurrent,
                    const  SparseSet::const_iterator &p_iterLevel,
                    const unsigned int &p_idim,
                    const SparseSet &p_dataSet,
                    const TT &p_hierarValues,
                    TT &p_nodalValues)
    {
        T leftValue ;
        T rightValue ;
        Eigen::ArrayXui  leftBound(p_positionCurrent) ;
        leftBound(p_idim) = 0;
        SparseLevel::const_iterator iterLeft = p_iterLevel->second.find(leftBound);
        if (iterLeft != p_iterLevel->second.end())
            leftValue = DoubleOrArray()(p_hierarValues, iterLeft->second);
        Eigen::ArrayXui  rightBound(p_positionCurrent);
        rightBound(p_idim) = 2;
        SparseLevel::const_iterator iterRight = p_iterLevel->second.find(rightBound);
        if (iterRight != p_iterLevel->second.end())
            rightValue = DoubleOrArray()(p_hierarValues, iterRight->second);
        // first level dehierarchization
        dehierarchizationFirstLevel<T, TT>(p_levelCurrent, p_positionCurrent, p_iterLevel, p_idim, leftValue, rightValue, p_dataSet, p_hierarValues, p_nodalValues);
    }

};

///@}

}
#endif
