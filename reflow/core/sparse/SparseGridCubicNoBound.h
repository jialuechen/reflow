
#ifndef SPARSEGRIDCUBICNOBOUND_H
#define SPARSEGRIDCUBICNOBOUND_H
#include <array>
#include <Eigen/Dense>
#include "reflow/core/sparse/sparseGridTypes.h"
#include "reflow/core/sparse/SparseGridHierarDehierarNoBound.h"


/** \file SparseGridCubicNoBound.h
 * \brief Regroup hierarchization and dehierarchization for cubic sparse grids eliminating boundary points
 * \author Xavier Warin
 */

namespace reflow
{
/// \defgroup  cubicSparseNoBound Cubic  Hierarchization and Deheriarchization without  boundary points
/// \brief Regroup function used in hierarchization and dehierarchization for sparse grids without any points on the boundary and a cubic approximation per mesh
///@{

/// \class  Hierar1DCubicNoBound   SparseGridCubicNoBound.h
/// Hierarchization
class Hierar1DCubicNoBound: public HierarDehierarNoBound
{

protected :

    /// \brief Hierarchization in given dimension in 1D : general cubic
    /// \param p_levelCurrent                 Current level of the point
    /// \param p_positionCurrent              Current position of the point
    /// \param p_iterLevel                    Iteration on level
    /// \param p_idim                         Current dimension
    /// \param p_leftParentNodalValue         Left parent nodal value
    /// \param p_rightParentNodalValue        Right parent nodal value
    /// \param p_parentLinearHierarValue      Linear Hierarchical value of parent
    /// \param p_grandParentLinearHierarValue Linear Hierarchical value of grandparent
    /// \param p_dataSet                      Data structure with all the points
    /// \param p_nodalValues                  Nodal values
    /// \param p_hierarValues                 Hierarchical values
    template< class T, class TT>
    void recursive1DHierarchization(Eigen::ArrayXc &p_levelCurrent,
                                    Eigen::ArrayXui &p_positionCurrent,
                                    const  SparseSet::const_iterator &p_iterLevel,
                                    const unsigned int &p_idim,
                                    const T &p_leftParentNodalValue,
                                    const T &p_rightParentNodalValue,
                                    const T &p_parentLinearHierarValue,
                                    const T &p_grandParentLinearHierarValue,
                                    const SparseSet &p_dataSet,
                                    const TT &p_nodalValues,
                                    TT &p_hierarValues)
    {
        if (p_iterLevel == p_dataSet.end())
            return ;
        SparseLevel::const_iterator iterPosition = p_iterLevel->second.find(p_positionCurrent);
        if (iterPosition == p_iterLevel->second.end())
            return ;
        // position
        int iposPoint = iterPosition->second;
        T valueMidle = DoubleOrArray()(p_nodalValues, iposPoint);
        // hierarchization
        T LinearHierar = valueMidle  - 0.5 * (p_leftParentNodalValue + p_rightParentNodalValue);
        int iBaseType = iNodeToFunc[p_positionCurrent(p_idim) % 4];
        // begin by linear
        DoubleOrArray().affect(p_hierarValues, iposPoint, LinearHierar + weightParent[iBaseType]*p_parentLinearHierarValue + weightGrandParent[iBaseType]*p_grandParentLinearHierarValue);

        char oldLevel = p_levelCurrent(p_idim);
        unsigned int oldPosition = p_positionCurrent(p_idim);
        // child level
        p_levelCurrent(p_idim) += 1;
        SparseSet::const_iterator  iterLevelChild = p_dataSet.find(p_levelCurrent);

        // left
        p_positionCurrent(p_idim) = 2 * oldPosition;
        recursive1DHierarchization<T, TT>(p_levelCurrent, p_positionCurrent,  iterLevelChild, p_idim, p_leftParentNodalValue, valueMidle, LinearHierar, p_parentLinearHierarValue, p_dataSet, p_nodalValues,
                                          p_hierarValues);
        // right
        p_positionCurrent(p_idim) += 1;
        recursive1DHierarchization<T, TT>(p_levelCurrent, p_positionCurrent,  iterLevelChild, p_idim,  valueMidle, p_rightParentNodalValue, LinearHierar, p_parentLinearHierarValue, p_dataSet, p_nodalValues,
                                          p_hierarValues);

        p_positionCurrent(p_idim) = oldPosition;
        p_levelCurrent(p_idim) = oldLevel;
    }

    /// \brief Hierarchization in given dimension in 1D quadratic
    /// \param p_levelCurrent                  Current level of the point
    /// \param p_positionCurrent               Current position of the point
    /// \param p_iterLevel                     Iteration on level
    /// \param p_idim                          Current dimension
    /// \param p_leftParentNodalValue          Left parent nodal value
    /// \param p_rightParentNodalValue         Right parent nodal value
    /// \param p_parentLinearHierarValue       Linear Hierarchical value of parent
    /// \param p_dataSet                       Data structure with all the points
    /// \param p_nodalValues                   Nodal values
    /// \param p_iNodeNum                      Node number
    /// \param p_hierarValues                  Hierarchical values
    template< class T, class TT>
    void HierarchizationStep3(Eigen::ArrayXc &p_levelCurrent,
                              Eigen::ArrayXui &p_positionCurrent,
                              const  SparseSet::const_iterator &p_iterLevel,
                              const unsigned int &p_idim,
                              const T &p_leftParentNodalValue,
                              const T &p_rightParentNodalValue,
                              const T &p_parentLinearHierarValue,
                              const SparseSet &p_dataSet,
                              const TT &p_nodalValues,
                              const int &p_iNodeNum,
                              TT &p_hierarValues)
    {
        if (p_iterLevel == p_dataSet.end())
            return ;
        // position
        SparseLevel::const_iterator iterPosition = p_iterLevel->second.find(p_positionCurrent);
        if (iterPosition == p_iterLevel->second.end())
            return ;
        int iposPoint = iterPosition->second;
        T valueMidle = DoubleOrArray()(p_nodalValues, iposPoint);
        // hierarchization
        T LinearHierar = valueMidle  - 0.5 * (p_leftParentNodalValue + p_rightParentNodalValue);
        // quadratic
        DoubleOrArray().affect(p_hierarValues, iposPoint, LinearHierar - 0.25 * p_parentLinearHierarValue) ;

        char oldLevel = p_levelCurrent(p_idim);
        unsigned int oldPosition = p_positionCurrent(p_idim);
        // child level
        p_levelCurrent(p_idim) += 1;
        SparseSet::const_iterator  iterLevelChild = p_dataSet.find(p_levelCurrent);

        // modified left and right values
        T leftParentNodalValueLoc = p_leftParentNodalValue;
        T rightParentNodalValueLoc = p_rightParentNodalValue;
        T parentLinearHierarValueLoc = LinearHierar;

        if (oldPosition == 0)
        {
            // First point for the level
            // Only linear estimation of error in hierarchization
            leftParentNodalValueLoc = 2 * valueMidle - p_rightParentNodalValue;
            // LINEAR
            DoubleOrArray().zero(parentLinearHierarValueLoc, p_nodalValues);
            // son is linear quadratic for Left node
            p_positionCurrent(p_idim) = 2 * oldPosition;
            HierarchizationStep3<T, TT>(p_levelCurrent, p_positionCurrent,  iterLevelChild, p_idim, leftParentNodalValueLoc, valueMidle, parentLinearHierarValueLoc,
                                        p_dataSet, p_nodalValues,  2 * p_iNodeNum, p_hierarValues);
            p_positionCurrent(p_idim) += 1;
            HierarchizationStep3<T, TT>(p_levelCurrent, p_positionCurrent,  iterLevelChild,  p_idim, valueMidle, rightParentNodalValueLoc, parentLinearHierarValueLoc,
                                        p_dataSet, p_nodalValues,  2 * p_iNodeNum + 1, p_hierarValues);
        }
        else if (oldPosition == lastNode[oldLevel - 1])
        {
            // Last point for the level
            // Only linear estimation of error in hierarchization
            rightParentNodalValueLoc = 2 * valueMidle - p_leftParentNodalValue;
            // LINEAR
            DoubleOrArray().zero(parentLinearHierarValueLoc, p_nodalValues);
            // son is linear quadratic for Left node
            p_positionCurrent(p_idim) = 2 * oldPosition;
            HierarchizationStep3<T, TT>(p_levelCurrent, p_positionCurrent,   iterLevelChild, p_idim, leftParentNodalValueLoc, valueMidle, parentLinearHierarValueLoc,
                                        p_dataSet, p_nodalValues,  2 * p_iNodeNum, p_hierarValues);
            p_positionCurrent(p_idim) += 1;
            HierarchizationStep3<T, TT>(p_levelCurrent, p_positionCurrent,    iterLevelChild, p_idim, valueMidle, rightParentNodalValueLoc, parentLinearHierarValueLoc,
                                        p_dataSet, p_nodalValues,  2 * p_iNodeNum + 1, p_hierarValues);
        }
        else if ((oldPosition == 1) || (oldPosition == lastNode[oldLevel - 1] - 1))
        {
            // USE QUADRATIC
            p_positionCurrent(p_idim) = 2 * oldPosition;
            HierarchizationStep3<T, TT>(p_levelCurrent, p_positionCurrent,   iterLevelChild, p_idim, leftParentNodalValueLoc, valueMidle, parentLinearHierarValueLoc,
                                        p_dataSet, p_nodalValues,  2 * p_iNodeNum, p_hierarValues);
            p_positionCurrent(p_idim) += 1;
            HierarchizationStep3<T, TT>(p_levelCurrent, p_positionCurrent,    iterLevelChild, p_idim, valueMidle, rightParentNodalValueLoc, parentLinearHierarValueLoc,
                                        p_dataSet, p_nodalValues,  2 * p_iNodeNum + 1, p_hierarValues);
        }
        else
        {
            T grandParentLinearHierarValueLoc = p_parentLinearHierarValue;
            p_positionCurrent(p_idim) = 2 * oldPosition;
            recursive1DHierarchization<T, TT>(p_levelCurrent, p_positionCurrent,  iterLevelChild, p_idim, leftParentNodalValueLoc, valueMidle, parentLinearHierarValueLoc,
                                              grandParentLinearHierarValueLoc, p_dataSet, p_nodalValues,  p_hierarValues);
            p_positionCurrent(p_idim) += 1;
            recursive1DHierarchization<T, TT>(p_levelCurrent, p_positionCurrent,   iterLevelChild, p_idim, valueMidle, rightParentNodalValueLoc, parentLinearHierarValueLoc,
                                              grandParentLinearHierarValueLoc, p_dataSet, p_nodalValues,  p_hierarValues);
        }
        p_positionCurrent(p_idim) = oldPosition;
        p_levelCurrent(p_idim) = oldLevel;
    }

    /// \brief Hierarchization in given dimension in 1D  linear
    /// \param p_levelCurrent                 Current level of the point
    /// \param p_positionCurrent              Current position of the point
    /// \param p_iterLevel                    Iteration on level
    /// \param p_idim                         Current dimension
    /// \param p_leftParentNodalValue         Left parent nodal value
    /// \param p_rightParentNodalValue        Right parent nodal value
    /// \param p_dataSet                      Data structure with all the points
    /// \param p_nodalValues                  Nodal values
    /// \param p_iNodeNum                     Node number
    /// \param p_hierarValues                 Hierarchical values
    template< class T, class TT>
    void HierarchizationStep2(Eigen::ArrayXc &p_levelCurrent,
                              Eigen::ArrayXui &p_positionCurrent,
                              const  SparseSet::const_iterator &p_iterLevel,
                              const unsigned int &p_idim,
                              const T &p_leftParentNodalValue,
                              const T &p_rightParentNodalValue,
                              const SparseSet &p_dataSet,
                              const TT &p_nodalValues,
                              const int &p_iNodeNum,
                              TT &p_hierarValues)
    {
        if (p_iterLevel == p_dataSet.end())
            return ;
        // position
        SparseLevel::const_iterator iterPosition = p_iterLevel->second.find(p_positionCurrent);
        if (iterPosition == p_iterLevel->second.end())
            return ;
        int iposPoint = iterPosition->second;
        T valueMidle = DoubleOrArray()(p_nodalValues, iposPoint);
        // hierarchization
        T LinearHierar = valueMidle  - 0.5 * (p_leftParentNodalValue + p_rightParentNodalValue);
        // cubic
        DoubleOrArray().affect(p_hierarValues, iposPoint, LinearHierar);

        char oldLevel = p_levelCurrent(p_idim);
        unsigned int oldPosition = p_positionCurrent(p_idim);
        // child level
        p_levelCurrent(p_idim) += 1;
        SparseSet::const_iterator  iterLevelChild = p_dataSet.find(p_levelCurrent);

        // modified left and right values
        T leftParentNodalValueLoc = p_leftParentNodalValue;
        T rightParentNodalValueLoc = p_rightParentNodalValue;
        T parentLinearHierarValueLoc = LinearHierar;
        if (oldPosition == 0)
        {
            // LINEAR
            DoubleOrArray().zero(parentLinearHierarValueLoc, p_nodalValues);
            // cubic  on the boundary not used. Use linear
            leftParentNodalValueLoc = 2 * valueMidle - p_rightParentNodalValue;
            // still linear on left for son
            p_positionCurrent(p_idim) = 2 * oldPosition;
            HierarchizationStep3<T, TT>(p_levelCurrent, p_positionCurrent, iterLevelChild,  p_idim, leftParentNodalValueLoc, valueMidle, parentLinearHierarValueLoc,
                                        p_dataSet, p_nodalValues,  2 * p_iNodeNum, p_hierarValues);
            // quadratic on right for son
            p_positionCurrent(p_idim) += 1;
            HierarchizationStep3<T, TT>(p_levelCurrent, p_positionCurrent, iterLevelChild, p_idim, valueMidle, rightParentNodalValueLoc, parentLinearHierarValueLoc,
                                        p_dataSet, p_nodalValues,  2 * p_iNodeNum + 1, p_hierarValues);

        }
        else if (oldPosition == lastNode[oldLevel - 1])
        {
            // LINEAR
            DoubleOrArray().zero(parentLinearHierarValueLoc, p_nodalValues);
            // quadratic left
            p_positionCurrent(p_idim) = 2 * oldPosition;
            HierarchizationStep3<T, TT>(p_levelCurrent, p_positionCurrent, iterLevelChild, p_idim, leftParentNodalValueLoc, valueMidle, parentLinearHierarValueLoc,
                                        p_dataSet, p_nodalValues,  2 * p_iNodeNum, p_hierarValues);
            // cubic  on the boundary not used. Use linear
            rightParentNodalValueLoc = 2 * valueMidle - p_leftParentNodalValue;
            // linear for right
            p_positionCurrent(p_idim) += 1;
            HierarchizationStep3<T, TT>(p_levelCurrent, p_positionCurrent, iterLevelChild, p_idim, valueMidle, rightParentNodalValueLoc, parentLinearHierarValueLoc,
                                        p_dataSet, p_nodalValues,  2 * p_iNodeNum + 1, p_hierarValues);
        }
        else
        {
            abort();
        }
        p_positionCurrent(p_idim) = oldPosition;
        p_levelCurrent(p_idim) = oldLevel;
    }

    /// \brief Hierarchization in given dimension in 1D (first step  : subtract constant value)
    /// \param p_levelCurrent                 Current level of the point
    /// \param p_positionCurrent              Current position of the point
    /// \param p_iterLevel                    Iteration on level
    /// \param p_idim                         Current dimension
    /// \param p_dataSet                      Data structure with all the points
    /// \param p_nodalValues                  Nodal values
    /// \param p_hierarValues                 Hierarchical values
    template< class T, class TT>
    void HierarchizationStep1(Eigen::ArrayXc &p_levelCurrent,
                              Eigen::ArrayXui &p_positionCurrent,
                              const  SparseSet::const_iterator &p_iterLevel,
                              const unsigned int &p_idim,
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
        int iposPoint = iterPosition->second;
        T valueMidle = DoubleOrArray()(p_nodalValues, iposPoint);
        // hierarchization
        DoubleOrArray().affect(p_hierarValues, iposPoint, valueMidle) ;

        char oldLevel = p_levelCurrent(p_idim);
        unsigned int oldPosition = p_positionCurrent(p_idim);
        // child level
        p_levelCurrent(p_idim) += 1;
        SparseSet::const_iterator  iterLevelChild = p_dataSet.find(p_levelCurrent);

        // modified left and right values
        T leftParentNodalValueLoc = valueMidle;
        T rightParentNodalValueLoc = valueMidle;
        // left
        p_positionCurrent(p_idim) = 2 * oldPosition;
        HierarchizationStep2<T, TT>(p_levelCurrent, p_positionCurrent,  iterLevelChild,  p_idim, leftParentNodalValueLoc, valueMidle,
                                    p_dataSet, p_nodalValues, 0, p_hierarValues);
        // right
        p_positionCurrent(p_idim) += 1;
        HierarchizationStep2<T, TT>(p_levelCurrent, p_positionCurrent, iterLevelChild,  p_idim, valueMidle, rightParentNodalValueLoc,
                                    p_dataSet, p_nodalValues, 1, p_hierarValues);

        p_positionCurrent(p_idim) = oldPosition;
        p_levelCurrent(p_idim) = oldLevel;
    }

public :

    // default
    Hierar1DCubicNoBound() {}

    // operator for 1D Hierarchization
    /// \brief Hierarchization in given dimension in 1D
    /// \param p_levelCurrent                Current index of the point
    /// \param p_positionCurrent             Current level of the point
    /// \param p_iterLevel                     Iterator on current level
    /// \param p_idim                             Current dimension
    /// \param p_dataSet                   Data structure with all the points
    /// \param p_nodalValues      Nodal values
    /// \param p_hierarValues     Hierarchical values
    template< class T, class TT>
    void operator()(Eigen::ArrayXc &p_levelCurrent,
                    Eigen::ArrayXui &p_positionCurrent,
                    const  SparseSet::const_iterator &p_iterLevel,
                    const unsigned int &p_idim,
                    const SparseSet &p_dataSet,
                    const TT &p_nodalValues,
                    TT &p_hierarValues)
    {
        // left and right value
        HierarchizationStep1<T, TT>(p_levelCurrent, p_positionCurrent, p_iterLevel, p_idim, p_dataSet, p_nodalValues, p_hierarValues);

    }
};

/// \class Dehierar1DCubicNoBound SparseGridCubicNoBound.h
///  Dehierarchization for cubic without bound
class Dehierar1DCubicNoBound: public HierarDehierarNoBound
{
protected :

    /// \brief Dehierarchization 1D in given dimension (modified bound) (cubic so available after two level)
    /// \param p_levelCurrent                Current index of the point
    /// \param p_positionCurrent             Current level of the point
    /// \param p_iterLevel                     Iterator on current level
    /// \param p_idim                             Current dimension
    /// \param p_leftParentHierarValue          Left parent nodal value
    /// \param p_rightParentHierarValue         Right parent nodal value
    /// \param p_linearHierarParent             Linear hierarchical value of parent
    /// \param p_linearHierarGrandParent        Linear Hierarchical value of parent
    /// \param p_dataSet                    Data structure with all the points
    /// \param p_hierarValues                    Hierarchical values
    /// \param p_nodalValues                     Nodal values
    template< class T, class TT>
    void recursive1DDehierarchization(Eigen::ArrayXc &p_levelCurrent,
                                      Eigen::ArrayXui &p_positionCurrent,
                                      const  SparseSet::const_iterator &p_iterLevel,
                                      const unsigned int &p_idim,
                                      const T &p_leftParentHierarValue,
                                      const T &p_rightParentHierarValue,
                                      const T &p_linearHierarParent,
                                      const T &p_linearHierarGrandParent,
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
        // calculation linear hierarchical coefficient
        int iBaseType = iNodeToFunc[p_positionCurrent(p_idim) % 4];
        T LinearHierarchical = valueMidle - (weightParent[iBaseType] * p_linearHierarParent + weightGrandParent[iBaseType] * p_linearHierarGrandParent);
        valueMidle = LinearHierarchical + 0.5 * (p_leftParentHierarValue + p_rightParentHierarValue);
        DoubleOrArray().affect(p_nodalValues, iposPoint, valueMidle);

        char oldLevel = p_levelCurrent(p_idim);
        unsigned int oldPosition = p_positionCurrent(p_idim);
        // child level
        p_levelCurrent(p_idim) += 1;
        SparseSet::const_iterator  iterLevelChild = p_dataSet.find(p_levelCurrent);

        // left
        p_positionCurrent(p_idim) = 2 * oldPosition;
        recursive1DDehierarchization<T, TT>(p_levelCurrent, p_positionCurrent, iterLevelChild, p_idim, p_leftParentHierarValue, valueMidle, LinearHierarchical, p_linearHierarParent,
                                            p_dataSet, p_hierarValues, p_nodalValues);
        // right
        p_positionCurrent(p_idim) += 1;
        recursive1DDehierarchization<T, TT>(p_levelCurrent, p_positionCurrent, iterLevelChild,  p_idim, valueMidle, p_rightParentHierarValue, LinearHierarchical, p_linearHierarParent,
                                            p_dataSet, p_hierarValues, p_nodalValues);

        p_positionCurrent(p_idim) = oldPosition;
        p_levelCurrent(p_idim) = oldLevel;
    }

    /// \brief Dehierarchization 1D in given dimension (modified bound, step 2 quadratic)
    /// \param p_levelCurrent                Current index of the point
    /// \param p_positionCurrent             Current level of the point
    /// \param p_iterLevel                   Iterator on current level
    /// \param p_idim                        Current dimension
    /// \param p_leftParentHierarValue         Left parent nodal value
    /// \param p_rightParentHierarValue        Right parent nodal value
    /// \param p_linearHierarParent            Linear hierarchical value of parent
    /// \param p_dataSet                     Data structure with all the points
    /// \param p_hierarValues                Hierarchical values
    /// \param p_iNodeNum                    Node number
    /// \param p_nodalValues                Nodal values
    template< class T, class TT>
    void dehierarchizationStep3(Eigen::ArrayXc &p_levelCurrent,
                                Eigen::ArrayXui &p_positionCurrent,
                                const  SparseSet::const_iterator &p_iterLevel,
                                const unsigned int &p_idim,
                                const T &p_leftParentHierarValue,
                                const T &p_rightParentHierarValue,
                                const T &p_linearHierarParent,
                                const SparseSet &p_dataSet,
                                const TT &p_hierarValues,
                                const int &p_iNodeNum,
                                TT &p_nodalValues)
    {
        if (p_iterLevel == p_dataSet.end())
            return ;
        // position
        SparseLevel::const_iterator iterPosition = p_iterLevel->second.find(p_positionCurrent);
        if (iterPosition == p_iterLevel->second.end())
            return ;
        int iposPoint = iterPosition->second;
        T valueMidle = DoubleOrArray()(p_hierarValues, iposPoint);

        // do dehierarchization
        // calculation linear hierarchical coefficient
        T LinearHierarchical = valueMidle + 0.25 * p_linearHierarParent;
        valueMidle = LinearHierarchical + 0.5 * (p_leftParentHierarValue + p_rightParentHierarValue);
        DoubleOrArray().affect(p_nodalValues, iposPoint, valueMidle);

        char oldLevel = p_levelCurrent(p_idim);
        unsigned int oldPosition = p_positionCurrent(p_idim);
        // child level
        p_levelCurrent(p_idim) += 1;
        SparseSet::const_iterator  iterLevelChild = p_dataSet.find(p_levelCurrent);

        // modified left and right values
        T leftParentHierarValueLoc = p_leftParentHierarValue;
        T rightParentHierarValueLoc = p_rightParentHierarValue;
        T LinearHierarchicalLoc = LinearHierarchical;
        if (oldPosition == 0)
        {
            // extrapolation use linear
            leftParentHierarValueLoc = 2 * valueMidle - p_rightParentHierarValue;
            DoubleOrArray().zero(LinearHierarchicalLoc, p_nodalValues);
            p_positionCurrent(p_idim) = 2 * oldPosition;
            dehierarchizationStep3<T, TT>(p_levelCurrent, p_positionCurrent,  iterLevelChild, p_idim, leftParentHierarValueLoc, valueMidle, LinearHierarchicalLoc,
                                          p_dataSet, p_hierarValues, 2 * p_iNodeNum, p_nodalValues);
            p_positionCurrent(p_idim) += 1;
            dehierarchizationStep3<T, TT>(p_levelCurrent, p_positionCurrent,  iterLevelChild, p_idim, valueMidle, rightParentHierarValueLoc, LinearHierarchicalLoc,
                                          p_dataSet, p_hierarValues, 2 * p_iNodeNum + 1, p_nodalValues);
        }
        else if (oldPosition == lastNode[oldLevel - 1])
        {
            // linear extrapolation and not cubic
            rightParentHierarValueLoc = 2 * valueMidle - p_leftParentHierarValue;
            DoubleOrArray().zero(LinearHierarchicalLoc, p_nodalValues);
            p_positionCurrent(p_idim) = 2 * oldPosition;
            dehierarchizationStep3<T, TT>(p_levelCurrent, p_positionCurrent,  iterLevelChild, p_idim, leftParentHierarValueLoc, valueMidle, LinearHierarchicalLoc,
                                          p_dataSet, p_hierarValues, 2 * p_iNodeNum, p_nodalValues);
            p_positionCurrent(p_idim) += 1;
            dehierarchizationStep3<T, TT>(p_levelCurrent, p_positionCurrent, iterLevelChild, p_idim, valueMidle, rightParentHierarValueLoc, LinearHierarchicalLoc,
                                          p_dataSet, p_hierarValues, 2 * p_iNodeNum + 1, p_nodalValues);
        }
        else if ((oldPosition == 1) || (oldPosition == lastNode[oldLevel - 1] - 1))
        {
            p_positionCurrent(p_idim) = 2 * oldPosition;
            dehierarchizationStep3<T, TT>(p_levelCurrent, p_positionCurrent,  iterLevelChild, p_idim, leftParentHierarValueLoc, valueMidle, LinearHierarchicalLoc, p_dataSet, p_hierarValues, 2 * p_iNodeNum, p_nodalValues);
            p_positionCurrent(p_idim) += 1;
            dehierarchizationStep3<T, TT>(p_levelCurrent, p_positionCurrent, iterLevelChild, p_idim, valueMidle, rightParentHierarValueLoc, LinearHierarchicalLoc, p_dataSet, p_hierarValues, 2 * p_iNodeNum + 1, p_nodalValues);
        }
        else
        {
            T GrandLinearHierarchicalLoc = p_linearHierarParent;
            p_positionCurrent(p_idim) = 2 * oldPosition;
            recursive1DDehierarchization<T, TT>(p_levelCurrent, p_positionCurrent,  iterLevelChild, p_idim, leftParentHierarValueLoc, valueMidle, LinearHierarchicalLoc,
                                                GrandLinearHierarchicalLoc, p_dataSet, p_hierarValues, p_nodalValues);
            p_positionCurrent(p_idim) += 1;
            recursive1DDehierarchization<T, TT>(p_levelCurrent, p_positionCurrent, iterLevelChild, p_idim, valueMidle, rightParentHierarValueLoc, LinearHierarchicalLoc,
                                                GrandLinearHierarchicalLoc, p_dataSet, p_hierarValues, p_nodalValues);
        }
        p_positionCurrent(p_idim) = oldPosition;
        p_levelCurrent(p_idim) = oldLevel;
    }

    /// \brief Dehierarchization 1D in given dimension (modified bound, step 2 quadratic)
    /// \param p_levelCurrent                 Current level of the point
    /// \param p_positionCurrent              Current position of the point
    /// \param p_iterLevel                          Iteration on level
    /// \param p_idim                             Current dimension
    /// \param p_leftParentHierarValue          Left parent nodal value
    /// \param p_rightParentHierarValue         Right parent nodal value
    /// \param p_dataSet                   Data structure with all the points
    /// \param p_hierarValues     Hierarchical values
    /// \param p_iNodeNum           Node number
    /// \param p_nodalValues      Nodal values
    template< class T, class TT>
    void dehierarchizationStep2(Eigen::ArrayXc &p_levelCurrent,
                                Eigen::ArrayXui &p_positionCurrent,
                                const  SparseSet::const_iterator &p_iterLevel,
                                const unsigned int &p_idim,
                                const T &p_leftParentHierarValue,
                                const T &p_rightParentHierarValue,
                                const SparseSet   &p_dataSet,
                                const TT &p_hierarValues,
                                const int &p_iNodeNum,
                                TT &p_nodalValues)
    {
        if (p_iterLevel == p_dataSet.end())
            return ;
        // position
        SparseLevel::const_iterator iterPosition = p_iterLevel->second.find(p_positionCurrent);
        if (iterPosition == p_iterLevel->second.end())
            return ;
        int iposPoint = iterPosition->second;
        T valueMidle = DoubleOrArray()(p_hierarValues, iposPoint);
        // do dehierarchization
        // calculation linear hierarchical coefficient
        T LinearHierarchical = valueMidle ;
        valueMidle = LinearHierarchical + 0.5 * (p_leftParentHierarValue + p_rightParentHierarValue);
        DoubleOrArray().affect(p_nodalValues, iposPoint, valueMidle);


        char oldLevel = p_levelCurrent(p_idim);
        unsigned int oldPosition = p_positionCurrent(p_idim);
        // child level
        p_levelCurrent(p_idim) += 1;
        SparseSet::const_iterator  iterLevelChild = p_dataSet.find(p_levelCurrent);

        // modified left and right values
        T leftParentHierarValueLoc = p_leftParentHierarValue;
        T rightParentHierarValueLoc = p_rightParentHierarValue;
        T LinearHierarchicalLoc = LinearHierarchical;
        if (oldPosition == 0)
        {
            // extrapolation use linear
            leftParentHierarValueLoc = 2 * valueMidle - p_rightParentHierarValue;
            DoubleOrArray().zero(LinearHierarchicalLoc, p_nodalValues);
            p_positionCurrent(p_idim) = 2 * oldPosition;
            dehierarchizationStep3<T, TT>(p_levelCurrent, p_positionCurrent,  iterLevelChild, p_idim, leftParentHierarValueLoc, valueMidle, LinearHierarchicalLoc,
                                          p_dataSet, p_hierarValues, 2 * p_iNodeNum, p_nodalValues);
            p_positionCurrent(p_idim) += 1;
            dehierarchizationStep3<T, TT>(p_levelCurrent, p_positionCurrent, iterLevelChild,  p_idim, valueMidle, rightParentHierarValueLoc, LinearHierarchicalLoc,
                                          p_dataSet, p_hierarValues, 2 * p_iNodeNum + 1, p_nodalValues);
        }
        else if (oldPosition == lastNode[oldLevel - 1])
        {
            // linear extrapolation and not cubic
            rightParentHierarValueLoc = 2 * valueMidle - p_leftParentHierarValue;
            DoubleOrArray().zero(LinearHierarchicalLoc, p_nodalValues);
            p_positionCurrent(p_idim) = 2 * oldPosition;
            dehierarchizationStep3<T, TT>(p_levelCurrent, p_positionCurrent,  iterLevelChild,  p_idim, leftParentHierarValueLoc, valueMidle, LinearHierarchicalLoc,
                                          p_dataSet, p_hierarValues, 2 * p_iNodeNum, p_nodalValues);
            p_positionCurrent(p_idim) += 1;
            dehierarchizationStep3<T, TT>(p_levelCurrent, p_positionCurrent,  iterLevelChild, p_idim, valueMidle, rightParentHierarValueLoc, LinearHierarchicalLoc,
                                          p_dataSet, p_hierarValues, 2 * p_iNodeNum + 1, p_nodalValues);
        }
        else
        {
            abort();
        }
        p_positionCurrent(p_idim) = oldPosition;
        p_levelCurrent(p_idim) = oldLevel;
    }

    /// \brief Dehierarchization 1D in given dimension (modified bound, step 1 linear)
    /// \param p_levelCurrent                 Current level of the point
    /// \param p_positionCurrent              Current position of the point
    /// \param p_iterLevel                    iterator to locate the current level
    /// \param p_idim                         Current dimension
    /// \param p_dataSet                      Data structure with all the points
    /// \param p_hierarValues                 Hierarchical values
    /// \param p_nodalValues                  Nodal values
    template< class T, class TT>
    void dehierarchizationStep1(Eigen::ArrayXc &p_levelCurrent,
                                Eigen::ArrayXui &p_positionCurrent,
                                const  SparseSet::const_iterator &p_iterLevel,
                                const unsigned int &p_idim,
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
        int iposPoint = iterPosition->second;
        T valueMidle =  DoubleOrArray()(p_hierarValues, iposPoint);
        // do dehierarchization
        DoubleOrArray().affect(p_nodalValues, iposPoint, valueMidle);
        // modified left and right values
        T leftParentHierarValueLoc = valueMidle;
        T rightParentHierarValueLoc = valueMidle;

        char oldLevel = p_levelCurrent(p_idim);
        unsigned int oldPosition = p_positionCurrent(p_idim);
        // child level
        p_levelCurrent(p_idim) += 1;
        SparseSet::const_iterator  iterLevelChild = p_dataSet.find(p_levelCurrent);

        // left
        p_positionCurrent(p_idim) = 2 * oldPosition;
        dehierarchizationStep2<T, TT>(p_levelCurrent, p_positionCurrent, iterLevelChild,  p_idim, leftParentHierarValueLoc, valueMidle, p_dataSet, p_hierarValues, 0, p_nodalValues);
        // right
        p_positionCurrent(p_idim) += 1;
        dehierarchizationStep2<T, TT>(p_levelCurrent, p_positionCurrent,  iterLevelChild, p_idim, valueMidle, rightParentHierarValueLoc,
                                      p_dataSet, p_hierarValues, 1, p_nodalValues);

        p_positionCurrent(p_idim) = oldPosition;
        p_levelCurrent(p_idim) = oldLevel;
    }

public :

    Dehierar1DCubicNoBound() {}

    /// \brief Dehierarchization 1D in given dimension
    /// \param p_levelCurrent                 Current level of the point
    /// \param p_positionCurrent              Current position of the point
    /// \param p_iterLevel                          Iteration on level
    /// \param p_idim                             Current dimension
    /// \param p_dataSet                   Data structure with all the points
    /// \param p_hierarValues     Hierarchical values
    /// \param p_nodalValues      Nodal values
    template< class T, class TT>
    void operator()(Eigen::ArrayXc &p_levelCurrent,
                    Eigen::ArrayXui &p_positionCurrent,
                    const SparseSet::const_iterator &p_iterLevel,
                    const unsigned int &p_idim,
                    const SparseSet &p_dataSet,
                    const TT &p_hierarValues,
                    TT &p_nodalValues)
    {
        dehierarchizationStep1<T, TT>(p_levelCurrent, p_positionCurrent, p_iterLevel, p_idim,  p_dataSet, p_hierarValues, p_nodalValues);
    }
};
///@}


}
#endif /* SPARSEGRIDCUBICNONOUND_H */

