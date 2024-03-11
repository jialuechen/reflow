
#ifndef SPARSEGRIDCUBICBOUND_H
#define SPARSEGRIDCUBICBOUND_H
#include <Eigen/Dense>
#include "libflow/core/sparse/sparseGridTypes.h"
#include "libflow/core/sparse/SparseGridHierarDehierarBound.h"


namespace libflow
{
/// \defgroup Cubic Hierarchization and Deheriarchization with boundary points
/// \brief Regroup function used in hierarchization and dehierarchization for sparse grids without any points on the boundary and a cubic approximation per mesh
///@{

/// \class  Hierar1DCubicBound   SparseGridCubicBound.h
/// Hierarchization
class Hierar1DCubicBound: public HierarDehierarBound
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
        recursive1DHierarchization<T, TT>(p_levelCurrent, p_positionCurrent, iterLevelChild,  p_idim, p_leftParentNodalValue, valueMidle, LinearHierar, p_parentLinearHierarValue, p_dataSet, p_nodalValues,
                                          p_hierarValues);
        // right
        p_positionCurrent(p_idim) += 1;
        recursive1DHierarchization<T, TT>(p_levelCurrent, p_positionCurrent, iterLevelChild, p_idim,  valueMidle, p_rightParentNodalValue, LinearHierar, p_parentLinearHierarValue, p_dataSet, p_nodalValues,
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
    /// \param p_hierarValues                  Hierarchical values
    template< class T, class TT>
    void hierarchizationStep2(Eigen::ArrayXc &p_levelCurrent,
                              Eigen::ArrayXui &p_positionCurrent,
                              const  SparseSet::const_iterator &p_iterLevel,
                              const unsigned int &p_idim,
                              const T &p_leftParentNodalValue,
                              const T &p_rightParentNodalValue,
                              const T &p_parentLinearHierarValue,
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
        // quadratic
        DoubleOrArray().affect(p_hierarValues, iposPoint, LinearHierar - 0.25 * p_parentLinearHierarValue) ;

        char oldLevel = p_levelCurrent(p_idim);
        unsigned int oldPosition = p_positionCurrent(p_idim);
        // child level
        p_levelCurrent(p_idim) += 1;
        SparseSet::const_iterator  iterLevelChild = p_dataSet.find(p_levelCurrent);

        // left
        p_positionCurrent(p_idim) = 2 * oldPosition;
        recursive1DHierarchization<T, TT>(p_levelCurrent, p_positionCurrent,  iterLevelChild,  p_idim, p_leftParentNodalValue, valueMidle, LinearHierar, p_parentLinearHierarValue,
                                          p_dataSet, p_nodalValues,  p_hierarValues);
        // right
        p_positionCurrent(p_idim) += 1;
        recursive1DHierarchization<T, TT>(p_levelCurrent, p_positionCurrent,  iterLevelChild, p_idim, valueMidle, p_rightParentNodalValue, LinearHierar, p_parentLinearHierarValue,
                                          p_dataSet, p_nodalValues,  p_hierarValues);

        p_positionCurrent(p_idim) = oldPosition;
        p_levelCurrent(p_idim) = oldLevel;
    }


    /// \brief Hierarchization in given dimension in 1D (first step  : subtract constant value)
    /// \param p_levelCurrent                 Current level of the point
    /// \param p_positionCurrent              Current position of the point
    /// \param p_iterLevel                    Iteration on level
    /// \param p_idim                         Current dimension
    /// \param p_leftParentNodalValue         Left parent nodal value
    /// \param p_rightParentNodalValue        Right parent nodal value
    /// \param p_dataSet                      Data structure with all the points
    /// \param p_nodalValues                  Nodal values
    /// \param p_hierarValues                 Hierarchical values
    template< class T, class TT>
    void hierarchizationStep1(Eigen::ArrayXc &p_levelCurrent,
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
        SparseLevel::const_iterator iterPosition = p_iterLevel->second.find(p_positionCurrent);
        if (iterPosition == p_iterLevel->second.end())
            return ;
        // position
        int iposPoint = iterPosition->second;
        T valueMidle = DoubleOrArray()(p_nodalValues, iposPoint);
        // hierarchization
        T LinearHierar = valueMidle  - 0.5 * (p_leftParentNodalValue + p_rightParentNodalValue);
        // cubic
        DoubleOrArray().affect(p_hierarValues, iposPoint, LinearHierar) ;

        char oldLevel = p_levelCurrent(p_idim);
        unsigned int oldPosition = p_positionCurrent(p_idim);
        // child level
        p_levelCurrent(p_idim) += 1;
        SparseSet::const_iterator  iterLevelChild = p_dataSet.find(p_levelCurrent);

        // left
        p_positionCurrent(p_idim) = 0;
        hierarchizationStep2<T, TT>(p_levelCurrent, p_positionCurrent,  iterLevelChild, p_idim, p_leftParentNodalValue, valueMidle, LinearHierar, p_dataSet, p_nodalValues,  p_hierarValues);
        // right
        p_positionCurrent(p_idim) = 1;
        hierarchizationStep2<T, TT>(p_levelCurrent, p_positionCurrent,  iterLevelChild,  p_idim, valueMidle, p_rightParentNodalValue, LinearHierar, p_dataSet, p_nodalValues, p_hierarValues);

        p_positionCurrent(p_idim) = oldPosition;
        p_levelCurrent(p_idim) = oldLevel;

    }

public :

    // default
    Hierar1DCubicBound() {}

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
        // get back boundary key in the dimension
        T leftValue ;
        T rightValue ;
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

        hierarchizationStep1<T, TT>(p_levelCurrent, p_positionCurrent, p_iterLevel, p_idim, leftValue, rightValue,
                                    p_dataSet, p_nodalValues, p_hierarValues);

    }
};


/// \class Dehierar1DCubicBound SparseGridCubicBound.h
///  Dehierarchization for cubic with boundary points
class Dehierar1DCubicBound: public HierarDehierarBound
{
protected :

    /// \brief Dehierarchization 1D in given dimension (modified bound) (cubic so available after two level)
    /// \param p_levelCurrent                  Current index of the point
    /// \param p_positionCurrent               Current level of the point
    /// \param p_iterLevel                     Iterator on current level
    /// \param p_idim                           Current dimension
    /// \param p_leftParentHierarValue          Left parent nodal value
    /// \param p_rightParentHierarValue         Right parent nodal value
    /// \param p_linearHierarParent             Linear hierarchical value of parent
    /// \param p_linearHierarGrandParent        Linear Hierarchical value of parent
    /// \param p_dataSet                         Data structure with all the points
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
    /// \param p_nodalValues                Nodal values
    template< class T, class TT>
    void dehierarchizationStep2(Eigen::ArrayXc &p_levelCurrent,
                                Eigen::ArrayXui &p_positionCurrent,
                                const  SparseSet::const_iterator &p_iterLevel,
                                const unsigned int &p_idim,
                                const T &p_leftParentHierarValue,
                                const T &p_rightParentHierarValue,
                                const T &p_linearHierarParent,
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
        T LinearHierarchical = valueMidle + 0.25 * p_linearHierarParent;
        valueMidle = LinearHierarchical + 0.5 * (p_leftParentHierarValue + p_rightParentHierarValue);
        DoubleOrArray().affect(p_nodalValues, iposPoint, valueMidle);

        char oldLevel = p_levelCurrent(p_idim);
        unsigned int oldPosition = p_positionCurrent(p_idim);
        // child level
        p_levelCurrent(p_idim) += 1;
        SparseSet::const_iterator  iterLevelChild = p_dataSet.find(p_levelCurrent);
        // left
        p_positionCurrent(p_idim) = 2 * oldPosition;
        recursive1DDehierarchization<T, TT>(p_levelCurrent, p_positionCurrent,  iterLevelChild, p_idim, p_leftParentHierarValue, valueMidle, LinearHierarchical, p_linearHierarParent,
                                            p_dataSet, p_hierarValues, p_nodalValues);
        // right
        p_positionCurrent(p_idim) += 1;
        recursive1DDehierarchization<T, TT>(p_levelCurrent, p_positionCurrent, iterLevelChild, p_idim, valueMidle, p_rightParentHierarValue, LinearHierarchical, p_linearHierarParent,
                                            p_dataSet, p_hierarValues, p_nodalValues);

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
    /// \param p_nodalValues      Nodal values
    template< class T, class TT>
    void dehierarchizationStep1(Eigen::ArrayXc &p_levelCurrent,
                                Eigen::ArrayXui &p_positionCurrent,
                                const  SparseSet::const_iterator &p_iterLevel,
                                const unsigned int &p_idim,
                                const T &p_leftParentHierarValue,
                                const T &p_rightParentHierarValue,
                                const SparseSet   &p_dataSet,
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
        T LinearHierarchical =  DoubleOrArray()(p_hierarValues, iposPoint);

        // do dehierarchization
        // calculation linear hierarchical coefficient
        T valueMidle = LinearHierarchical + 0.5 * (p_leftParentHierarValue + p_rightParentHierarValue);
        DoubleOrArray().affect(p_nodalValues, iposPoint, valueMidle);

        char oldLevel = p_levelCurrent(p_idim);
        unsigned int oldPosition = p_positionCurrent(p_idim);

        // child level
        p_levelCurrent(p_idim) += 1;
        SparseSet::const_iterator  iterLevelChild = p_dataSet.find(p_levelCurrent);

        // left
        p_positionCurrent(p_idim) = 0;
        dehierarchizationStep2<T, TT>(p_levelCurrent, p_positionCurrent, iterLevelChild, p_idim, p_leftParentHierarValue, valueMidle, LinearHierarchical,
                                      p_dataSet, p_hierarValues, p_nodalValues);
        // right
        p_positionCurrent(p_idim) = 1;
        dehierarchizationStep2<T, TT>(p_levelCurrent, p_positionCurrent, iterLevelChild, p_idim, valueMidle, p_rightParentHierarValue, LinearHierarchical,
                                      p_dataSet, p_hierarValues, p_nodalValues);

        p_positionCurrent(p_idim) = oldPosition;
        p_levelCurrent(p_idim) = oldLevel;
    }


public :

    Dehierar1DCubicBound() {}

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
                    const  SparseSet::const_iterator &p_iterLevel,
                    const unsigned int &p_idim,
                    const SparseSet &p_dataSet,
                    const TT &p_hierarValues,
                    TT &p_nodalValues)
    {
        // get back boundary key in the dimension
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
            rightValue = DoubleOrArray()(p_hierarValues, iterRight->second);   // get back boundary key in the dimension

        dehierarchizationStep1<T, TT>(p_levelCurrent, p_positionCurrent, p_iterLevel, p_idim, leftValue, rightValue, p_dataSet, p_hierarValues, p_nodalValues);
    }
};
///@}
}
#endif /* SPARSEGRIDCUBICNONOUND_H */

