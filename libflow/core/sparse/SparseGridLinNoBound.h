
#ifndef SPARSEGRIDLINNOBOUND_H
#define SPARSEGRIDLINNOBOUND_H
#include <Eigen/Dense>
#include "libflow/core/sparse/sparseGridTypes.h"
#include "libflow/core/sparse/SparseGridHierarDehierarNoBound.h"


/** \file SparseGridLinNoBound.h
 * \brief Regroup linearSparseNoBound hierarchization and dehierarchization for linear sparse grids
 * \author Xavier Warin
 */
namespace libflow
{
/// \defgroup linearSparseNoBound Linear Hierarchization and Deheriarchization without  boundary points
/// Regroup function used in hierarchization ond dehierarchizatoin for sparse grids without any points on the boundary and a linear approximation per mesh
///@{



/// \class  Hierar1DLinNoBound   SparseGridLinNoBound.h
/// Hierarchization
class Hierar1DLinNoBound : public HierarDehierarNoBound
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
    /// \param p_nodalValues               Nodalvalues
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
        int iposPoint = iterPosition->second;
        T valueMidle =  DoubleOrArray()(p_nodalValues, iposPoint);
        // hierarchization
        DoubleOrArray().affect(p_hierarValues, iposPoint, valueMidle  - 0.5 * (p_leftParentNodalValue + p_rightParentNodalValue));

        char oldLevel = p_levelCurrent(p_idim);
        unsigned int oldPosition = p_positionCurrent(p_idim);
        // child level
        p_levelCurrent(p_idim) += 1;
        SparseSet::const_iterator  iterLevelChild = p_dataSet.find(p_levelCurrent);

        // modified left and right values
        T leftParentNodalValueLoc = p_leftParentNodalValue;
        T rightParentNodalValueLoc = p_rightParentNodalValue;
        if (oldLevel == 1)
        {
            leftParentNodalValueLoc = valueMidle;
            rightParentNodalValueLoc = valueMidle;
        }
        else if (oldPosition == 0)
        {
            // linear extrapolation
            leftParentNodalValueLoc = 2 * valueMidle - p_rightParentNodalValue;
        }
        else if (oldPosition == lastNode[oldLevel - 1])
        {
            // linear extrapolation
            rightParentNodalValueLoc = 2 * valueMidle - p_leftParentNodalValue;
        }
        // left
        p_positionCurrent(p_idim) = 2 * oldPosition;
        recursive1DHierarchization<T, TT>(p_levelCurrent, p_positionCurrent,  iterLevelChild, p_idim, leftParentNodalValueLoc, valueMidle, p_dataSet, p_nodalValues, p_hierarValues);
        // right
        p_positionCurrent(p_idim) += 1;
        recursive1DHierarchization<T, TT>(p_levelCurrent, p_positionCurrent,  iterLevelChild, p_idim, valueMidle, rightParentNodalValueLoc, p_dataSet, p_nodalValues, p_hierarValues);

        p_positionCurrent(p_idim) = oldPosition;
        p_levelCurrent(p_idim) = oldLevel;
    }

public :

    // default
    Hierar1DLinNoBound() {}

    // operator for 1D Hierarchization
    /// \brief Hierarchization in given dimension in 1D
    /// \param p_levelCurrent            Current level of the point
    /// \param p_positionCurrent         Current position  of the point in the current level
    /// \param p_iterLevel               Iterator on current level
    /// \param p_idim                    Current dimension
    /// \param p_dataSet                 Data structure with all the points
    /// \param p_nodalValues             Nodalvalues
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
        T leftValue ;
        DoubleOrArray().zero(leftValue, p_nodalValues);
        T rightValue ;
        DoubleOrArray().zero(rightValue, p_nodalValues);
        recursive1DHierarchization<T, TT>(p_levelCurrent, p_positionCurrent, p_iterLevel, p_idim, leftValue, rightValue, p_dataSet, p_nodalValues, p_hierarValues);

    }

};


/// \class Dehierar1DLinNoBound   SparseGridLinNoBound.h
/// Dehierarchization
class Dehierar1DLinNoBound : public HierarDehierarNoBound
{
protected :
    /// \brief Dehierarchization 1D in given dimension
    /// \param p_levelCurrent            Current level of the point
    /// \param p_positionCurrent         Current position of the point in the current level
    /// \param p_iterLevel               Iterator on current level
    /// \param p_idim                    Current dimension
    /// \param p_leftParentHierarValue   Left parent nodal value
    /// \param rightParentHierarValue    Left parent nodal value
    /// \param p_dataSet                 Data structure with all the points
    /// \param p_hierarValues            Hierarchical values
    /// \param p_nodalValues             Nodalvalues
    template< class T, class TT>
    void recursive1DDehierarchization(Eigen::ArrayXc &p_levelCurrent,
                                      Eigen::ArrayXui &p_positionCurrent,
                                      const  SparseSet::const_iterator &p_iterLevel,
                                      const unsigned int &p_idim,
                                      const T &p_leftParentHierarValue,
                                      const T &rightParentHierarValue,
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
        T valueMidle = DoubleOrArray()(p_nodalValues, iposPoint);
        // do dehierarchization
        valueMidle += 0.5 * (p_leftParentHierarValue + rightParentHierarValue);
        DoubleOrArray().affect(p_nodalValues, iposPoint, valueMidle);

        char oldLevel = p_levelCurrent(p_idim);
        unsigned int oldPosition = p_positionCurrent(p_idim);
        // child level
        p_levelCurrent(p_idim) += 1;
        SparseSet::const_iterator  iterLevelChild = p_dataSet.find(p_levelCurrent);

        // modified left and right values
        T leftParentHierarValueLoc = p_leftParentHierarValue;
        T rightParentHierarValueLoc = rightParentHierarValue;
        if (oldLevel == 1)
        {
            leftParentHierarValueLoc = valueMidle;
            rightParentHierarValueLoc = valueMidle;
        }
        else  if (oldPosition == 0)
        {
            // linear extrapolation
            leftParentHierarValueLoc = 2 * valueMidle - rightParentHierarValue;
        }
        else if (oldPosition == lastNode[oldLevel - 1])
        {
            // linear extrapolation
            rightParentHierarValueLoc = 2 * valueMidle - p_leftParentHierarValue;
        }
        // left
        p_positionCurrent(p_idim) = 2 * oldPosition;
        recursive1DDehierarchization<T, TT>(p_levelCurrent, p_positionCurrent,  iterLevelChild, p_idim, leftParentHierarValueLoc, valueMidle, p_dataSet, p_hierarValues, p_nodalValues);
        // right
        p_positionCurrent(p_idim) += 1;
        recursive1DDehierarchization<T, TT>(p_levelCurrent, p_positionCurrent, iterLevelChild, p_idim, valueMidle, rightParentHierarValueLoc, p_dataSet, p_hierarValues, p_nodalValues);

        p_positionCurrent(p_idim) = oldPosition;
        p_levelCurrent(p_idim) = oldLevel;
    }

public :

    Dehierar1DLinNoBound() {}

    /// \brief Dehierarchization 1D in given dimension
    /// \param p_levelCurrent             Current level of the point
    /// \param p_positionCurrent          Current position of the point
    /// \param p_iterLevel                Iterator on current level
    /// \param p_idim                     Current dimension
    /// \param p_dataSet                  Data structure with all the points
    /// \param p_hierarValues             Hierarchical values
    /// \param p_nodalValues              Nodalvalues
    template< class T, class TT>
    void operator()(Eigen::ArrayXc &p_levelCurrent,
                    Eigen::ArrayXui &p_positionCurrent,
                    const SparseSet::const_iterator &p_iterLevel,
                    const unsigned int &p_idim,
                    const SparseSet &p_dataSet,
                    const TT &p_hierarValues,
                    TT &p_nodalValues)
    {
        T leftValue ;
        DoubleOrArray().zero(leftValue, p_nodalValues);
        T rightValue ;
        DoubleOrArray().zero(rightValue, p_nodalValues);
        recursive1DDehierarchization<T, TT>(p_levelCurrent, p_positionCurrent, p_iterLevel, p_idim, leftValue, rightValue, p_dataSet, p_hierarValues, p_nodalValues);
    }

};

///@}

}
#endif
