
#ifndef SPARSEGRIDQUADNOBOUND_H
#define SPARSEGRIDQUADNOBOUND_H
#include <Eigen/Dense>
#include "libflow/core/sparse/sparseGridTypes.h"
#include "libflow/core/sparse/SparseGridHierarDehierarNoBound.h"


/** \file SparseGridQuadNoBound.h
 * \brief Regroup hierarchization and dehierarchization for quadratic sparse grids
 * \author Xavier Warin
 */
namespace libflow
{
/// \defgroup qudSparseNoBound  Quadratic  Hierarchization and Deheriarchization without  boundary points
/// \brief Regroup function used in hierarchization and dehierarchization for sparse grids without any points on the boundary and a quadratic approximation per mesh
///@{

/// \class  Hierar1DQuadNoBound   SparseGridQuadNoBound.h
/// Hierarchization
class Hierar1DQuadNoBound : public HierarDehierarNoBound
{

protected :

    /// \brief Hierarchization in given dimension in 1D
    /// \param p_levelCurrent                 Current level of the point
    /// \param p_positionCurrent              Current position of the point
    /// \param p_iterLevel                    Iteration on level
    /// \param p_idim                         Current dimension
    /// \param p_leftParentNodalValue         Left parent nodal value
    /// \param p_rightParentNodalValue        Left parent nodal value
    /// \param p_parentLinearHierarValue      Linear hierarchical value of parent
    /// \param p_dataSet                      Data structure with all the points
    /// \param p_nodalValues                  Nodal values
    /// \param p_hierarValues                 Hierarchized values
    template< class T, class TT>
    void recursive1DHierarchization(Eigen::ArrayXc   &p_levelCurrent,
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
        // position
        SparseLevel::const_iterator iterPosition = p_iterLevel->second.find(p_positionCurrent);
        if (iterPosition == p_iterLevel->second.end())
            return ;
        int iposPoint = iterPosition->second;
        T valueMidle = DoubleOrArray()(p_nodalValues, iposPoint);
        // if not  top
        // hierarchization
        T linearHierar = valueMidle  - 0.5 * (p_leftParentNodalValue + p_rightParentNodalValue);
        // hierarchization
        DoubleOrArray().affect(p_hierarValues, iposPoint, linearHierar - 0.25 * p_parentLinearHierarValue);

        char oldLevel = p_levelCurrent(p_idim);
        unsigned int oldPosition = p_positionCurrent(p_idim);
        // child level
        p_levelCurrent(p_idim) += 1;
        SparseSet::const_iterator  iterLevelChild = p_dataSet.find(p_levelCurrent);

        // modified left and right values
        T leftParentNodalValueLoc = p_leftParentNodalValue;
        T rightParentNodalValueLoc = p_rightParentNodalValue;
        T parentLinearHierarValueLoc = linearHierar;
        if (oldLevel == 1)
        {
            leftParentNodalValueLoc = valueMidle;
            rightParentNodalValueLoc = valueMidle;
            DoubleOrArray().zero(parentLinearHierarValueLoc, p_nodalValues);
        }
        else  if (oldPosition == 0)
        {
            leftParentNodalValueLoc = 2 * valueMidle - p_rightParentNodalValue;
            DoubleOrArray().zero(parentLinearHierarValueLoc, p_nodalValues);
        }
        else if (oldPosition == lastNode[oldLevel - 1])
        {
            rightParentNodalValueLoc = 2 * valueMidle - p_leftParentNodalValue;
            DoubleOrArray().zero(parentLinearHierarValueLoc, p_nodalValues);
        }
        // left
        p_positionCurrent(p_idim) = 2 * oldPosition;
        recursive1DHierarchization<T, TT>(p_levelCurrent, p_positionCurrent,  iterLevelChild, p_idim, leftParentNodalValueLoc, valueMidle,  parentLinearHierarValueLoc, p_dataSet, p_nodalValues, p_hierarValues);
        // right
        p_positionCurrent(p_idim) += 1;
        recursive1DHierarchization<T, TT>(p_levelCurrent, p_positionCurrent,   iterLevelChild, p_idim, valueMidle, rightParentNodalValueLoc,  parentLinearHierarValueLoc, p_dataSet, p_nodalValues, p_hierarValues);

        p_positionCurrent(p_idim) = oldPosition;
        p_levelCurrent(p_idim) = oldLevel;
    }

public :

    /// \brief Default constructor
    Hierar1DQuadNoBound() {}

    /// \brief Hierarchization in given dimension in 1D
    /// \param p_levelCurrent             Current level of the point
    /// \param p_positionCurrent          Current position of the point
    /// \param p_iterLevel                Iterator on current level
    /// \param p_idim                     Current dimension
    /// \param p_dataSet                  Data structure with all the points
    /// \param p_nodalValues              Nodal values
    /// \param p_hierarValues             Hierarchical values
    template< class T, class TT>
    void operator()(Eigen::ArrayXc   &p_levelCurrent,
                    Eigen::ArrayXui &p_positionCurrent,
                    const  SparseSet::const_iterator &p_iterLevel,
                    const unsigned int &p_idim,
                    const SparseSet   &p_dataSet,
                    const TT &p_nodalValues,
                    TT &p_hierarValues)
    {
        // left and right value
        T leftValue ;
        DoubleOrArray().zero(leftValue, p_nodalValues);
        T rightValue ;
        DoubleOrArray().zero(rightValue, p_nodalValues);
        T parentValue  ;
        DoubleOrArray().zero(parentValue, p_nodalValues);

        recursive1DHierarchization<T, TT>(p_levelCurrent, p_positionCurrent, p_iterLevel, p_idim, leftValue, rightValue, parentValue, p_dataSet, p_nodalValues, p_hierarValues);
    }
};


/// \class  Dehierar1DQuadNoBound   SparseGridQuadNoBound.h
/// Dehierarchization
class Dehierar1DQuadNoBound: public HierarDehierarNoBound
{

protected :

    /// \brief Dehierarchization 1D in given dimension (modified bound)
    /// \param p_levelCurrent             Current level of the point
    /// \param p_positionCurrent          Current  position of the point
    /// \param p_iterLevel                Iterator on current level
    /// \param p_idim                     Current dimension
    /// \param p_leftParentHierarValue    Left parent nodal value
    /// \param p_rightParentHierarValue   Right parent nodal value
    /// \param p_linearHierarParent         Linear hierarchical value of parent
    /// \param p_dataSet                  Data structure with all the points
    /// \param p_hierarValues             Hierarchical values
    /// \param p_nodalValues              Nodal values
    template< class T, class TT>
    void recursive1DDehierarchization(Eigen::ArrayXc   &p_levelCurrent,
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
        T linearHierarchical = valueMidle + 0.25 * p_linearHierarParent;
        valueMidle = linearHierarchical + 0.5 * (p_leftParentHierarValue + p_rightParentHierarValue);
        DoubleOrArray().affect(p_nodalValues, iposPoint, valueMidle);

        char oldLevel = p_levelCurrent(p_idim);
        unsigned int oldPosition = p_positionCurrent(p_idim);
        // child level
        p_levelCurrent(p_idim) += 1;
        SparseSet::const_iterator  iterLevelChild = p_dataSet.find(p_levelCurrent);

        // modified left and right values
        T leftParentHierarValueLoc = p_leftParentHierarValue;
        T rightParentHierarValueLoc = p_rightParentHierarValue;
        T linearHierarchicalLoc = linearHierarchical;
        if (oldLevel == 1)
        {
            leftParentHierarValueLoc = valueMidle;
            rightParentHierarValueLoc = valueMidle;
            DoubleOrArray().zero(linearHierarchicalLoc, p_nodalValues) ;
        }
        else  if (oldPosition == 0)
        {
            // extrapolation use linear
            leftParentHierarValueLoc = 2 * valueMidle - p_rightParentHierarValue;
            DoubleOrArray().zero(linearHierarchicalLoc, p_nodalValues) ;
        }
        else if (oldPosition == lastNode[oldLevel - 1])
        {
            // linear extrapolation and not quadratic
            rightParentHierarValueLoc = 2 * valueMidle - p_leftParentHierarValue;
            DoubleOrArray().zero(linearHierarchicalLoc, p_nodalValues) ;
        }
        // left
        p_positionCurrent(p_idim) = 2 * oldPosition;
        recursive1DDehierarchization<T, TT>(p_levelCurrent, p_positionCurrent, iterLevelChild, p_idim, leftParentHierarValueLoc, valueMidle, linearHierarchicalLoc, p_dataSet, p_hierarValues, p_nodalValues);
        // right
        p_positionCurrent(p_idim) += 1;
        recursive1DDehierarchization<T, TT>(p_levelCurrent, p_positionCurrent, iterLevelChild, p_idim, valueMidle, rightParentHierarValueLoc,  linearHierarchicalLoc, p_dataSet, p_hierarValues, p_nodalValues);

        p_positionCurrent(p_idim) = oldPosition;
        p_levelCurrent(p_idim) = oldLevel;
    }

public :

    /// \brief default constructor
    Dehierar1DQuadNoBound() {}

    /// \brief Dehierarchization 1D in given dimension
    /// \param  p_levelCurrent             Current  level of the point
    /// \param p_positionCurrent           Current  position of the point
    /// \param p_iterLevel                 Iterator on current level
    /// \param p_idim                      Current dimension
    /// \param p_dataSet                   Data structure with all the points
    /// \param p_hierarValues              Hierarchical values
    /// \param p_nodalValues               Nodal values
    template< class T, class TT>
    void operator()(Eigen::ArrayXc   &p_levelCurrent,
                    Eigen::ArrayXui &p_positionCurrent,
                    const  SparseSet::const_iterator &p_iterLevel,
                    const unsigned int &p_idim,
                    const SparseSet &p_dataSet,
                    const TT &p_hierarValues,
                    TT &p_nodalValues)
    {
        T leftValue ;
        DoubleOrArray().zero(leftValue, p_nodalValues);
        T rightValue ;
        DoubleOrArray().zero(rightValue, p_nodalValues);
        T parentValue  ;
        DoubleOrArray().zero(parentValue, p_nodalValues);

        recursive1DDehierarchization<T, TT>(p_levelCurrent, p_positionCurrent, p_iterLevel,  p_idim, leftValue, rightValue, parentValue, p_dataSet, p_hierarValues, p_nodalValues);
    }

};
///@}

}
#endif /* SPARSEGRIDQUADNONOUND.H */
