// Copyright (C) 2016 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#ifndef GRIDREACH_H
#define GRIDREACH_H
#include "libflow/core/utils/types.h"
#include "libflow/core/grids/FullGrid.h"

/** \file GridReach.h
 * \brief From a subgrid of a grid, get the subgrid reached by an optimizer
 */

namespace libflow
{
/// \class GridReach GridReach.h
/// From a subgrid of a grid, get the subgrid reached by an optimizer
template< class OptimizerBase>
class GridReach
{
private :
    std::shared_ptr<FullGrid> m_pGridCur; ///< current grid at current time step
    std::shared_ptr<FullGrid> m_pGridPrev; ///< grid at previously treated time step
    std::shared_ptr< OptimizerBase >  m_pOptimize;
public :
    /// \brief  constructor
    /// \param  p_pGridCur a full grid at current time step
    /// \param  p_pGridPrev a full grid at previous time step
    /// \param  p_pOptimize  this optimizer used in dynamic programing
    GridReach(const   std::shared_ptr<FullGrid> &p_pGridCur,
              const   std::shared_ptr<FullGrid> &p_pGridPrev,
              const   std::shared_ptr<OptimizerBase >   &p_pOptimize):  m_pGridCur(p_pGridCur), m_pGridPrev(p_pGridPrev), m_pOptimize(p_pOptimize)
    {
    }

    /// \brief operator ()
    /// \param p_subMesh subgrid treated by the processor at the current date
    /// \return the subgrid needed at the next step
    SubMeshIntCoord   operator()(const SubMeshIntCoord &p_subMesh)
    {
        // submesh to values at current grid
        Eigen::ArrayXi iCoordMin(m_pGridCur->getDimension()),  iCoordMax(m_pGridCur->getDimension());
        for (int id = 0; id < m_pGridCur->getDimension(); ++id)
        {
            iCoordMin(id) = p_subMesh(id)[0];
            iCoordMax(id) = p_subMesh(id)[1] - 1; // because  p_subMesh(id)[1] corresponds to the first point outside the grid
        }
        Eigen::ArrayXd xCoordMin = m_pGridCur->getCoordinateFromIntCoord(iCoordMin);
        Eigen::ArrayXd xCoordMax = m_pGridCur->getCoordinateFromIntCoord(iCoordMax);
        std::vector<  std::array< double, 2>  >  regionByProcessor(m_pGridCur->getDimension());
        for (int id = 0; id < m_pGridCur->getDimension(); ++id)
        {
            regionByProcessor[id][0] = xCoordMin(id);
            regionByProcessor[id][1] = xCoordMax(id);
        }
        std::vector<  std::array< double, 2>  > cone = m_pOptimize->getCone(regionByProcessor);
        SubMeshIntCoord retGrid(m_pGridPrev->getDimension());
        std::vector <std::array< double, 2>  > extremVal = m_pGridPrev->getExtremeValues();
        Eigen::ArrayXd xCapMin(m_pGridPrev->getDimension()), xCapMax(m_pGridPrev->getDimension());
        for (int id = 0; id < m_pGridPrev->getDimension(); ++id)
        {
            xCapMin(id)   = std::min(std::max(cone[id][0], extremVal[id][0]), extremVal[id][1]);
            xCapMax(id)  = std::max(std::min(cone[id][1], extremVal[id][1]), extremVal[id][0]);
        }
        Eigen::ArrayXi  iCapMin = m_pGridPrev->lowerPositionCoord(xCapMin);
        Eigen::ArrayXi  iCapMax = m_pGridPrev->upperPositionCoord(xCapMax) + 1; // last is excluded
        for (int id = 0; id < m_pGridPrev->getDimension(); ++id)
        {
            retGrid(id)[0] = iCapMin(id);
            retGrid(id)[1] = iCapMax(id);
        }
        return retGrid;
    }
}
;
}
#endif
