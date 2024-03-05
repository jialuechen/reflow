
#include "libflow/tree/ContinuationCutsTree.h"
#include "libflow/core/utils/comparisonUtils.h"


using namespace std;
using namespace Eigen;

namespace libflow
{
ContinuationCutsTree::ContinuationCutsTree(const  std::shared_ptr< SpaceGrid >   &p_grid,
        const std::shared_ptr< Tree >   &p_condExp,
        const ArrayXXd &p_values) : m_grid(p_grid),  m_cutCoeff(p_grid->getDimension() + 1)

{
    // nest on cuts
    for (int ic = 0; ic < p_grid->getDimension() + 1; ++ic)
    {
        ArrayXXd  valLoc = p_values.block(ic * p_condExp->getNbNodesNextDate(), 0, p_condExp->getNbNodesNextDate(), p_values.cols());
        m_cutCoeff[ic] = p_condExp->expCondMultiple(valLoc.transpose()).transpose();
    }
    // for first coefficient cuts calculate  \f$ \bar a_0  = a_0 - \sum_{i=1}^d a_i \bar x_i \f$
    // iterator
    shared_ptr<GridIterator> iterGrid = m_grid->getGridIterator();
    while (iterGrid->isValid())
    {
        // coordinates
        ArrayXd pointCoord = iterGrid->getCoordinate();
        // point number
        int ipoint =  iterGrid->getCount();
        // grid cuts
        for (int id = 0 ; id < pointCoord.size(); ++id)
            m_cutCoeff[0].col(ipoint) -= m_cutCoeff[id + 1].col(ipoint) * pointCoord(id);
        iterGrid->next();
    }
}


ArrayXXd ContinuationCutsTree::getCutsANode(const ArrayXXd &p_hypStock, const int &p_node) const
{
    int nbCutsCoeff = m_grid->getDimension() + 1;
    // for return
    ArrayXXd  cuts(nbCutsCoeff, m_grid->getNbPoints());
    int iPointCut = 0;
    // nest on grid points
    shared_ptr<GridIterator> iterRegGrid = m_grid->getGridIterator();
    while (iterRegGrid->isValid())
    {
        // coordinates
        Eigen::ArrayXd pointCoordReg = iterRegGrid->getCoordinate();
        // test if inside the hypercube
        bool bInside = true;
        for (int id = 0 ; id < pointCoordReg.size(); ++id)
            if (isStrictlyLesser(pointCoordReg(id), p_hypStock(id, 0)) || (isStrictlyMore(pointCoordReg(id), p_hypStock(id, 1))))
            {
                bInside = false;
                break;
            }
        if (bInside)
        {
            // point number
            int ipoint =  iterRegGrid->getCount();

            for (int jc = 0; jc < nbCutsCoeff; ++jc)
            {
                // reconstruct the value for all simulations
                cuts(jc, iPointCut) =  m_cutCoeff[jc](p_node, ipoint);
            }
            iPointCut += 1;
        }
        iterRegGrid->next();
    }
    cuts.conservativeResize(nbCutsCoeff, iPointCut);
    return cuts;
}


ArrayXXd  ContinuationCutsTree::getCutsAllNodes(const ArrayXXd &p_hypStock) const
{
    int nbCutsCoeff = m_grid->getDimension() + 1;
    int nbNodes =  m_cutCoeff[0].rows();
    // for return
    ArrayXXd  cuts(nbCutsCoeff * nbNodes, m_grid->getNbPoints());
    int iPointCut = 0;
    // nest on grid points
    shared_ptr<GridIterator> iterRegGrid = m_grid->getGridIterator();
    while (iterRegGrid->isValid())
    {
        // coordinates
        Eigen::ArrayXd pointCoordReg = iterRegGrid->getCoordinate();
        // test if inside the hypercube
        bool bInside = true;
        for (int id = 0 ; id < pointCoordReg.size(); ++id)
            if (isStrictlyLesser(pointCoordReg(id), p_hypStock(id, 0)) || (isStrictlyMore(pointCoordReg(id), p_hypStock(id, 1))))
            {
                bInside = false;
                break;
            }
        if (bInside)
        {
            // point number
            int ipoint =  iterRegGrid->getCount();

            for (int jc = 0; jc < nbCutsCoeff; ++jc)
            {
                // reconstruct the value for all simulations
                cuts.col(iPointCut).segment(jc * nbNodes, nbNodes) = m_cutCoeff[jc].col(ipoint);
            }
            iPointCut += 1;
        }
        iterRegGrid->next();
    }
    cuts.conservativeResize(nbCutsCoeff * nbNodes, iPointCut);
    return cuts;
}
}
