// Copyright (C) 2016 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#ifdef USE_MPI
#include <memory>
#include <array>
#include <functional>
#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include "libflow/core/parallelism/ParallelComputeGridSplitting.h"
#include "libflow/core/parallelism/ParallelHiter.h"


using namespace Eigen ;

namespace libflow
{
ArrayXi paraOptimalSplitting(const ArrayXi &p_initDimension, const Array< bool, Dynamic, 1> &p_bdimToSplit, const boost::mpi::communicator &p_world)
{
    ArrayXi splittingRatio = ArrayXi::Constant(p_initDimension.size(), 1);
    std::vector<int> prime = primeNumber(p_world.size());
    ArrayXi currentDimension(p_initDimension);
    for (int id  = 0; id < currentDimension.size(); ++id)
        if (!p_bdimToSplit(id))
            currentDimension(id) = 1; // so that it won't split

    for (int i = static_cast<int>(prime.size()) - 1; i >= 0 ; --i)
    {
        ArrayXi::Index maxDim ;
        int  idimMax = currentDimension.maxCoeff(&maxDim);
        currentDimension(maxDim) = idimMax / prime[i];
        // check the dimension is not degenerated
        if (currentDimension(maxDim) < 2)
            // go on  and test smaller prime number
            currentDimension(maxDim) = idimMax;
        else
            splittingRatio(maxDim) *= prime[i];
    }
    return splittingRatio;
}


Array< std::array<int, 2 >, Dynamic, 1 >   paraSplitComputationGridsProc(const ArrayXi   &p_grid,
        const  ArrayXi   &p_splittingRatio, const int &p_iProc)
{
    size_t nDim = p_grid.size();
    // check ratio and processor
    int dim_mult = 1;
    for (size_t id = 0 ; id < nDim; ++id)
        dim_mult *= p_splittingRatio(id);
    //assert(dim_mult <= p_world.size());
    if (p_iProc >= dim_mult)
    {
        return Array< std::array<int, 2 >, Dynamic, 1 > () ;
    }
    else
    {
        int iProcLoc =  p_iProc;
        Array< std::array<int, 2 >, Dynamic, 1 > dim_ret(nDim) ;
        for (size_t id = 0 ; id < nDim; ++id)
        {
            int processor_coordinate = iProcLoc % p_splittingRatio(id);
            iProcLoc /= p_splittingRatio(id);
            int nb_m_meshPerProc = p_grid(id) / p_splittingRatio(id);
            int nrest = p_grid(id) % p_splittingRatio(id);
            int igridMin = processor_coordinate * nb_m_meshPerProc + ((processor_coordinate < nrest) ? processor_coordinate : nrest);
            int nb_m_meshPerProc_cor = nb_m_meshPerProc + ((processor_coordinate < nrest) ? 1 : 0);
            int igridMax = ((processor_coordinate == p_splittingRatio(id) - 1) ? p_grid(id) : igridMin + nb_m_meshPerProc_cor);
            dim_ret(id)[0] =   igridMin;
            dim_ret(id)[1] =   igridMax;
        }
        return dim_ret ;
    }
}



Array< std::array<int, 2 >, Dynamic, 1 >  ParallelComputeGridSplitting::paraInterHCube(const Ref<const Array< std::array<int, 2 >, Dynamic, 1 > >   &p_hCube1,
        const Ref<const Array< std::array<int, 2 >, Dynamic, 1 > >   &p_hCube2,
        bool &p_bIntersectionFlag)
{

    Array< std::array<int, 2 >, Dynamic, 1 >  hCubeRes(m_nDim);

    bool bRes = true;     // Intersection flag: hcubes have not null intersection ?
    // Compute intersection of each dimension and stop when
    // one dimension has no intersection!
    if ((p_hCube1(0)[0] != -1) && (p_hCube2(0)[0] != -1))
    {
        for (int  n = 0; n < p_hCube1.size() && bRes; n++)
        {
            hCubeRes(n)[0] = std::max(p_hCube1(n)[0], p_hCube2(n)[0]);
            hCubeRes(n)[1] = std::min(p_hCube1(n)[1], p_hCube2(n)[1]);
            bRes = (hCubeRes(n)[0] < hCubeRes(n)[1]);
        }
    }
    else
    {
        bRes = false;
    }
    if (!bRes)
    {
        for (size_t n = 0; n < m_nDim; n++)
        {
            hCubeRes(n)[0] = -1;
            hCubeRes(n)[1] = -1;
        }
    }

    // Save the intersection result
    p_bIntersectionFlag = bRes;
    return hCubeRes;
}



void ParallelComputeGridSplitting::paraRoutingSchedule(const Array<  std::array<int, 2 >, Dynamic, Dynamic >   &p_gridNeededPerProc,
        const int &p_colNeededPerProc,
        std::vector<bool>    &p_bIntersecHCubeRecLoc,
        std::vector<bool>   &p_bIntersecHCubeSendLoc,
        Array<  std::array<int, 2 >, Dynamic, Dynamic >   &p_gridComingFromProcessorLoc,
        Array<  std::array<int, 2 >, Dynamic, Dynamic >   &p_gridToSendToProcessorLoc)
{
    p_gridToSendToProcessorLoc.resize(p_gridNeededPerProc.rows(), p_colNeededPerProc);
    p_gridComingFromProcessorLoc.resize(p_gridNeededPerProc.rows(), m_nbProcessorUsedPrev);
    //  do we have to send to processor p
    p_bIntersecHCubeSendLoc.resize(p_gridNeededPerProc.cols());
    // do we have to receive
    p_bIntersecHCubeRecLoc.resize(m_nbProcessorUsedPrev);
    // Compute 'Send routing plan' ---------------------------------------------------
    //   compute intersections of hcubes GRID2 needed by each processor   with hcubes GRID1 of each processor
    if (m_world.rank() < m_nbProcessorUsedPrev)
    {
        for (int p = 0; p < p_colNeededPerProc; p++)
        {
            bool bInterSec ;
            p_gridToSendToProcessorLoc.col(p) = paraInterHCube(m_meshPerProcOldGrid.col(m_world.rank()), p_gridNeededPerProc.col(p),  bInterSec);
            p_bIntersecHCubeSendLoc[p] = bInterSec;
        }
        // Else: the proc has nothing to send
    }
    else
    {
        // - set the send intersection flags to false for any target processor
        for (int p = 0; p < p_colNeededPerProc; p++)
        {
            p_bIntersecHCubeSendLoc[p] = false;
        }
    }
    // Compute 'Recv routing plan' ---------------------------------------------------
    if (m_world.rank() < p_colNeededPerProc)
    {
        for (int p = 0; p < m_nbProcessorUsedPrev ; p++)
        {
            bool bInterSec ;
            p_gridComingFromProcessorLoc.col(p) = paraInterHCube(p_gridNeededPerProc.col(m_world.rank()), m_meshPerProcOldGrid.col(p), bInterSec);
            p_bIntersecHCubeRecLoc[p] = bInterSec;
        }
    }
    else
    {
        for (int p = 0; p < m_nbProcessorUsedPrev; p++)
        {
            p_bIntersecHCubeRecLoc[p] = false;
        }
    }
}




ParallelComputeGridSplitting::ParallelComputeGridSplitting(const ArrayXi  &p_initialDimension,
        const std::function<  Array<  std::array<int, 2 >, Dynamic, 1 >(const Array<  std::array<int, 2 >, Dynamic, 1 > &) > &p_gridCalcExt,
        const ArrayXi  &p_splittingRatio, const boost::mpi::communicator &p_world):
    m_nDim(p_initialDimension.size()), m_nbProcessorUsed(0), m_meshPerProc(), m_meshPerProcOldGrid(), m_world(p_world)
{
    m_nbProcessorUsed = p_splittingRatio.prod();
    m_nbProcessorUsedPrev = m_nbProcessorUsed;
    assert(m_nbProcessorUsed <= m_world.size());

    m_meshPerProc.resize(p_initialDimension.size(), m_world.size());
    m_meshPerProcOldGrid.resize(m_nDim, m_world.size());
    for (int iproc = 0 ; iproc < m_world.size() ; ++iproc)
    {
        Array< std::array<int, 2 >, Dynamic, 1 >  paraSplit  = paraSplitComputationGridsProc(p_initialDimension, p_splittingRatio, iproc);
        if (paraSplit.size() > 0)
            m_meshPerProc.col(iproc)  = paraSplit;
    }
    // default
    m_meshPerProcOldGrid = m_meshPerProc;
    m_extendGridProcOldGrid.resize(m_nDim,   m_world.size());
    // calculated extended grid for each processor
    for (int ip = 0; ip < m_nbProcessorUsed; ++ip)
    {
        Map<Array<  std::array<int, 2 >, Dynamic, 1 > > meshPerProcOldGridLoc(m_meshPerProcOldGrid.col(ip).data(), m_meshPerProcOldGrid.rows());
        m_extendGridProcOldGrid.col(ip) =  p_gridCalcExt(meshPerProcOldGridLoc);
    }
    if (m_world.rank() < m_nbProcessorUsed)
    {
        m_iSizeExtendedArray = 1;
        for (size_t id = 0 ; id < m_nDim; ++id)
            m_iSizeExtendedArray *= m_extendGridProcOldGrid(id, m_world.rank())[1] -  m_extendGridProcOldGrid(id, m_world.rank())[0];
    }
    else
    {
        m_iSizeExtendedArray = 0;
    }
    // routing plan to be effected
    paraRoutingSchedule(m_extendGridProcOldGrid, m_nbProcessorUsed, m_bIntersecHCubeRec, m_bIntersecHCubeSend, m_gridComingFromProcessor, m_gridToSendToProcessor);
}

ParallelComputeGridSplitting::ParallelComputeGridSplitting(const ArrayXi  &p_initialDimension, const  ArrayXi  &p_initialDimensionPrev,
        const std::function<  Array<  std::array<int, 2 >, Dynamic, 1 >(const Array<  std::array<int, 2 >, Dynamic, 1 > &) > &p_gridCalcExt,
        const ArrayXi   &p_splittingRatio, const ArrayXi    &p_splittingRatioPrev, const boost::mpi::communicator &p_world):
    m_nDim(p_initialDimension.size()), m_nbProcessorUsed(0), m_meshPerProc(), m_meshPerProcOldGrid(), m_world(p_world)
{
    m_nbProcessorUsed = p_splittingRatio.prod();
    m_nbProcessorUsedPrev = p_splittingRatioPrev.prod();
    assert(m_nbProcessorUsed <= m_world.size());
    // GRID1
    m_meshPerProc.resize(m_nDim, m_world.size());
    for (int iproc = 0 ; iproc < m_world.size() ; ++iproc)
    {
        Array< std::array<int, 2 >, Dynamic, 1 >  paraSplit  = paraSplitComputationGridsProc(p_initialDimension, p_splittingRatio, iproc);
        if (paraSplit.size() > 0)
            m_meshPerProc.col(iproc)  = paraSplit;
        else
            break ;
    }
    m_meshPerProcOldGrid.resize(m_nDim, m_world.size());
    for (int iproc = 0 ; iproc < m_world.size() ; ++iproc)
    {
        Array< std::array<int, 2 >, Dynamic, 1 >  paraSplit  = paraSplitComputationGridsProc(p_initialDimensionPrev, p_splittingRatioPrev, iproc);
        if (paraSplit.size() > 0)
            m_meshPerProcOldGrid.col(iproc) = paraSplit;
        else
            break ;
    }
    m_extendGridProcOldGrid.resize(m_nDim, m_world.size());
    // calculated extended GRID2
    for (int ip = 0; ip < m_nbProcessorUsed; ++ip)
    {
        Map<Array<  std::array<int, 2 >, Dynamic, 1 > > meshPerProcLoc(m_meshPerProc.col(ip).data(), m_meshPerProc.rows());
        m_extendGridProcOldGrid.col(ip) = p_gridCalcExt(meshPerProcLoc);
    }
    if (m_world.rank() < m_nbProcessorUsed)
    {
        m_iSizeExtendedArray = 1;
        for (size_t id = 0 ; id < m_nDim; ++id)
            m_iSizeExtendedArray *= m_extendGridProcOldGrid(id, m_world.rank())[1] -  m_extendGridProcOldGrid(id, m_world.rank())[0];
    }
    else
    {
        m_iSizeExtendedArray = 0;
    }
    // routing plan to be effected
    paraRoutingSchedule(m_extendGridProcOldGrid, m_nbProcessorUsed, m_bIntersecHCubeRec, m_bIntersecHCubeSend, m_gridComingFromProcessor, m_gridToSendToProcessor);
}


ParallelComputeGridSplitting::ParallelComputeGridSplitting(const ArrayXi  &p_initialDimension,
        const ArrayXi  &p_splittingRatio, const boost::mpi::communicator &p_world):
    m_nDim(p_initialDimension.size()), m_nbProcessorUsed(0), m_meshPerProc(), m_world(p_world)
{
    m_nbProcessorUsed = p_splittingRatio.prod();
    m_nbProcessorUsedPrev = m_nbProcessorUsed ;
    assert(m_nbProcessorUsed <= m_world.size());
    // resize extended
    m_extendGridProcOldGrid.resize(m_nDim, m_world.size());
    m_meshPerProc.resize(m_nDim, m_world.size());
    int nbProcessorReceiving = 0;
    bool bExtended = false ;
    for (int iproc = 0 ; iproc < m_world.size() ; ++iproc)
    {
        Array< std::array<int, 2 >, Dynamic, 1 >  paraSplit  = paraSplitComputationGridsProc(p_initialDimension, p_splittingRatio, iproc);
        if (paraSplit.size() > 0)
        {
            m_meshPerProc.col(iproc)  = paraSplit;
            nbProcessorReceiving += 1 ;
            if (m_world.rank() == iproc)
                bExtended = true;
        }
    }
    m_meshPerProcOldGrid = m_meshPerProc;
    m_extendGridProcOldGrid = m_meshPerProc;
    if (bExtended)
    {
        m_iSizeExtendedArray = 1;
        for (size_t id = 0 ; id < m_nDim ; ++id)
            m_iSizeExtendedArray *= m_extendGridProcOldGrid(id, m_world.rank())[1] -  m_extendGridProcOldGrid(id, m_world.rank())[0];
    }
    else
    {
        m_iSizeExtendedArray = 0;
    }
}

Array<  std::array<int, 2 >, Dynamic, 1 >  ParallelComputeGridSplitting::getCurrentCalculationGrid() const
{
    if (m_world.rank() < m_nbProcessorUsed)
        return m_meshPerProc.col(m_world.rank());
    else
        return  Array<  std::array<int, 2 >, Dynamic, 1 > ();
}

Array<  std::array<int, 2 >, Dynamic, 1 >  ParallelComputeGridSplitting::getPreviousCalculationGrid() const
{
    boost::mpi::communicator m_world;
    if (m_world.rank() < m_nbProcessorUsedPrev)
        return m_meshPerProcOldGrid.col(m_world.rank());
    else
        return  Array<  std::array<int, 2 >, Dynamic, 1 > ();

}

const Array<  std::array<int, 2 >, Dynamic, Dynamic > &ParallelComputeGridSplitting::getCurrentCalculationGridPerProc()
{
    return m_meshPerProc ;
}

int ParallelComputeGridSplitting::getNbProcessorUsed() const
{
    return  m_nbProcessorUsed;
}

int ParallelComputeGridSplitting::getNbProcessorUsedPrev() const
{
    return  m_nbProcessorUsedPrev;
}


Array<  std::array<int, 2 >, Dynamic, 1 >  ParallelComputeGridSplitting::getExtendedGridProcOldGrid() const
{
    if (m_world.rank() < m_nbProcessorUsed)
        return m_extendGridProcOldGrid.col(m_world.rank());
    else
        return Array<  std::array<int, 2 >, Dynamic, 1 >();
}

}
#endif
