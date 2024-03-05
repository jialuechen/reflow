
#ifndef PARALLELCOMPUTEGRIDSPLITING_H
#define PARALLELCOMPUTEGRIDSPLITING_H
#include <memory>
#include <array>
#include <functional>
#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <Eigen/Dense>
#include "libflow/core/utils/primeNumber.h"
#include "libflow/core/utils/eigenSerialization.h"
#include "libflow/core/parallelism/ParallelHiter.h"



/** \file ParallelComputeGridSplitting.h
 * \brief  A multidimensional Cartesian grid split is achieved between processors
 *         Data are spread on a grid (GRID1) local to the processor.
 *         Different cases are possible :
 *             -Each processor may need to get from a grid (GRID3) the data for an extended grid (GRID2) defined from grid GRID1
 *             -Each processor may need to get from a grid (GRID3) the data for an extended grid (GRID2) defined from grid GRID3
 *             -Each processor needs to get the data on another grid (GRID2) (GRID3 identical to GRID1)
 *         Needed data from other processors are sent and received.
 *         see article "Stochastic control optimization & simulation applied to energy management:
 *         From 1-D to N-D problem distributions, on clusters, supercomputers and Grids "
 *         by Vialle, Warin, Mercier
 * \author Xavier Warin
 */

namespace libflow
{
/// \brief Optimal split of the grid
///  The dimension with higher number of point is first split recursively
/// \param  p_initDimension   number of meshes in each direction
/// \param  p_bdimToSplit     for each dimension,  true if the dimension should be split
/// \param  p_world           MPI communicator
/// \return splitting of the grid in each dimension
Eigen::ArrayXi paraOptimalSplitting(const Eigen::ArrayXi &p_initDimension, const Eigen::Array< bool, Eigen::Dynamic, 1> &p_bdimToSplit, const boost::mpi::communicator &p_world);


///  \brief Split the grid  and give the grid for a given processor
/// \param p_grid             Give the current grid
/// \param p_splittingRatio   For each dimension give the splitting ratio
/// \param p_iProc            Processor number
/// \param  p_world           MPI communicator
/// \return splitting of the grid in each dimension
Eigen::ArrayXi paraOptimalSplitting(const Eigen::ArrayXi &p_initDimension, const Eigen::Array< bool, Eigen::Dynamic, 1> &p_bdimToSplit, const boost::mpi::communicator &p_world);


///  \brief Split the grid  and give the grid for a given processor
/// \param p_grid             Give the current grid
/// \param p_splittingRatio   For each dimension give the splitting ratio
/// \param p_iProc            Processor number
/// \return grid owned by the processor
Eigen::Array< std::array<int, 2 >, Eigen::Dynamic, 1 >   paraSplitComputationGridsProc(const Eigen::ArrayXi   &p_grid,
        const  Eigen::ArrayXi   &p_splittingRatio, const int &p_iProc);

/// \class  ParallelComputeGridSplitting ParallelComputeGridSplitting.h
///         Split the grids of point to split work between processors
class ParallelComputeGridSplitting
{
private :

    size_t m_nDim ; ///< dimension of the hypercube
    int m_nbProcessorUsed; ///< Number of processors used in parallel
    int m_nbProcessorUsedPrev; ///< Number of processors used in parallel at previous step
    Eigen::Array<  std::array<int, 2 >, Eigen::Dynamic, Eigen::Dynamic > m_meshPerProc ; ///<  For  each processor (column) , each stock number (row), store the meshes owned by the processor
    Eigen::Array<  std::array<int, 2 >, Eigen::Dynamic, Eigen::Dynamic > m_meshPerProcOldGrid ; ///<  For  each processor (column) , each stock number (row), store the meshes owned by the processor at previous iteration

    boost::mpi::communicator m_world ; ///< MPI communicator
    // for current processor defines extended grid for all processors (GRID2)
    Eigen::Array<  std::array<int, 2 >, Eigen::Dynamic, Eigen::Dynamic >   m_extendGridProcOldGrid;

    // size of extended array for current processor
    int m_iSizeExtendedArray ;

    // To receive  data
    // store the Hcube to be received
    Eigen::Array<  std::array<int, 2 >, Eigen::Dynamic, Eigen::Dynamic >  m_gridComingFromProcessor;
    // store if receiving from other processor
    std::vector< bool>   m_bIntersecHCubeRec;
    // store Hcube to be send to other processors
    Eigen::Array<  std::array<int, 2 >, Eigen::Dynamic, Eigen::Dynamic >  m_gridToSendToProcessor;
    // store if sending data to other processor
    std::vector< bool>   m_bIntersecHCubeSend;

    /// \brief Compute the intersection of Hypercubes (grids)  p_hCube1, p_hCube2
    /// \param p_hCube1             Hypercube 1
    /// \param p_hCube2             Hypercube 2
    /// \param p_bIntersectionFlag   true if non empty intersection
    Eigen::Array< std::array<int, 2 >, Eigen::Dynamic, 1 >  paraInterHCube(const Eigen::Ref<const Eigen::Array< std::array<int, 2 >, Eigen::Dynamic, 1 > >   &p_hCube1,
            const Eigen::Ref<const Eigen::Array< std::array<int, 2 >, Eigen::Dynamic, 1 > >   &p_hCube2,
            bool &p_bIntersectionFlag);

    /// \brief Intersect data from different grids
    /// \param p_dataToIntersect      Data on grid used to intersect
    /// \param p_gridToIntersectWith  Grid associated to dataToIntersect
    /// \param p_grid                 Grid to intersect with p_gridToIntersectWith
    /// \param p_data                 Data resulting from intersection of the grids
    /// \param p_firstDimData         Size of the first dimension of  p_tabOwnedByProcessor if allocated
    /// \param p_gridIntersected      Hypercube resulting from intersection
    template< typename T>
    void  paraIntersecDataHyperCube(const  Eigen::Ref < const Eigen::Array< T, Eigen::Dynamic, Eigen::Dynamic > > &p_dataToIntersect,
                                    const  Eigen::Array< std::array<int, 2 >, Eigen::Dynamic, 1 > &p_gridToIntersectWith,
                                    const  Eigen::Array< std::array<int, 2 >, Eigen::Dynamic, 1 > &p_grid,
                                    Eigen::Array< T, Eigen::Dynamic, Eigen::Dynamic >    &p_data,
                                    const int &p_firstDimData,
                                    Eigen::Array< std::array<int, 2 >, Eigen::Dynamic, 1 >    &p_gridIntersected)
    {
        bool bIntersectionFlag;
        p_gridIntersected = paraInterHCube(p_gridToIntersectWith, p_grid, bIntersectionFlag);
        int isize  = 1 ;
        for (size_t id = 0 ; id < m_nDim ; ++id)
            isize *= p_gridIntersected(id)[1] - p_gridIntersected(id)[0];
        p_data.resize(p_firstDimData, isize);
        ParallelHiter hiter(p_gridIntersected, p_gridToIntersectWith);
        int segment = hiter.segmentSize();
        int iposStock = 0;
        while (hiter.isValid())
        {
            int iposIter = hiter.get();
            p_data.block(0, iposStock, p_firstDimData, segment) = p_dataToIntersect.block(0, iposIter, p_firstDimData, segment);
            iposStock += segment;
            hiter.next();
        }
    }

    /// \brief Compute the routing plan
    ///         That is  Hypercube to send to each processor and the HyperCube to receive from other processor
    /// \param p_gridNeededPerProc                          Grid needed  by each processor (index by processor)
    /// \param p_colNeededPerProc                           Number of columns of p_gridNeededPerProc to consider
    /// \param p_bIntersecHCubeRecLOC                      True if current processor  receives data from current processor
    /// \param p_bIntersecHCubeSendLoc                     True if current processor sends to processor p
    /// \param p_gridComingFromProcessorLoc                vector of grids coming from other processors
    /// \param p_gridToSendToProcessorLoc                  vector grid of points to send to other processor
    void paraRoutingSchedule(const Eigen::Array<  std::array<int, 2 >, Eigen::Dynamic, Eigen::Dynamic >   &p_gridNeededPerProc,
                             const int &p_colNeededPerProc,
                             std::vector<bool>    &p_bIntersecHCubeRecLoc,
                             std::vector<bool>   &p_bIntersecHCubeSendLoc,
                             Eigen::Array<  std::array<int, 2 >, Eigen::Dynamic, Eigen::Dynamic >   &p_gridComingFromProcessorLoc,
                             Eigen::Array<  std::array<int, 2 >, Eigen::Dynamic, Eigen::Dynamic >   &p_gridToSendToProcessorLoc);

    /// \brief Execute the previously calculated routing
    /// \param p_bIntersecHCubeRecLoc                   True if current processor  receives data from current processor
    /// \param p_bIntersecHCubeSendLoc                  True if current processor sends to processor p
    /// \param p_gridComingFromProcessorLoc              vector of grids coming from other processors
    /// \param p_gridToSendToProcessorLoc                vector grid of points to send to other processor
    /// \param p_tabOwnedByProcessor                     Array owned by the processor
    /// \param p_firstDimData                            Size of the first dimension of  p_tabOwnedByProcessor if allocated
    /// \param p_tabReceiveFromOther                     Array  coming from other processor
    ///  T can be short int, int, double , float
    template< typename T>
    void paraRoutingExec(const  std::vector<bool>    &p_bIntersecHCubeRecLoc,
                         const  std::vector<bool>   &p_bIntersecHCubeSendLoc,
                         const Eigen::Array<  std::array<int, 2 >, Eigen::Dynamic, Eigen::Dynamic >    &p_gridComingFromProcessorLoc,
                         const Eigen::Array<  std::array<int, 2 >, Eigen::Dynamic, Eigen::Dynamic >    &p_gridToSendToProcessorLoc,
                         const Eigen::Array< T, Eigen::Dynamic, Eigen::Dynamic >   &p_tabOwnedByProcessor,
                         const int &p_firstDimData,
                         std::vector< std::shared_ptr< Eigen::Array< T, Eigen::Dynamic, Eigen::Dynamic > > >    &p_tabReceiveFromOther)
    {
        // Size of communicator
        int iRecSize = 0 ;
        for (int iproc = 0 ; iproc < m_world.size() ; ++iproc)
            if (iproc < static_cast<int>(p_bIntersecHCubeRecLoc.size()))
                if ((p_bIntersecHCubeRecLoc[iproc]) && (iproc != m_world.rank())) iRecSize++;
        std::vector<  boost::mpi::request > reqRec(iRecSize);
        int iRec = 0 ;
        // only one message right now (communication are not spread)
        for (int iproc = 0 ; iproc < static_cast<int>(p_bIntersecHCubeRecLoc.size()) ; ++iproc)
        {
            // receive
            if ((p_bIntersecHCubeRecLoc[iproc]) && (iproc != m_world.rank()))
            {
                // Number of mesh point associated
                int nbPointRec = 1;
                for (int idim = 0 ; idim < p_gridComingFromProcessorLoc.rows() ; ++idim)
                {
                    nbPointRec *= p_gridComingFromProcessorLoc(idim, iproc)[1] - p_gridComingFromProcessorLoc(idim, iproc)[0];
                }
                p_tabReceiveFromOther[iproc] = std::make_shared< Eigen::Array< T, Eigen::Dynamic, Eigen::Dynamic > >(p_firstDimData, nbPointRec);
                // communication
                int imesg_iproc = 0;
                //   receive the data from  other processors
                reqRec[iRec++] = m_world.irecv(iproc, imesg_iproc++, *p_tabReceiveFromOther[iproc]);
            }
        }
        // send
        // Array for sending to other processors
        std::vector< std::shared_ptr< Eigen::Array< T,  Eigen::Dynamic, Eigen::Dynamic >  >  > tabSend(m_world.size());
        int iSendSize = 0 ;
        for (int iproc = 0 ; iproc < m_world.size() ; ++iproc)
            if (iproc < static_cast<int>(p_bIntersecHCubeSendLoc.size()))
                if ((p_bIntersecHCubeSendLoc[iproc]) && (iproc != m_world.rank())) iSendSize++ ;
        std::vector<   boost::mpi::request > reqSend(iSendSize);
        int iSend = 0 ;
        for (int iproc = 0 ; iproc < static_cast<int>(p_bIntersecHCubeSendLoc.size()) ; ++iproc)
        {
            if ((p_bIntersecHCubeSendLoc[iproc]) && (iproc != m_world.rank()))
            {
                ParallelHiter hiter(p_gridToSendToProcessorLoc.col(iproc), m_meshPerProcOldGrid.col(m_world.rank()));
                // - compute the number of data to send
                int nbPointSend = hiter.hCubeSize();
                tabSend[iproc] =  std::make_shared<Eigen::Array< T, Eigen::Dynamic, Eigen::Dynamic > >(p_firstDimData, nbPointSend);
                // copy HyperCube Data MPI operation
                int segment = hiter.segmentSize();
                int iposStock = 0;
                while (hiter.isValid())
                {
                    int iposIter = hiter.get();
                    tabSend[iproc]->block(0, iposStock, p_firstDimData, segment) = p_tabOwnedByProcessor.block(0, iposIter,  p_firstDimData, segment);
                    iposStock += segment;
                    hiter.next();
                }
                // send
                int imesg_iproc = 0;
                // Communications
                reqSend[iSend++] =  m_world.isend(iproc, imesg_iproc++, * tabSend[iproc]);
            }
        }
        boost::mpi::wait_all(reqRec.begin(), reqRec.end());
        boost::mpi::wait_all(reqSend.begin(), reqSend.end());
    }


    /// \brief Intersect with itself
    /// \param p_gridSendingFromItself                       Grids coming  a processor to itself
    /// \param p_tabOwnedByProcessor                         Array owned by processor
    /// \param p_firstDimData                            Size of the first dimension of  p_tabOwnedByProcessor if allocated
    /// \param p_gridTarget                                 Target grid to fill (so to intersect with  p_gridSendingFromItself )
    /// \param p_tabOwnedByProcessorExtended                 Array extended with contribution from other processor
    template<typename T>
    void paraIntersecWithItself(const Eigen::Ref<const Eigen::Array< std::array<int, 2 >, Eigen::Dynamic, 1 > > &p_gridSendingFromItself,
                                const Eigen::Array< T, Eigen::Dynamic, Eigen::Dynamic >     &p_tabOwnedByProcessor,
                                const int &p_firstDimData,
                                const Eigen::Ref<const Eigen::Array< std::array<int, 2 >, Eigen::Dynamic, 1 > > &p_gridTarget,
                                Eigen::Array< T, Eigen::Dynamic, Eigen::Dynamic >     &p_tabOwnedByProcessExtended)
    {
        ParallelHiter hiter(p_gridSendingFromItself, p_gridTarget);
        // copy HyperCube
        int segment = hiter.segmentSize();
        int iposStock = 0;
        while (hiter.isValid())
        {
            int iposIter = hiter.get();
            p_tabOwnedByProcessExtended.block(0, iposIter, p_firstDimData, segment) = p_tabOwnedByProcessor.block(0, iposStock, p_firstDimData, segment);
            iposStock += segment;
            hiter.next();
        }
    }

    /// \brief Unpack data previously sent from other processor and fill the Array value
    /// \param p_tabReceiveFromOther                         All array portions  coming from other processors to unpack
    /// \param p_firstDimData                                 Size of the first dimension of  p_tabReceiveFromOther if allocated
    /// \param p_bIntersecHCubeRecLoc                        True if current processor  receive data from current processor
    /// \param p_gridComingFromProcessorLoc                  vector of grids coming from other processors
    /// \param p_gridTarget                                  Target grid to fill
    /// \param p_tabOwnedByProcessExtended                   Extended array of data owned by processor
    template< typename T>
    void  paraUnpackData(const std::vector< std::shared_ptr< Eigen::Array< T, Eigen::Dynamic, Eigen::Dynamic > > > &p_tabReceiveFromOther,
                         const int &p_firstDimData,
                         const  std::vector<bool>     &p_bIntersecHCubeRecLoc,
                         const  Eigen::Array<  std::array<int, 2 >, Eigen::Dynamic, Eigen::Dynamic >            &p_gridComingFromProcessorLoc,
                         const  Eigen::Ref<const Eigen::Array< std::array<int, 2 >, Eigen::Dynamic, 1 > >        &p_gridTarget,
                         Eigen::Array< T, Eigen::Dynamic, Eigen::Dynamic >   &p_tabOwnedByProcessExtended)
    {
        for (int iproc = 0 ; iproc <  static_cast<int>(p_bIntersecHCubeRecLoc.size()) ; ++iproc)
        {
            if ((iproc != m_world.rank()) && (p_bIntersecHCubeRecLoc[iproc]))
            {
                ParallelHiter hiter(p_gridComingFromProcessorLoc.col(iproc), p_gridTarget);
                int segment = hiter.segmentSize();
                int iposStock = 0;
                while (hiter.isValid())
                {
                    int iposIter = hiter.get();
                    p_tabOwnedByProcessExtended.block(0, iposIter, p_firstDimData, segment) = p_tabReceiveFromOther[iproc]->block(0, iposStock, p_firstDimData, segment);
                    iposStock += segment;
                    hiter.next();
                }
            }
        }
    }


public :

    /// \brief default : case with GRID1 and GRID2
    /// \param p_initialDimension            Vector of number of points in each dimension
    /// \param p_gridCalcExt                 Permits to calculated extended grid for a given grid
    /// \param p_splittingRatio              For each dimension define how to split  the mesh
    /// \param  p_world                      MPI communicator
    ParallelComputeGridSplitting(const Eigen::ArrayXi  &p_initialDimension,
                                 const std::function<  Eigen::Array<  std::array<int, 2 >, Eigen::Dynamic, 1 >(const Eigen::Array<  std::array<int, 2 >, Eigen::Dynamic, 1 > &) > &p_gridCalcExt,
                                 const Eigen::ArrayXi  &p_splittingRatio, const boost::mpi::communicator &p_world);

    /// \brief Here constructor with GRID1, GRID2 and GRID3
    /// \param p_initialDimension                  Vector of grid discretization for GRID1
    /// \param p_initialDimensionPrev              Vector of grid discretization for GRID3
    /// \param p_gridCalcExt                       Permits to calculated extended grid GRID2 from GRID1
    /// \param p_splittingRatio                    For each dimension define how to split  the mesh at the current step
    /// \param p_splittingRatioPrev                For each dimension define how to split  the mesh at the previous step
    /// \param  p_world                      MPI communicator
    ParallelComputeGridSplitting(const Eigen::ArrayXi  &p_initialDimension, const  Eigen::ArrayXi  &p_initialDimensionPrev,
                                 const std::function<  Eigen::Array<  std::array<int, 2 >, Eigen::Dynamic, 1 >(const Eigen::Array<  std::array<int, 2 >, Eigen::Dynamic, 1 > &) > &p_gridCalcExt,
                                 const Eigen::ArrayXi   &p_splittingRatio,
                                 const Eigen::ArrayXi   &p_splittingRatioPrev, const boost::mpi::communicator &p_world);

    ///  \brief Last constructor used in simulation : no extended grid created
    ///  Only recalculate the repartition between processor of the grid
    /// \param p_initialDimension                Vector of the problem dimension
    /// \param p_splittingRatio                  For each dimension define how to split  the mesh so that prod( p_splittingRatio)
    /// \param  p_world                      MPI communicator
    ParallelComputeGridSplitting(const Eigen::ArrayXi  &p_initialDimension,
                                 const Eigen::ArrayXi  &p_splittingRatio, const boost::mpi::communicator &p_world);

    /// Get grid calculated by current processor
    Eigen::Array<  std::array<int, 2 >, Eigen::Dynamic, 1 >  getCurrentCalculationGrid() const;

    /// Get grid calculated by current processor previous date
    Eigen::Array<  std::array<int, 2 >, Eigen::Dynamic, 1 >  getPreviousCalculationGrid() const;

    /// Get Current grid per processor
    const Eigen::Array<  std::array<int, 2 >, Eigen::Dynamic, Eigen::Dynamic > &getCurrentCalculationGridPerProc();

    /// Get number of processor used
    int getNbProcessorUsed() const ;

    /// Get number of processor used at the previous step
    int getNbProcessorUsedPrev() const ;

    /// \brief Get extended grid used by current processor
    Eigen::Array<  std::array<int, 2 >, Eigen::Dynamic, 1 >  getExtendedGridProcOldGrid() const ;

    /// \brief Calculate an array value extended on a grid extended
    /// \param  p_tabOwnedByProcess                  Array of data owned by processor
    /// \return Extended array of data needed by processor
    template< typename T>
    Eigen::Array< T, Eigen::Dynamic, Eigen::Dynamic> runOneStep(const Eigen::Array< T, Eigen::Dynamic, Eigen::Dynamic>     &p_tabOwnedByProcess)
    {

        // get row associated to returned array
        int firstDimSize = p_tabOwnedByProcess.rows();
        boost::mpi::broadcast(m_world, firstDimSize, 0);
        // allocate  new array
        Eigen::Array< T, Eigen::Dynamic, Eigen::Dynamic>  tabOwnedByProcessExtended(firstDimSize, m_iSizeExtendedArray);
        // Array to receive from other processors
        std::vector< std::shared_ptr< Eigen::Array< T, Eigen::Dynamic, Eigen::Dynamic>  > > tabReceiveFromOther(m_world.size());
        // routing achieved :
        paraRoutingExec<T>(m_bIntersecHCubeRec, m_bIntersecHCubeSend, m_gridComingFromProcessor, m_gridToSendToProcessor, p_tabOwnedByProcess, firstDimSize, tabReceiveFromOther);
        // use local mesh : if it has data to retrieve from its own data
        if (m_world.rank() < static_cast<int>(m_bIntersecHCubeRec.size()))
            if (m_bIntersecHCubeRec[m_world.rank()])
            {
                // now if necessary intersect the data with on the target grid
                int isize  = 1 ;
                for (size_t id = 0 ; id < m_nDim ; ++id)
                    isize *= m_gridToSendToProcessor(id, m_world.rank())[1] - m_gridToSendToProcessor(id, m_world.rank())[0];
                Eigen::Array< T, Eigen::Dynamic, Eigen::Dynamic>  data(p_tabOwnedByProcess.rows(), isize);
                ParallelHiter hiter(m_gridToSendToProcessor.col(m_world.rank()), m_meshPerProcOldGrid.col(m_world.rank()));
                int segment = hiter.segmentSize();
                int iposStock = 0;
                while (hiter.isValid())
                {
                    int iposIter = hiter.get();
                    data.block(0, iposStock, data.rows(), segment) = p_tabOwnedByProcess.block(0, iposIter, data.rows(), segment);
                    iposStock += segment;
                    hiter.next();
                }
                paraIntersecWithItself<T>(m_gridToSendToProcessor.col(m_world.rank()), data, firstDimSize, m_extendGridProcOldGrid.col(m_world.rank()), tabOwnedByProcessExtended);
            }
        // get back to cash flow object all the data that where stored in tabReceiveFromOther
        paraUnpackData<T>(tabReceiveFromOther, firstDimSize,  m_bIntersecHCubeRec,  m_gridComingFromProcessor, m_extendGridProcOldGrid.col(m_world.rank()), tabOwnedByProcessExtended);
        return tabOwnedByProcessExtended;
    }

    /// \brief Calculate an array value extended on a grid extended (one dimensional array)
    /// \param  p_tabOwnedByProcess                  Array of data owned by processor
    /// \return Extended array of data needed by processor
    template< typename T>
    Eigen::Array< T, Eigen::Dynamic, 1 > runOneStep(const Eigen::Array< T, Eigen::Dynamic, 1>     &p_tabOwnedByProcess)
    {
        // call to previous runOneStep
        Eigen::Array< T, Eigen::Dynamic, Eigen::Dynamic> tabOwnedByProcess = p_tabOwnedByProcess.transpose();
        return runOneStep<T>(tabOwnedByProcess).transpose();
    }


    /// \brief Calculate an array of  values extended on a grid extended
    /// \param p_tabOwnedByProcess                  Array owned by processor
    /// \param p_gridOnProc0                        Grid on processor p_iReconsProc
    /// \param p_iReconsProc                          Processor to reconstruct solution (default 0)
    /// \return reconstructed array
    template< typename T>
    Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic >  reconstruct(const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic >    &p_tabOwnedByProcess,
            const Eigen::Array< std::array<int, 2 >, Eigen::Dynamic, 1 >   &p_gridOnProc0,
            int p_iReconsProc = 0)
    {
        Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic > reconstructedArray;
        if (m_world.rank() == p_iReconsProc)
        {
            int isizeRecons = 1 ;
            for (size_t id = 0 ; id < m_nDim; ++id)
            {
                isizeRecons *= p_gridOnProc0(id)[1] - p_gridOnProc0(id)[0];
            }
            // allocate  new array
            reconstructedArray.resize(p_tabOwnedByProcess.rows(), isizeRecons);
        }
        // Array to receive from other processors
        std::vector< std::shared_ptr< Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic > > > tabReceiveFromOther(m_world.size());
        // local array for intersection, receive,send localization
        std::vector<bool>  bIntersecHCubeRecLoc;
        std::vector<bool>   bIntersecHCubeSendLoc;
        Eigen::Array<  std::array<int, 2 >, Eigen::Dynamic, Eigen::Dynamic >   gridComingFromProcessorLoc;
        Eigen::Array<  std::array<int, 2 >, Eigen::Dynamic, Eigen::Dynamic >    gridToSendToProcessorLoc;
        // Routing
        paraRoutingSchedule(p_gridOnProc0, 1, bIntersecHCubeRecLoc, bIntersecHCubeSendLoc, gridComingFromProcessorLoc, gridToSendToProcessorLoc);
        // first dimension of data
        int firstDimSize = p_tabOwnedByProcess.rows();
        // routing achieved :
        paraRoutingExec<T>(bIntersecHCubeRecLoc, bIntersecHCubeSendLoc, gridComingFromProcessorLoc, gridToSendToProcessorLoc, p_tabOwnedByProcess, firstDimSize, tabReceiveFromOther);
        // only for processor p_ReconsProc
        if (m_world.rank() == p_iReconsProc)
        {
            if (bIntersecHCubeRecLoc[p_iReconsProc])
            {
                // first intersect data owned by processor 0 with p_gridOnProc0
                Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic > tabOwnedByProcessIntersec0;
                Eigen::Array< std::array<int, 2 >, Eigen::Dynamic, 1 >   gridIntersected;
                paraIntersecDataHyperCube<T>(p_tabOwnedByProcess,  m_meshPerProc.col(0), p_gridOnProc0, tabOwnedByProcessIntersec0, firstDimSize, gridIntersected);
                // then
                paraIntersecWithItself<T>(gridIntersected, tabOwnedByProcessIntersec0, firstDimSize, p_gridOnProc0, reconstructedArray);
            }
            // get back to cash flow object all the data that where stored in tabReceiveFromOther
            paraUnpackData<T>(tabReceiveFromOther, firstDimSize, bIntersecHCubeRecLoc,  gridComingFromProcessorLoc, p_gridOnProc0, reconstructedArray);
        }
        return reconstructedArray;
    }

    /// \brief Calculate an array of  values extended on a grid extended (one dimensional array)
    /// \param p_tabOwnedByProcess                  Array owned by processor
    /// \param p_gridOnProc0                        Grid on processor p_iReconsProc
    /// \param p_iReconsProc                          Processor to reconstruct solution (default 0)
    /// \return reconstructed array
    template< typename T>
    Eigen::Array<T, Eigen::Dynamic, 1>  reconstruct(const Eigen::Array<T, Eigen::Dynamic, 1 >    &p_tabOwnedByProcess,
            const Eigen::Array< std::array<int, 2 >, Eigen::Dynamic, 1 >   &p_gridOnProc0,
            int p_iReconsProc = 0)
    {
        Eigen::Array< T, Eigen::Dynamic, Eigen::Dynamic> tabOwnedByProcess	= p_tabOwnedByProcess.transpose();
        Eigen::Array< T, Eigen::Dynamic, Eigen::Dynamic> ret = reconstruct(tabOwnedByProcess, p_gridOnProc0, p_iReconsProc).transpose();
        if (ret.size())
            return ret;
        else
            return Eigen::Array<T, Eigen::Dynamic, 1>();
    }





    ///  Reconstruction for all processor together
    /// \brief Calculate an array value extended on a grid extended
    /// \param p_tabOwnedByProcess                  Array owned by processor
    /// \param p_gridOnProc                         Grid needed on current processor
    /// \return  reconstructed array
    template< typename T>
    Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic >  reconstructAll(const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic >   &p_tabOwnedByProcess,
            const  Eigen::Array< std::array<int, 2 >, Eigen::Dynamic, 1 > &p_gridOnProc)
    {
        // to store all dimension needed by all processors
        std::vector<int>  DimPerProc;
        std::vector<int> DimCurrentProc(2 * m_nDim);
        for (size_t id = 0 ; id < m_nDim; ++id)
        {
            DimCurrentProc[id * 2] = p_gridOnProc(id)[0];
            DimCurrentProc[id * 2 + 1] = p_gridOnProc(id)[1];
        }
        // mpi stuff
        boost::mpi::all_gather<int>(m_world, DimCurrentProc.data(), 2 * m_nDim, DimPerProc);
        // get size of first dimension (can be 0 for some processor)
        int nSizeFirstDim = 0;
        boost::mpi::all_reduce(m_world, static_cast<int>(p_tabOwnedByProcess.rows()), nSizeFirstDim, boost::mpi::maximum<int>());
        // calculated extended grid for each processor
        Eigen::Array<  std::array<int, 2 >, Eigen::Dynamic, Eigen::Dynamic > extendedGridProcLoc(m_nDim, m_world.size());
        for (int ip = 0; ip < m_world.size(); ++ip)
            for (size_t id = 0 ; id < m_nDim; ++id)
            {
                extendedGridProcLoc(id, ip)[0] = DimPerProc[ip * 2 * m_nDim +  id * 2];
                extendedGridProcLoc(id, ip)[1] = DimPerProc[ip * 2 * m_nDim  +  id * 2 + 1];
            }

        int iSizeExtendedArrayLoc = 1;
        for (size_t id = 0 ; id < m_nDim; ++id)
            iSizeExtendedArrayLoc *= extendedGridProcLoc(id, m_world.rank())[1] - extendedGridProcLoc(id, m_world.rank())[0];
        // allocate  new array
        Eigen::Array< T, Eigen::Dynamic, Eigen::Dynamic > reconstructedArray(nSizeFirstDim, iSizeExtendedArrayLoc);
        // array and vector needed locally ( add _loc to definition)
        // store the ]cube to be received
        Eigen::Array< std::array< int, 2 >, Eigen::Dynamic,  Eigen::Dynamic >   gridComingFromProcessorLoc;
        // store if receiving from other processor
        std::vector< bool>   bIntersecHCubeRecLoc;
        // store Hcube to be send to other processors
        Eigen::Array< std::array< int, 2 >, Eigen::Dynamic,  Eigen::Dynamic >   gridToSendToProcessorLoc;
        // store if sending data to other processor
        std::vector< bool>   bIntersecHCubeSendLoc;
        // routing plan to be effected
        paraRoutingSchedule(extendedGridProcLoc, m_world.size(), bIntersecHCubeRecLoc, bIntersecHCubeSendLoc,
                            gridComingFromProcessorLoc, gridToSendToProcessorLoc);
        // Array to receive from other processors
        std::vector< std::shared_ptr< Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic > > > tabReceiveFromOther(m_world.size());
        // routing achieved :
        paraRoutingExec<T>(bIntersecHCubeRecLoc, bIntersecHCubeSendLoc, gridComingFromProcessorLoc, gridToSendToProcessorLoc,
                           p_tabOwnedByProcess, nSizeFirstDim, tabReceiveFromOther);
        // use local mesh : if it has data to retrieve from its own data
        if (m_world.rank() < static_cast<int>(bIntersecHCubeRecLoc.size()))
            if (bIntersecHCubeRecLoc[m_world.rank()])
            {
                // first intersect data owned by processor 0 with p_gridOnProc
                Eigen::Array< T, Eigen::Dynamic, Eigen::Dynamic > tabOwnedByProcessIntersec;
                Eigen::Array< std::array< int, 2 >, Eigen::Dynamic, 1 > gridIntersected(m_nDim);
                paraIntersecDataHyperCube<T>(p_tabOwnedByProcess,  m_meshPerProc.col(m_world.rank()), p_gridOnProc, tabOwnedByProcessIntersec, nSizeFirstDim, gridIntersected);
                paraIntersecWithItself<T>(gridIntersected, tabOwnedByProcessIntersec, nSizeFirstDim, p_gridOnProc, reconstructedArray);
            }
        // get back to cash flow objet all the data that where stored in tabReceiveFromOther
        paraUnpackData<T>(tabReceiveFromOther, nSizeFirstDim, bIntersecHCubeRecLoc,  gridComingFromProcessorLoc, extendedGridProcLoc.col(m_world.rank()), reconstructedArray);

        return reconstructedArray;
    }

    ///  Reconstruction for all processor together for one dimensional array
    /// \brief Calculate an array value extended on a grid extended
    /// \param p_tabOwnedByProcess                  Array owned by processor
    /// \param p_gridOnProc                         Grid needed on current processor
    /// \return  reconstructed array
    template< typename T>
    Eigen::Array<T, Eigen::Dynamic, 1 >  reconstructAll(const Eigen::Array<T, Eigen::Dynamic, 1 >   &p_tabOwnedByProcess,
            const  Eigen::Array< std::array<int, 2 >, Eigen::Dynamic, 1 > &p_gridOnProc)
    {
        Eigen::Array< T, Eigen::Dynamic, Eigen::Dynamic> tabOwnedByProcess	= p_tabOwnedByProcess.transpose();
        Eigen::Array< T, Eigen::Dynamic, Eigen::Dynamic> ret = reconstructAll(tabOwnedByProcess, p_gridOnProc).transpose();
        if (ret.size())
            return ret;
        else
            return Eigen::Array<T, Eigen::Dynamic, 1>();
    }
};
}

#endif /* PARALLELCOMPUTEGRIDSPLITING_H */
