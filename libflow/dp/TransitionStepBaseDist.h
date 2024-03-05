
#ifndef TRANSITIONSTEPBASEDIST_H
#define TRANSITIONSTEPBASEDIST_H
#include <functional>
#include <memory>
#include <boost/mpi.hpp>
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "geners/Record.hh"
#include "libflow/dp/TransitionStepBase.h"
#include "libflow/core/parallelism/ParallelComputeGridSplitting.h"
#include "libflow/core/grids/FullGrid.h"
#include "libflow/regression/GridAndRegressedValue.h"
#include "libflow/dp/OptimizerBase.h"

/** \file TransitionStepBaseDist.h
 * \brief Solve the dynamic programming  problem on one time step by  with multi thread and mpi with distribution of the data
 *         without using regression method
 * \author Xavier Warin
 */
namespace libflow
{
/// \class TransitionStepBaseDist TransitionStepBaseDist.h
///        One step of dynamic programming using MPI
class TransitionStepBaseDist
{
protected :

    std::shared_ptr<FullGrid>  m_pGridCurrent ; ///< global grid at current time step
    std::shared_ptr<FullGrid> m_pGridPrevious ; ///< global grid at previous time step
    std::shared_ptr<OptimizerBase >        m_pOptimize ; ///< optimizer solving the problem for one point and one step
    std::shared_ptr<ParallelComputeGridSplitting> m_paral ; ///< parallel object
    std::shared_ptr<FullGrid>   m_gridCurrentProc ; ///< local grid  treated by the processor
    std::shared_ptr<FullGrid>   m_gridExtendPreviousStep; ///< give extended grid at previous step
    boost::mpi::communicator  m_world; ///< Mpi communicator

public :

    /// \brief default
    TransitionStepBaseDist() {}
    virtual ~TransitionStepBaseDist() {}

    /// \brief Constructor
    /// \param p_pGridCurrent  grid (stock points) at the current time step
    /// \param p_pridPrevious  grid (stock points) at the previusly treated time step
    /// \param p_pOptimize           optimizer object to optimizer the problem on one time step
    /// \param p_world               MPI communicator
    TransitionStepBaseDist(const  std::shared_ptr<FullGrid> &p_pGridCurrent,
                           const  std::shared_ptr<FullGrid> &p_pridPrevious,
                           const  std::shared_ptr<OptimizerBase> &p_pOptimize,
                           const boost::mpi::communicator &p_world);

    /// \brief get back local grid (to processor) associated to current step
    inline std::shared_ptr<FullGrid >   getGridCurrentProc()const
    {
        return m_gridCurrentProc ;
    }

    /// \brief Reconstruct on processor 0 on the current grid
    /// \param p_phiIn data owned by current processor
    /// \param p_phiOut data reconstructed
    void reconstructOnProc0(const std::vector< std::shared_ptr< Eigen::ArrayXXd > > &p_phiIn, std::vector< std::shared_ptr< Eigen::ArrayXXd > > &p_phiOut);
};
}
#endif /* TRANSITIONSTEPBASEDIST_H */

