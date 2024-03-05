
#ifndef SIMULATESTEPSEMILAGRANGCONTROLDIST_H
#define SIMULATESTEPSEMILAGRANGCONTROLDIST_H
#ifdef OMP
#include <omp.h>
#endif
#include <boost/mpi.hpp>
#include <boost/lexical_cast.hpp>
#include "geners/BinaryFileArchive.hh"
#include "libflow/semilagrangien/SimulateStepSemilagrangBase.h"
#include "libflow/semilagrangien/OptimizerSLBase.h"
#include "libflow/core/grids/FullGrid.h"
#include "libflow/core/parallelism/ParallelComputeGridSplitting.h"

/** \file SimulateStepSemilagrangControlDist.h
 *  \brief  In simulation part, permits to  use  the PDE function  value to
 *          implement an optimal strategy
 *  \author Xavier Warin
 */
namespace libflow
{

/// \class SimulateStepSemilagrangControlDist SimulateStepSemilagrangControlDist.h
/// One step in forward simulation but with MPI, data can be distributed on processors
class SimulateStepSemilagrangControlDist : public SimulateStepSemilagrangBase
{
private :

    std::shared_ptr<FullGrid> m_gridCur ; ///< grid at the current time step
    std::shared_ptr<FullGrid> m_gridNext ; ///< grid at the next time step
    std::vector<std::shared_ptr< Eigen::ArrayXd  > > m_control ; ///< Optimal control for each control
    std::shared_ptr<libflow::OptimizerSLBase >  m_pOptimize ; ///< optimizer solving the problem for one point and one step
    bool m_bOneFile ; /// do we use one file for continuation values
    std::shared_ptr<ParallelComputeGridSplitting>  m_parall  ; ///< parallel object for splitting and reconstruction
    boost::mpi::communicator  m_world ; ///< MPI communicator

public :

    /// \brief Constructor
    /// \param p_ar               Archive where continuation values are stored
    /// \param p_iStep            Step number identifier
    /// \param p_name             Name use to store continuation valuation
    /// \param p_gridCur          grid at the current time step
    /// \param p_gridNext         grid at the next time step
    /// \param p_pOptimize        Optimize object defining the transition step
    /// \param p_bOneFile              do we store continuation values  in only one file
    /// \param p_world            MPI communicator
    SimulateStepSemilagrangControlDist(gs::BinaryFileArchive &p_ar,  const int &p_iStep,  const std::string &p_name,
                                       const std::shared_ptr<FullGrid>   &p_gridCur,
                                       const std::shared_ptr<FullGrid> &p_gridNext,
                                       const std::shared_ptr<OptimizerSLBase > &p_pOptimize,
                                       const bool &p_bOneFile,
                                       const boost::mpi::communicator &p_world);

    /// \brief Define one step arbitraging between possible commands
    /// \param p_gaussian       2 dimensional Gaussian array (size : number of Brownians motion by number of simulations)
    /// \param p_statevector    Vector of states for each simulation (size of the state by the number of simulations) in the current regime
    /// \param p_iReg           regime number for each simulation
    /// \param p_phiInOut       actual contracts values modified at current time step by applying an optimal command
    void oneStep(const Eigen::ArrayXXd   &p_gaussian, Eigen::ArrayXXd &p_statevector, Eigen::ArrayXi   &p_iReg, Eigen::ArrayXXd  &p_phiInOut) const;

};
}
#endif /* SIMULATESTEPSEMILAGRANGCONTROLDIST_H */
