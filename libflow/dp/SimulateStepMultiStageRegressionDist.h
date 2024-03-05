// Copyright (C) 2023 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#ifndef SIMULATESTEPMULTISTAGEREGRESSIONDIST_H
#define SIMULATESTEPMULTISTAGEREGRESSIONDIST_H
#include <vector>
#include <array>
#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/lexical_cast.hpp>
#include "geners/BinaryFileArchive.hh"
#include "libflow/core/grids/FullGrid.h"
#include "libflow/core/utils/StateWithStocks.h"
#include "libflow/core/utils/types.h"
#include "libflow/regression/GridAndRegressedValueGeners.h"
#include "libflow/core/parallelism/ParallelComputeGridSplitting.h"
#include "libflow/dp/OptimizerMultiStageDPBase.h"

/** SimulateStepMultiStageRegressionDist.h
 *  In simulation part, permits to  use  continuation values stored in in optimization step and
 *  calculate optimal control
 *  \author Xavier Warin
 */

namespace libflow
{
/// \class SimulateStepMultiStageRegressionDist SimulateStepMultiStageRegressionDist.h
/// One step in forward simulation using mpi
class SimulateStepMultiStageRegressionDist
{
private :

    std::shared_ptr<FullGrid>  m_pGridCurrent ; ///< global grid at current  time step
    std::shared_ptr<FullGrid>  m_pGridFollowing ; ///< global grid at following time step
    std::shared_ptr<OptimizerMultiStageDPBase >          m_pOptimize ; ///< optimizer solving the problem for one point and one step
    std::shared_ptr<gs::BinaryFileArchive> m_ar ; ///< pointer on the archive
    int m_iStep ; ///< time step number
    std::string m_nameCont ; ///< name for stochastic bellman value
    std::string m_nameDetCont; ///< name for deterministic bellman values
    bool m_bOneFile ; /// do we use one file for continuation values
    std::shared_ptr<ParallelComputeGridSplitting>  m_parall  ; ///< parallel object for splitting and reconstruction
    std::shared_ptr<ParallelComputeGridSplitting>  m_parallDet  ; ///< parallel object for splitting and reconstruction for deterministic optimization
    boost::mpi::communicator  m_world; ///< Mpi communicator

    /// \brief function to read continuation values in archive
    /// \param p_name         key in archive
    /// \param p_stepString   secondary key in archive
    std::vector< GridAndRegressedValue> readContinuationInArchive(const std::string &p_name, const std::string &p_stepString);

    /// \brief read regressed value and regressor in archive   (Bellman values split into multiple files)
    /// \param p_name  key in archive
    /// \param p_stepString   secondary key in archive
    std::pair< std::shared_ptr<BaseRegression>, std::vector< Eigen::ArrayXXd > >  readRegressedValues(const std::string &p_name, const std::string &p_stepString) ;

    /// \brief calculate extended grids coordinates number  for parallelism when Bellman value are dispatched in memory
    /// \param p_gridFollow      grid at end of period
    /// \param p_regionByProcessor fr each processor define the domain treated (min , max in each dimension)
    SubMeshIntCoord  calculateSubMeshExtended(const std::shared_ptr<FullGrid> &p_gridFollow,  const std::vector<  std::array< double, 2>  >   &p_regionByProcessor) const;

    /// \brief Affect each particle to a calculation mesh by grouping particles with similar space position
    /// \param p_statevector   the state vectors of the particles
    /// \param p_gridFollow    grid at the follwoing period
    /// \return an affectation for each particle  to a  processor, and the region assiociated to each process (min and max of the cell)
    std::pair< std::vector<int>, std::vector<  std::array< double, 2>  > > splitParticleOnProcessor(const std::vector<StateWithStocks >    &p_statevector, const std::shared_ptr<FullGrid> &p_gridFollow) const;

public :

    /// \brief default
    SimulateStepMultiStageRegressionDist() {}
    virtual ~SimulateStepMultiStageRegressionDist() {}

    /// \brief Constructor
    /// \param p_ar               Archive where continuation values are stored
    /// \param p_iStep            Step number identifier
    /// \param p_nameCont         Name use to store continuation valuation
    /// \param p_nameDetCont      Name use to store continution value in deterministic part
    /// \param p_pGridCurrent     grid at current time step
    /// \param p_pGridFollowing   grid at following time step
    /// \param p_pOptimize        Optimize object defining the transition step
    /// \param p_bOneFile         do we store continuation values  in only one file
    /// \param p_world            MPI communicator
    SimulateStepMultiStageRegressionDist(const std::shared_ptr<gs::BinaryFileArchive> &p_ar,
                                         const int &p_iStep,  const std::string &p_nameCont,
                                         const std::string &p_nameDetCont,
                                         const  std::shared_ptr<FullGrid> &p_pGridCurrent,
                                         const  std::shared_ptr<FullGrid> &p_pGridFollowing,
                                         const  std::shared_ptr<OptimizerMultiStageDPBase > &p_pOptimize,
                                         const bool &p_bOneFile, const boost::mpi::communicator &p_world);

    /// \brief Define one step arbitraging between possible commands
    /// \param p_statevector    Vector of states (regime, stock descriptor, uncertainty)
    /// \param p_phiInOut       actual contract value modified at current time step by applying an optimal command (size number of functions by number of simulations), one by period (size of the vector)
    void oneStep(std::vector<StateWithStocks > &p_statevector, std::vector<Eigen::ArrayXXd>  &p_phiInOut) ;

};
}
#endif /* SIMULATESTEPMULTISTAGREGRESSIONDIST_H */
