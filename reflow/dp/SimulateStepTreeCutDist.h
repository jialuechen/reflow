
#ifndef SIMULATESTEPTREECUTDIST_H
#define SIMULATESTEPTREECUTDIST_H
#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/lexical_cast.hpp>
#include "geners/BinaryFileArchive.hh"
#include "reflow/dp/SimulateStepTreeBase.h"
#include "reflow/core/grids/FullGrid.h"
#include "reflow/tree/StateTreeStocks.h"
#include "reflow/tree/ContinuationCutsTree.h"
#include "reflow/core/parallelism/ParallelComputeGridSplitting.h"
#include "reflow/dp/OptimizerDPCutTreeBase.h"

/** SimulateStepTreeCutDist.h
 *  In simulation part, permits to  use  continuation cuts values stored in in optimization step and
 *  calculate optimal controls using a LP programm to calculate transition problems. Tree version with MPI
 *  \author Xavier Warin
 */

namespace reflow
{
/// \class SimulateStepTreeCutDist SimulateStepTreeCutDist.h
/// One step in forward simulation using mpi
class SimulateStepTreeCutDist : public SimulateStepTreeBase
{
private :

    std::shared_ptr<FullGrid>  m_pGridFollowing ; ///< global grid at following time step
    std::shared_ptr<OptimizerDPCutTreeBase >          m_pOptimize ; ///< optimizer solving the problem for one point and one step
    std::vector< ContinuationCutsTree>  m_continuationObj ; ///< to store continuation value per regime  on the grid at following step
    std::vector< Eigen::ArrayXXd  > m_contValue ; ///< to store continuation cuts values split in memory if multiple files used
    bool m_bOneFile ; /// do we use one file for continuation values
    std::shared_ptr<ParallelComputeGridSplitting>  m_parall  ; ///< parallel object for splitting and reconstruction
    boost::mpi::communicator  m_world; ///< Mpi communicator

public :

    /// \brief Constructor
    /// \param p_ar               Archive where continuation values are stored
    /// \param p_iStep            Step number identifier
    /// \param p_nameCont         Name use to store continuation valuation
    /// \param p_pGridFollowing   grid at following time step
    /// \param p_pOptimize        Optimize object defining the transition step
    /// \param p_bOneFile         Do we store continuation values  in only one file
    /// \param p_world            MPI communicator
    SimulateStepTreeCutDist(gs::BinaryFileArchive &p_ar,  const int &p_iStep,  const std::string &p_nameCont,
                            const  std::shared_ptr<FullGrid> &p_pGridFollowing,
                            const  std::shared_ptr<OptimizerDPCutTreeBase > &p_pOptimize,
                            const bool &p_bOneFile, const boost::mpi::communicator &p_world);

    /// \brief Define one step arbitraging between possible commands
    /// \param p_statevector    Vector of states (regime, stock descriptor, uncertainty node number)
    /// \param p_phiInOut       actual contract value modified at current time step by applying an optimal command (size number of functions by number of simulations)
    void oneStep(std::vector<StateTreeStocks > &p_statevector, Eigen::ArrayXXd  &p_phiInOut) const;

};
}
#endif /* SIMULATESTEPTREECUTDIST_H */
