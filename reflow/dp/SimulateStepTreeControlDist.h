
#ifndef SIMULATESTEPTREECONTROLDIST_H
#define SIMULATESTEPTREECONTROLDIST_H
#include <boost/lexical_cast.hpp>
#include "geners/BinaryFileArchive.hh"
#include "reflow/dp/SimulateStepTreeBase.h"
#include "reflow/core/grids/FullGrid.h"
#include "reflow/tree/StateTreeStocks.h"
#include "reflow/tree/GridTreeValue.h"
#include "reflow/tree/GridTreeValueGeners.h"
#include "reflow/core/parallelism/ParallelComputeGridSplitting.h"
#include "reflow/dp/OptimizerDPTreeBase.h"

/** SimulateStepTreeControlDist.h
 *  In simulation part, permits to  use  continuation values stored in in optimization step and
 *  calculate optimal control when uncertainties belong to a tree
 */

namespace reflow
{
/// \class SimulateStepTreeControlDist SimulateStepTreeControlDist.h
/// One step in forward simulation using mpi for  uncertainties belonging to a tree (discrete values)
class SimulateStepTreeControlDist : public SimulateStepTreeBase
{
private :

    std::shared_ptr<FullGrid>  m_pGridCurrent ; ///< global grid at current time step
    std::shared_ptr<FullGrid>  m_pGridFollowing ; ///< global grid at following time step
    std::shared_ptr<OptimizerDPTreeBase >          m_pOptimize ; ///< optimizer solving the problem for one point and one step
    std::vector< GridTreeValue >  m_control ; ///< to store optimal control at the current step

    std::vector< std::shared_ptr<Eigen::ArrayXXd > > m_contValue ; ///< to store control values split on current processor
    bool m_bOneFile ; /// do we use one file for continuation values
    std::shared_ptr<ParallelComputeGridSplitting>  m_parall  ; ///< parallel object for splitting and reconstruction
    boost::mpi::communicator  m_world; ///< Mpi communicator

public :

    /// \brief default
    SimulateStepTreeControlDist() {}
    virtual ~SimulateStepTreeControlDist() {}

    /// \brief Constructor
    /// \param p_ar               Archive where continuation values are stored
    /// \param p_iStep            Step number identifier
    /// \param p_nameCont         Name use to store continuation valuation
    /// \param p_pGridCurrent     grid at current  time step
    /// \param p_pGridFollowing   grid at following time step
    /// \param p_pOptimize        Optimize object defining the transition step
    /// \param p_bOneFile              do we stoxre continuation values  in only one file
    /// \param p_world            MPI communicator
    SimulateStepTreeControlDist(gs::BinaryFileArchive &p_ar,  const int &p_iStep,  const std::string &p_nameCont,
                                const   std::shared_ptr<FullGrid> &p_pGridCurrent,
                                const   std::shared_ptr<FullGrid> &p_pGridFollowing,
                                const  std::shared_ptr<OptimizerDPTreeBase > &p_pOptimize,
                                const bool &p_bOneFile, const boost::mpi::communicator &p_world);

    /// \brief Define one step arbitraging between possible commands
    /// \param p_statevector    Vector of states (regime, stock descriptor, uncertainty)
    /// \param p_phiInOut       actual contract value modified at current time step by applying an optimal command
    void oneStep(std::vector<StateTreeStocks > &p_statevector, Eigen::ArrayXXd  &p_phiInOut) const;


};
}
#endif /* SIMULATESTEPTREECONTROLDIST_H */
