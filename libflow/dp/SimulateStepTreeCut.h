
#ifndef SIMULATESTEPTREECUT_H
#define SIMULATESTEPTREECUT_H
#ifdef OMP
#include <omp.h>
#endif
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <boost/lexical_cast.hpp>
#include "geners/BinaryFileArchive.hh"
#include "geners/Reference.hh"
#include "libflow/dp/SimulateStepTreeBase.h"
#include "libflow/tree/StateTreeStocks.h"
#include "libflow/core/grids/SpaceGrid.h"
#include "libflow/dp/OptimizerDPCutTreeBase.h"
#include "libflow/tree/ContinuationCutsTree.h"

/** \file SimulateStepTreeCut.h
 *  \brief  In simulation part, permits to  use  continuation cuts values stored in in optimization step and
 *           calculate optimal control. Locally, controls are calculated by a LP so that conditional cuts are used as terminal condition. Tree version.
 *  \author Xavier Warin
 */
namespace libflow
{

/// \class SimulateStepTreeCut SimulateStepTreeCut.h
/// One step in forward simulation using trees
class SimulateStepTreeCut : public SimulateStepTreeBase
{
private :

    std::shared_ptr<SpaceGrid>  m_pGridFollowing ; ///< global grid at following time step
    std::shared_ptr<libflow::OptimizerDPCutTreeBase >          m_pOptimize ; ///< optimizer solving the problem for one point and one step
    std::vector< ContinuationCutsTree >  m_continuationObj ; ///< to store continuation cut values per regime  on the grid at following step
#ifdef USE_MPI
    boost::mpi::communicator  m_world; ///< Mpi communicator
#endif

public :

    /// \brief Constructor
    /// \param p_ar               Archive where continuation values are stored
    /// \param p_iStep            Step number indentifier
    /// \param p_nameCont         Name use to store continuation values
    /// \param p_pGridFollowing   grid at following time step
    /// \param p_pOptimize        Optimize object defining the transition step
    /// \param p_world            MPI communicator
    SimulateStepTreeCut(gs::BinaryFileArchive &p_ar,  const int &p_iStep,  const std::string &p_nameCont,
                        const   std::shared_ptr<SpaceGrid> &p_pGridFollowing, const  std::shared_ptr<libflow::OptimizerDPCutTreeBase > &p_pOptimize
#ifdef USE_MPI
                        , const boost::mpi::communicator &p_world
#endif
                       );

    /// \brief Define one step arbitraging between possible commands
    /// \param p_statevector    Vector of states (regime, stock descritor, uncertainty node number)
    /// \param p_phiInOut       actual contract values modified at current time step by applying an optimal command (number of function by numver of simulations)
    void oneStep(std::vector<StateTreeStocks > &p_statevector, Eigen::ArrayXXd  &p_phiInOut) const ;

};
}
#endif /* SIMULATESTEPTREECUT_H */
