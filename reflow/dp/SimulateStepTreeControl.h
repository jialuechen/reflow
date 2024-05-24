
#ifndef SIMULATESTEPTREECONTROL_H
#define SIMULATESTEPTREECONTROL_H
#ifdef OMP
#include <omp.h>
#endif
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <boost/lexical_cast.hpp>
#include "geners/BinaryFileArchive.hh"
#include "geners/Reference.hh"
#include "reflow/dp/SimulateStepTreeBase.h"
#include "reflow/tree/StateTreeStocks.h"
#include "reflow/core/grids/SpaceGrid.h"
#include "reflow/dp/OptimizerDPTreeBase.h"
#include "reflow/tree/GridTreeValue.h"
#include "reflow/tree/GridTreeValueGeners.h"

/** \file SimulateStepTreeControl.h
 *  \brief  In simulation part, permits to  use the optimal control stored in in optimization step to calculate  Monte Carlo optimal trajectory
 *          using Tree method
 *  \author Xavier Warin
 */
namespace reflow
{

/// \class SimulateStepTreeControl SimulateStepTreeControl.h
/// One step in forward simulation, uncertainties belong to a tree.
class SimulateStepTreeControl : public SimulateStepTreeBase
{
private :

    std::shared_ptr<SpaceGrid>  m_pGridFollowing ; ///< global grid at following time step
    std::shared_ptr<reflow::OptimizerDPTreeBase >          m_pOptimize ; ///< optimizer solving the problem for one point and one step
    std::vector< GridTreeValue >  m_control ; ///< to store the optimal control calculated in optimization
#ifdef USE_MPI
    boost::mpi::communicator  m_world; ///< Mpi communicator
#endif

public :

    /// \brief default
    SimulateStepTreeControl() {}
    virtual ~SimulateStepTreeControl() {}

    /// \brief Constructor
    /// \param p_ar               Archive where continuation values are stored
    /// \param p_iStep            Step number identifier
    /// \param p_nameCont         Name use to store continuation valuation
    /// \param p_pGridFollowing   grid at following time step
    /// \param p_pOptimize        Optimize object defining the transition step
    /// \param p_world            MPI communicator
    SimulateStepTreeControl(gs::BinaryFileArchive &p_ar,  const int &p_iStep,  const std::string &p_nameCont,
                            const   std::shared_ptr<SpaceGrid> &p_pGridFollowing,
                            const  std::shared_ptr<reflow::OptimizerDPTreeBase > &p_pOptimize
#ifdef USE_MPI
                            , const boost::mpi::communicator &p_world
#endif
                           );

    /// \brief Define one step arbitraging between possible commands
    /// \param p_statevector    Vector of states (regime, stock descriptor, uncertainty)
    /// \param p_phiInOut       actual contract value modified at current time step by applying an optimal command (size : number of function to follow by number of simulations)
    void oneStep(std::vector<StateTreeStocks > &p_statevector, Eigen::ArrayXXd  &p_phiInOut) const;

};
}
#endif /* SIMULATESTEPTREECONTROL_H */
