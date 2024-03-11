
#ifndef SIMULATESTEPSWITCH_H
#define SIMULATESTEPSWITCH_H
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <vector>
#include <boost/lexical_cast.hpp>
#include "geners/BinaryFileArchive.hh"
#include "libflow/core/grids/RegularSpaceIntGrid.h"
#include "libflow/core/utils/StateWithIntState.h"
#include "libflow/regression/BaseRegression.h"
#include "libflow/dp/OptimizerSwitchBase.h"

/** SimulateStepSwitch.h
 *  In simulation part, permits to  use  continuation values  basis function stored in in optimization step and
 *  calculate optimal control
 *  \author Xavier Warin
 */

namespace libflow
{
/// \class SimulateStepSwitch SimulateStepSwitch.h
/// One step in forward simulation using mpi or not
class SimulateStepSwitch
{
private :

    std::vector< std::shared_ptr<RegularSpaceIntGrid> >  m_pGridFollowing ; ///< global grid at following time step for each regime
    std::shared_ptr<OptimizerSwitchBase >          m_pOptimize ; ///< optimizer solving the problem for one point and one step
    std::vector< Eigen::ArrayXXd >  m_basisFunc ; ///<  Basis function  per regime of continuation value at each point of the grid    at following step
    std::shared_ptr<BaseRegression>  m_regressor ; ///< Regressor used
#ifdef USE_MPI
    boost::mpi::communicator  m_world; ///< Mpi communicator
#endif

public :

    /// \brief default
    SimulateStepSwitch() {}
    virtual ~SimulateStepSwitch() {}

    /// \brief Constructor
    /// \param p_ar               Archive where continuation values are stored
    /// \param p_iStep            Step number identifier
    /// \param p_nameCont         Name use to store continuation valuation
    /// \param p_pGridFollowing   grid at following time step for each regime
    /// \param p_pOptimize        Optimize object defining the transition step
    /// \param p_world            MPI communicator
    SimulateStepSwitch(gs::BinaryFileArchive &p_ar,  const int &p_iStep,  const std::string &p_nameCont,
                       const   std::vector< std::shared_ptr<RegularSpaceIntGrid> > &p_pGridFollowing,
                       const  std::shared_ptr<OptimizerSwitchBase > &p_pOptimize
#ifdef USE_MPI
                       , const boost::mpi::communicator &p_world
#endif
                      );

    /// \brief Define one step arbitraging between possible commands
    /// \param p_statevector    Vector of states (regime, deterministic integer, uncertainty)
    /// \param p_phiInOut       actual contract value modified at current time step by applying an optimal command (size number of functions by number of simulations)
    void oneStep(std::vector<StateWithIntState > &p_statevector, Eigen::ArrayXXd  &p_phiInOut) const;

};
}
#endif /* SIMULATESTEPSWITCH_H */
