
#ifndef SIMULATESTEPREGRESSIONCONTROL_H
#define SIMULATESTEPREGRESSIONCONTROL_H
#ifdef OMP
#include <omp.h>
#endif
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <boost/lexical_cast.hpp>
#include "geners/BinaryFileArchive.hh"
#include "geners/Reference.hh"
#include "libflow/dp/SimulateStepBase.h"
#include "libflow/core/utils/StateWithStocks.h"
#include "libflow/core/grids/SpaceGrid.h"
#include "libflow/dp/OptimizerBaseInterp.h"
#include "libflow/regression/GridAndRegressedValue.h"
#include "libflow/regression/GridAndRegressedValueGeners.h"

/** \file SimulateStepRegressionControl.h
 *  \brief  In simulation part, permits to  use the optimal control stored in in optimization step to calculate de Monte Carlo optimal trajectory
 *  \author Xavier Warin
 */
namespace libflow
{

/// \class SimulateStepRegressionControl SimulateStepRegressionControl.h
/// One step in forward simulation
class SimulateStepRegressionControl : public SimulateStepBase
{
private :

    std::shared_ptr<SpaceGrid>  m_pGridFollowing ; ///< global grid at following time step
    std::shared_ptr<libflow::OptimizerBaseInterp >          m_pOptimize ; ///< optimizer solving the problem for one point and one step
    std::vector< GridAndRegressedValue >  m_control ; ///< to store the optimal control calculated in optimization
#ifdef USE_MPI
    boost::mpi::communicator  m_world; ///< Mpi communicator
#endif

public :

    /// \brief default
    SimulateStepRegressionControl() {}
    virtual ~SimulateStepRegressionControl() {}

    /// \brief Constructor
    /// \param p_ar               Archive where continuation values are stored
    /// \param p_iStep            Step number identifier
    /// \param p_nameCont         Name use to store continuation valuation
    /// \param p_pGridFollowing   grid at following time step
    /// \param p_pOptimize        Optimize object defining the transition step
    /// \param p_world            MPI communicator
    SimulateStepRegressionControl(gs::BinaryFileArchive &p_ar,  const int &p_iStep,  const std::string &p_nameCont,
                                  const   std::shared_ptr<SpaceGrid> &p_pGridFollowing,
                                  const  std::shared_ptr<libflow::OptimizerBaseInterp > &p_pOptimize
#ifdef USE_MPI
                                  , const boost::mpi::communicator &p_world
#endif
                                 );

    /// \brief Define one step arbitraging between possible commands
    /// \param p_statevector    Vector of states (regime, stock descriptor, uncertainty)
    /// \param p_phiInOut       actual contract value modified at current time step by applying an optimal command (size : number of function to follow by number of simulations)
    void oneStep(std::vector<StateWithStocks > &p_statevector, Eigen::ArrayXXd  &p_phiInOut) const;

};
}
#endif /* SIMULATESTEPREGRESSIONCONTROL_H */
