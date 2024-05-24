
#ifndef SIMULATESTEPREGRESSION_H
#define SIMULATESTEPREGRESSION_H
#ifdef OMP
#include <omp.h>
#endif
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <boost/lexical_cast.hpp>
#include "geners/BinaryFileArchive.hh"
#include "geners/Reference.hh"
#include "reflow/dp/SimulateStepBase.h"
#include "reflow/core/utils/StateWithStocks.h"
#include "reflow/core/grids/SpaceGrid.h"
#include "reflow/dp/OptimizerDPBase.h"
#include "reflow/regression/GridAndRegressedValueGeners.h"

/** \file SimulateStepRegression.h
 *  \brief  In simulation part, permits to  use  continuation values stored in in optimization step and
 *           calculate optimal control
 *  \author Xavier Warin
 */
namespace reflow
{

/// \class SimulateStepRegression SimulateStepRegression.h
/// One step in forward simulation
class SimulateStepRegression : public SimulateStepBase
{
private :

    std::shared_ptr<SpaceGrid>  m_pGridFollowing ; ///< global grid at following time step
    std::shared_ptr<reflow::OptimizerDPBase >          m_pOptimize ; ///< optimizer solving the problem for one point and one step
    std::vector< GridAndRegressedValue >  m_continuationObj ; ///< to store continuation value per regime  on the grid at following step
#ifdef USE_MPI
    boost::mpi::communicator  m_world; ///< Mpi communicator
#endif

public :

    /// \brief default
    SimulateStepRegression() {}
    virtual ~SimulateStepRegression() {}

    /// \brief Constructor
    /// \param p_ar               Archive where continuation values are stored
    /// \param p_iStep            Step number indentifier
    /// \param p_nameCont         Name use to store conuation valuation
    /// \param p_pGridFollowing   grid at following time step
    /// \param p_pOptimize        Optimize object defining the transition step
    /// \param p_world            MPI communicator
    SimulateStepRegression(gs::BinaryFileArchive &p_ar,  const int &p_iStep,  const std::string &p_nameCont,
                           const   std::shared_ptr<SpaceGrid> &p_pGridFollowing, const  std::shared_ptr<reflow::OptimizerDPBase > &p_pOptimize
#ifdef USE_MPI
                           , const boost::mpi::communicator &p_world
#endif
                          );

    /// \brief Define one step arbitraging between possibhle commands
    /// \param p_statevector    Vector of states (regime, stock descritor, uncertainty)
    /// \param p_phiInOut       actual contract values modified at current time step by applying an optimal command (number of function by numver of simulations)
    void oneStep(std::vector<StateWithStocks > &p_statevector, Eigen::ArrayXXd  &p_phiInOut) const ;

};
}
#endif /* SIMULATESTEPREGRESSION_H */
