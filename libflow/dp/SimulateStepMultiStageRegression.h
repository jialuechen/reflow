

#ifndef SIMULATESTEPMULTISTAGEREGRESSION_H
#define SIMULATESTEPMULTISTAGEREGRESSION_H
#ifdef OMP
#include <omp.h>
#endif
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <boost/lexical_cast.hpp>
#include "geners/BinaryFileArchive.hh"
#include "geners/Reference.hh"
#include "libflow/core/utils/StateWithStocks.h"
#include "libflow/core/grids/SpaceGrid.h"
#include "libflow/dp/OptimizerMultiStageDPBase.h"
#include "libflow/regression/GridAndRegressedValueGeners.h"

/** \file SimulateStepMuliStageRegression.h
 *  \brief  In simulation part, permits to  use  continuation values stored in in optimization step and
 *           calculate optimal control
 *          Multi stage version  where a deterministic optimization by dynamic programming is achieved during time transition step
 *  \author Xavier Warin
 */
namespace libflow
{

/// \class SimulateStepMultiStageRegression SimulateStepMultiStageRegression.h
/// One step in forward simulation
class SimulateStepMultiStageRegression
{
private :

    std::shared_ptr<SpaceGrid>  m_pGridFollowing ; ///< global grid at following time step
    std::shared_ptr<libflow::OptimizerMultiStageDPBase >          m_pOptimize ; ///< optimizer solving the problem for one point and one step
    std::shared_ptr<gs::BinaryFileArchive> m_ar ; ///< pointer on the archive
    int m_iStep ; ///< time step number
    std::string m_nameCont ; ///< name for stochastic bellman value
    std::string m_nameDetCont; ///< name for deterministic bellman values
#ifdef USE_MPI
    boost::mpi::communicator  m_world; ///< Mpi communicator
#endif

public :

    /// \brief default
    SimulateStepMultiStageRegression() {}
    virtual ~SimulateStepMultiStageRegression() {}

    /// \brief Constructor
    /// \param p_ar               Archive where continuation values are stored
    /// \param p_iStep            Step number indentifier
    /// \param p_nameCont         Name use to store continuation values in stochastic
    /// \param p_nameDetCont      Name use to store continution value in deterministic part
    /// \param p_pGridFollowing   grid at following time step
    /// \param p_pOptimize        Optimize object defining the transition step
    /// \param p_world            MPI communicator
    SimulateStepMultiStageRegression(std::shared_ptr<gs::BinaryFileArchive>  &p_ar,  const int &p_iStep,  const std::string &p_nameCont, const std::string &p_nameDetCont,
                                     const   std::shared_ptr<SpaceGrid> &p_pGridFollowing, const  std::shared_ptr<libflow::OptimizerMultiStageDPBase > &p_pOptimize
#ifdef USE_MPI
                                     , const boost::mpi::communicator &p_world
#endif
                                    );



    /// \brief Define one step arbitraging between possibhle commands
    /// \param p_statevector    Vector of states (regime, stock descritor, uncertainty)
    /// \param p_phiInOut       actual contract values modified at current time step by applying an optimal command (number of function by numver of simulations), one by period (size of the vector)
    void oneStep(std::vector<StateWithStocks > &p_statevector, std::vector<Eigen::ArrayXXd>   &p_phiInOut) const ;

};
}
#endif /* SIMULATESTEPMULTISTAGEREGRESSION_H */
