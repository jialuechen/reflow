
#ifndef SIMULATESTEPSEMILAGRANG_H
#define SIMULATESTEPSEMILAGRANG_H
#ifdef OMP
#include <omp.h>
#endif
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <boost/lexical_cast.hpp>
#include "geners/BinaryFileArchive.hh"
#include "geners/Reference.hh"
#include "reflow/semilagrangien/SimulateStepSemilagrangBase.h"
#include "reflow/semilagrangien/OptimizerSLBase.h"
#include "reflow/core/grids/FullGrid.h"
#include "reflow/core/grids/InterpolatorSpectral.h"
#include "reflow/semilagrangien/SemiLagrangEspCond.h"

/** \file SimulateStepSemilagrang.h
 *  \brief  In simulation part, permits to  use  the PDE function  value to
 *          implement an optimal strategy
 *  \author Xavier Warin
 */
namespace reflow
{

/// \class SimulateStepSemilagrang SimulateStepSemilagrang.h
/// One step in forward simulation
class SimulateStepSemilagrang : public SimulateStepSemilagrangBase
{
private :

    std::shared_ptr<SpaceGrid> m_gridNext ; ///< grid at the next time step
    std::vector<std::shared_ptr<InterpolatorSpectral> >  m_specInterp ; //< Spectral interpolator for regime
    std::vector<std::shared_ptr<SemiLagrangEspCond> > m_semiLag ; ///< semi Lagrangian operator in each regime
    std::shared_ptr<reflow::OptimizerSLBase >  m_pOptimize ; ///< optimizer solving the problem for one point and one step
#ifdef USE_MPI
    boost::mpi::communicator  m_world; ///< Mpi communicator
#endif

public :

    /// \brief Constructor
    /// \param p_ar               Archive where continuation values are stored
    /// \param p_iStep            Step number identifier
    /// \param p_name             Name use to store continuation valuation
    /// \param p_gridNext         grid at following time step
    /// \param p_pOptimize        Optimize object defining the transition step
    /// \param p_world              MPi communicator
    SimulateStepSemilagrang(gs::BinaryFileArchive &p_ar,  const int &p_iStep,  const std::string &p_name,  const   std::shared_ptr<SpaceGrid>   &p_gridNext,
                            const  std::shared_ptr<OptimizerSLBase > &p_pOptimize
#ifdef USE_MPI
                            , const boost::mpi::communicator &p_world
#endif

                           );

    /// \brief Define one step arbitraging between possible commands
    /// \param p_gaussian       2 dimensional Gaussian array (size : number of Brownians motion by number of simulations)
    /// \param p_statevector    Vector of states for each simulation (size of the state by the number of simulations) in the current regime
    /// \param p_iReg           regime number for each simulation
    /// \param p_phiInOut       actual contract values modified at current time step by applying an optimal command
    void oneStep(const Eigen::ArrayXXd   &p_gaussian, Eigen::ArrayXXd &p_statevector, Eigen::ArrayXi   &p_iReg, Eigen::ArrayXXd  &p_phiInOut) const;

};
}
#endif /* SIMULATESTEPSEMILAGRANG_H */
