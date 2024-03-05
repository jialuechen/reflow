
#ifndef TRANSITIONSTEPSEMILAGRANG_H
#define TRANSITIONSTEPSEMILAGRANG_H
#ifdef OMP
#include <omp.h>
#endif
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <functional>
#include <memory>
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "libflow/semilagrangien/TransitionStepSemilagrangBase.h"
#include "libflow/core/grids/SpaceGrid.h"
#include "libflow/core/grids/InterpolatorSpectral.h"
#include "libflow/semilagrangien/OptimizerSLBase.h"

/** \file TransitionStepSemilagrang.h
 * \brief Solve one step  of explicit semi Lagrangian scheme
 * \author Xavier Warin
 */


namespace libflow
{

/// \class TransitionStepSemilagrang TransitionStepSemilagrang.h
///        One step  of semi Lagrangian scheme
class TransitionStepSemilagrang : public TransitionStepSemilagrangBase
{
private :

    std::shared_ptr<SpaceGrid>  m_gridCurrent ; ///< global grid at current time step
    std::shared_ptr<SpaceGrid>  m_gridPrevious ; ///< global grid at previous time step
    std::shared_ptr<OptimizerSLBase  >  m_optimize ; ///< optimizer solving the problem for one point and one step
#ifdef USE_MPI
    boost::mpi::communicator  m_world; ///< Mpi communicator
#endif

public :

    /// \brief Constructor
    TransitionStepSemilagrang(const  std::shared_ptr<SpaceGrid> &p_gridCurrent,
                              const  std::shared_ptr<SpaceGrid> &p_gridPrevious,
                              const  std::shared_ptr<OptimizerSLBase > &p_optimize
#ifdef USE_MPI
                              , const boost::mpi::communicator &p_world
#endif
                             );

    /// \brief One time step for resolution
    /// \param p_phiIn         for each regime the function value ( on the grid)
    /// \param p_time          current date
    /// \param p_boundaryFunc  Function at the boundary to impose Dirichlet conditions  (depending on regime and position)
    /// \return     solution obtained after one step of dynamic programming and the optimal control
    std::pair< std::vector< std::shared_ptr< Eigen::ArrayXd > >, std::vector< std::shared_ptr< Eigen::ArrayXd >  >  > oneStep(const std::vector< std::shared_ptr< Eigen::ArrayXd > > &p_phiIn, const double &p_time,   const std::function<double(const int &, const Eigen::ArrayXd &)> &p_boundaryFunc) const;

    /// \brief Permits to dump continuation values on archive
    /// \param p_ar                   archive to dump in
    /// \param p_name                 name used for object
    /// \param  p_iStep               Step number or identifier for time step
    /// \param p_phiIn                for each regime the function value
    /// \param p_control              for each control, the optimal value
    void dumpValues(std::shared_ptr<gs::BinaryFileArchive> p_ar, const std::string &p_name, const int &p_iStep, const std::vector< std::shared_ptr< Eigen::ArrayXd > > &p_phiIn,
                    const std::vector< std::shared_ptr< Eigen::ArrayXd > > &p_control) const;
};
}
#endif /* TRANSITIONSTEPSEMILAGRANG_H */

