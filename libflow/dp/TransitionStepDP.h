
#ifndef TRANSITIONSTEPDP_H
#define TRANSITIONSTEPDP_H
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#ifdef OMP
#include <omp.h>
#endif
#include <memory>
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "libflow/dp/TransitionStepBase.h"
#include "libflow/core/grids/FullGrid.h"
#include "libflow/regression/GridAndRegressedValue.h"
#include "libflow/regression/BaseRegression.h"
#include "libflow/dp/OptimizerNoRegressionDPBase.h"

/** \file TransitionStepDP.h
 * \brief Solve the dynamic programming  problem on one time step by  with multi thread and mpi without distribution of the data
 *        but without using regression methods
 * \author Xavier Warin
  */


namespace libflow
{

/// \class TransitionStepDP TransitionStepDP.h
///        One step of dynamic programming without using mpi
class TransitionStepDP  :  public TransitionStepBase
{
private :


    std::shared_ptr<FullGrid>  m_pGridPrevious ; ///< global grid at previous time step
    std::shared_ptr<FullGrid>  m_pGridCurrent ; ///< global grid at current time step
    std::shared_ptr<BaseRegression> m_regressorPrevious; /// regressor object at the previous date
    std::shared_ptr<BaseRegression> m_regressorCurrent; /// regressor object at the current date
    std::shared_ptr<OptimizerNoRegressionDPBase  >  m_pOptimize ; ///< optimizer solving the problem for one point and one step
#ifdef USE_MPI
    boost::mpi::communicator  m_world; ///< Mpi communicator
#endif

public :

    /// \brief Constructor
    /// \param p_pGridCurrent  grid (stock points) at the current time step
    /// \param p_pridPrevious  grid (stock points) at the previusly treated time step
    /// \param p_regressorCurrent   regressor (with respect to simulation) at the current time step
    /// \param p_regressorPrevious  regressor (with respect to simulation) at the previously treated time step
    /// \param p_pOptimize           optimizer object to optimizer the problem on one time step
    TransitionStepDP(const  std::shared_ptr<FullGrid> &p_pGridCurrent,
                     const  std::shared_ptr<FullGrid> &p_pridPrevious,
                     const  std::shared_ptr<BaseRegression> &p_regressorCurrent,
                     const  std::shared_ptr<BaseRegression> &p_regressorPrevious,
                     const  std::shared_ptr<OptimizerNoRegressionDPBase> &p_pOptimize
#ifdef USE_MPI
                     , const boost::mpi::communicator &p_world
#endif
                    );

    /// \brief One step for dynamic programming in optimization
    /// \param p_phiIn      for each regime the function value at the next time step : store as (regressed function by number of  grid points)
    /// \return     solution obtained after one step of dynamic programming and the optimal control  (regressed function by the number of grid points)
    virtual std::pair< std::shared_ptr< std::vector< Eigen::ArrayXXd > >, std::shared_ptr< std::vector< Eigen::ArrayXXd  > > >   oneStep(const std::vector< Eigen::ArrayXXd  > &p_phiIn)  const ;


    /// \brief Permits to dump the  control
    /// \param p_ar                   archive to dump in
    /// \param p_name                 name used for object
    /// \param p_iStep                 Step number or identifier for time step
    /// \param p_control              Optimal control ( nb of basis functions, number of points on the grid) for each control
    void dumpValues(std::shared_ptr<gs::BinaryFileArchive> p_ar, const std::string &p_name, const int &p_iStep,
                    const std::vector<  Eigen::ArrayXXd  > &p_control) const;
};
}
#endif /* TRANSITIONSTEPDP_H */

