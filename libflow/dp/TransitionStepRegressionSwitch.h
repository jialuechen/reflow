
#ifndef TRANSITIONSTEPREGRESSIONSWITCH_H
#define TRANSITIONSTEPREGRESSIONSWITCH_H
#ifdef OMP
#include <omp.h>
#endif
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <memory>
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "libflow/core/grids/RegularSpaceIntGrid.h"
#include "libflow/regression/BaseRegression.h"
#include "libflow/dp/OptimizerSwitchBase.h"

/** \file TransitionStepRegressionSwitch.h
 * \brief Solve the switching  problem  with deterministic integer states on one time step by regression with multi thread and mpi without distribution of the data
 * \author Xavier Warin
  */


namespace libflow
{

/// \class TransitionStepRegressionSwitch TransitionStepRegressionSwitch.h
///        One step of dynamic programming without using mpi
class TransitionStepRegressionSwitch
{
private :


    std::vector< std::shared_ptr<RegularSpaceIntGrid> >   m_pGridCurrent ; ///< global grid at current time step for each regime
    std::vector< std::shared_ptr<RegularSpaceIntGrid> >   m_pGridPrevious ; ///< global grid at previous time step for each regime
    std::shared_ptr<OptimizerSwitchBase  >  m_pOptimize ; ///< optimizer solving the problem for one point and one step
#ifdef USE_MPI
    boost::mpi::communicator  m_world; ///< Mpi communicator
#endif

public :

    /// \brief default
    TransitionStepRegressionSwitch() {}
    virtual ~TransitionStepRegressionSwitch() {}

    /// \brief Constructor
    TransitionStepRegressionSwitch(const  std::vector< std::shared_ptr<  RegularSpaceIntGrid> >  &p_pGridCurrent,
                                   const  std::vector< std::shared_ptr<RegularSpaceIntGrid> >  &p_pridPrevious,
                                   const  std::shared_ptr<OptimizerSwitchBase > &p_pOptimize
#ifdef USE_MPI
                                   , const boost::mpi::communicator &p_world
#endif
                                  );

    /// \brief One step for dynamic programming in optimization
    /// \param p_phiIn      for each regime the function value ( nb simulation, nb stocks )
    /// \param p_condExp    Conditional expectation object
    /// \return      solution obtained after one step of dynamic programming (  nb simulation, nb non stochastic states  )
    std::vector< std::shared_ptr< Eigen::ArrayXXd > >  oneStep(const std::vector< std::shared_ptr< Eigen::ArrayXXd > > &p_phiIn,  const std::shared_ptr< BaseRegression>     &p_condExp) const ;


    /// \brief Permits to dump continuation values on archive
    /// \param p_ar                   archive to dump in
    /// \param p_name                 name used for object
    /// \param p_iStep                 Step number or identifier for time step
    /// \param p_phiIn                for each regime the function value ( nb simulation ,nb stocks)
    /// \param p_condExp               conditional expectation operator
    void dumpContinuationValues(std::shared_ptr<gs::BinaryFileArchive> p_ar, const std::string &p_name, const int &p_iStep, const std::vector< std::shared_ptr< Eigen::ArrayXXd > > &p_phiIn,  const  std::shared_ptr<BaseRegression>    &p_condExp) const;

};
}
#endif /* TRANSITIONSTEPREGRESSIONSWITCH_H */

