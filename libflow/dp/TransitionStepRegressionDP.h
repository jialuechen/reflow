// Copyright (C) 2016 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#ifndef TRANSITIONSTEPREGRESSIONDP_H
#define TRANSITIONSTEPREGRESSIONDP_H
#ifdef OMP
#include <omp.h>
#endif
#include <memory>
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "libflow/dp/TransitionStepRegressionBase.h"
#include "libflow/core/grids/FullGrid.h"
#include "libflow/regression/BaseRegression.h"
#include "libflow/dp/OptimizerDPBase.h"

/** \file TransitionStepRegressionDP.h
 * \brief Solve the dynamic programming  problem on one time step by regression with multi thread and mpi without distribution of the data
 * \author Xavier Warin
  */


namespace libflow
{

/// \class TransitionStepRegressionDP TransitionStepRegressionDP.h
///        One step of dynamic programming without using mpi
class TransitionStepRegressionDP  :  public TransitionStepRegressionBase
{
private :


    std::shared_ptr<FullGrid>  m_pGridCurrent ; ///< global grid at current time step
    std::shared_ptr<FullGrid>  m_pGridPrevious ; ///< global grid at previous time step
    std::shared_ptr<OptimizerDPBase  >  m_pOptimize ; ///< optimizer solving the problem for one point and one step
#ifdef USE_MPI
    boost::mpi::communicator  m_world; ///< Mpi communicator
#endif

public :

    /// \brief default
    TransitionStepRegressionDP() {}
    virtual ~TransitionStepRegressionDP() {}

    /// \brief Constructor
    TransitionStepRegressionDP(const  std::shared_ptr<FullGrid> &p_pGridCurrent,
                               const  std::shared_ptr<FullGrid> &p_pGridPrevious,
                               const  std::shared_ptr<OptimizerDPBase > &p_pOptimize
#ifdef USE_MPI
                               , const boost::mpi::communicator &p_world
#endif
                              );


    /// \brief One step for dynamic programming in optimization
    /// \param p_phiIn      for each regime the function value ( nb simulation, nb stocks )
    /// \param p_condExp    Conditional expectation object
    /// \return     solution obtained after one step of dynamic programming and the optimal control
    std::pair< std::vector< std::shared_ptr< Eigen::ArrayXXd > >, std::vector<  std::shared_ptr<  Eigen::ArrayXXd > > >  oneStep(const std::vector< std::shared_ptr< Eigen::ArrayXXd > > &p_phiIn,
            const std::shared_ptr< BaseRegression>     &p_condExp) const ;


    /// \brief Permits to dump continuation values on archive
    /// \param p_ar                   archive to dump in
    /// \param p_name                 name used for object
    /// \param p_iStep                 Step number or identifier for time step
    /// \param p_phiIn                for each regime the function value ( nb simulation ,nb stocks)
    /// \param p_control              Optimal control ( nb simulation ,nb stocks) for each control
    /// \param p_condExp               conditional expectation operator
    void dumpContinuationValues(std::shared_ptr<gs::BinaryFileArchive> p_ar, const std::string &p_name, const int &p_iStep, const std::vector< std::shared_ptr< Eigen::ArrayXXd > > &p_phiIn,
                                const std::vector< std::shared_ptr< Eigen::ArrayXXd > > &p_control, const  std::shared_ptr<BaseRegression>    &p_condExp) const;


    /// \brief Permits to dump Bellman values on archive
    /// \param p_ar                   archive to dump in
    /// \param p_name                 name used for object
    /// \param p_iStep                 Step number or identifier for time step
    /// \param p_phiIn                for each regime the function value ( nb simulation ,nb stocks)
    /// \param p_condExp               conditional expectation operator
    void dumpBellmanValues(std::shared_ptr<gs::BinaryFileArchive> p_ar, const std::string &p_name, const int &p_iStep, const std::vector< std::shared_ptr< Eigen::ArrayXXd > > &p_phiIn,
                           const  std::shared_ptr<BaseRegression>    &p_condExp) const;
};
}
#endif /* TRANSITIONSTEPREGRESSIONDP_H */

