// Copyright (C) 2016 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#ifndef TRANSITIONSTEPREGRESSIONDPDIST_H
#define TRANSITIONSTEPREGRESSIONDPDIST_H
#include <functional>
#include <memory>
#include "boost/mpi.hpp"
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "geners/Record.hh"
#include "libflow/dp/TransitionStepRegressionBase.h"
#include "libflow/dp/TransitionStepBaseDist.h"
#include "libflow/core/grids/FullGrid.h"
#include "libflow/core/parallelism/ParallelComputeGridSplitting.h"
#include "libflow/regression/BaseRegression.h"
#include "libflow/dp/OptimizerDPBase.h"

/** \file TransitionStepRegressionDPDist.h
 * \brief Solve the dynamic programming  problem on one time step by regression with parallelization
 * \author Xavier Warin
 * \todo Developp MPI for regimes too : parallelism should be applied to stock and regimes.
 */
namespace libflow
{
/// \class TransitionStepRegressionDPDist TransitionStepRegressionDPDist.h
///        One step of dynamic programming using MPI
class TransitionStepRegressionDPDist :  public TransitionStepRegressionBase, public TransitionStepBaseDist
{

public :

    /// \brief default
    TransitionStepRegressionDPDist() {}
    virtual ~TransitionStepRegressionDPDist() {}

    /// \brief Constructor
    TransitionStepRegressionDPDist(const  std::shared_ptr<FullGrid> &p_pGridCurrent,
                                   const  std::shared_ptr<FullGrid> &p_pGridPrevious,
                                   const  std::shared_ptr<OptimizerDPBase > &p_pOptimize,
                                   const boost::mpi::communicator &p_world);


    /// \brief One step for optimization
    /// \param p_phiIn      for each regime the function value  ( nb simulation, nb stocks )
    /// \param p_condExp    Conditional expectation objet
    /// \return     solution obtained after one step of dynamic programming (  nb simulation, nb stocks ) and the optimal control for each control
    std::pair< std::vector< std::shared_ptr< Eigen::ArrayXXd > >, std::vector<  std::shared_ptr< Eigen::ArrayXXd > > > oneStep(const std::vector< std::shared_ptr< Eigen::ArrayXXd > > &p_phiIn,
            const std::shared_ptr< BaseRegression>  &p_condExp) const ;

    /// \brief Permits to dump continuation values on archive
    /// \param p_ar                   archive to dump in
    /// \param p_name                 name used for object
    /// \param  p_iStep               Step number or identifier for time step
    /// \param p_phiInPrev            for each regime the function value  ( nb simulation, nb stocks )
    /// \param p_control              Optimal control ( nb simulation ,nb stocks) for each control at the current date
    /// \param p_condExp              conditional expectation operator
    /// \param p_bOneFile             if true Bellman values are store in one file
    void dumpContinuationValues(std::shared_ptr<gs::BinaryFileArchive> p_ar, const std::string &p_name, const int &p_iStep,
                                const std::vector< std::shared_ptr< Eigen::ArrayXXd > > &p_phiInPrev,
                                const std::vector< std::shared_ptr< Eigen::ArrayXXd > >   &p_control,
                                const  std::shared_ptr<BaseRegression>    &p_condExp,
                                const bool &p_bOneFile) const;


    /// \brief Permits to dump Bellman values on archive
    /// \param p_ar                   archive to dump in
    /// \param p_name                 name used for object
    /// \param p_iStep                 Step number or identifier for time step
    /// \param p_phiIn                for each regime the function value ( nb simulation ,nb stocks)
    /// \param p_condExp               conditional expectation operator
    /// \param p_bOneFile             if true Bellman values are store in one file
    void dumpBellmanValues(std::shared_ptr<gs::BinaryFileArchive> p_ar, const std::string &p_name, const int &p_iStep, const std::vector< std::shared_ptr< Eigen::ArrayXXd > > &p_phiIn,
                           const  std::shared_ptr<BaseRegression>    &p_condExp,
                           const bool &p_bOneFile) const;
};
}
#endif /* TRANSITIONSTEPREGRESSIONDPDIST_H */

