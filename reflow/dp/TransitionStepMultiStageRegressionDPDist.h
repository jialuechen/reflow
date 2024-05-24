

#ifndef TRANSITIONSTEPMULTISTAGEREGRESSIONDPDIST_H
#define TRANSITIONSTEPMULTISTAGEREGRESSIONDPDIST_H
#include <functional>
#include <memory>
#include "boost/mpi.hpp"
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "geners/Record.hh"
#include "reflow/dp/TransitionStepBaseDist.h"
#include "reflow/core/grids/FullGrid.h"
#include "reflow/core/parallelism/ParallelComputeGridSplitting.h"
#include "reflow/regression/BaseRegression.h"
#include "reflow/regression/ContinuationValueBase.h"
#include "reflow/dp/OptimizerMultiStageDPBase.h"

/** \file TransitionStepMultiStageRegressionDPDist.h
 * \brief Solve the dynamic programming  problem on one time step by regression with parallelization
 *        In this version, on each time time a multistage deterministic problem is solved using DP
 * \author Xavier Warin
 */
namespace reflow
{
/// \class TransitionStepMultiStageRegressionDPDist TransitionStepMultiStageRegressionDPDist.h
///        One step of dynamic programming using MPI. Transition problem is itself a deterministic problem.
class TransitionStepMultiStageRegressionDPDist :  public TransitionStepBaseDist
{

private :

    std::shared_ptr<gs::BinaryFileArchive> m_arGen ; ///< geners archive
    std::string  m_nameDump ; ///< name to dump deterministic values
    bool m_bOneFileDet ; ///<  if true deterministic continuaiton values are store in one file
    bool m_bDump ; ///< True if we dump Bellman
    std::shared_ptr<ParallelComputeGridSplitting> m_paralDet ; ///< parallel object for deterministic optimization
    std::shared_ptr<FullGrid>   m_gridExtendCurrentStep; ///< give extended grid at current step

    /// \brief compute parallel object for deterministic transition
    void calParalDet();

    /// \brief calculate one step period
    /// \param p_phiIn                    cash values at the end of period (nb sim * nb stock points)
    /// \param p_phiOut                   cash to be filled at the beginning of period  (nb sim * nb stock points)
    /// \param p_condExp              conditional expectation operator
    /// \param p_paralStep                parallel object used for the subtransition
    /// \param p_pGridPrevTransExtended   extended grid at the previous sub step.
    void oneStageInStep(const std::vector< std::shared_ptr< Eigen::ArrayXXd > > &p_phiIn,
                        std::vector< std::shared_ptr< Eigen::ArrayXXd > > &p_phiOut,
                        const std::shared_ptr< BaseRegression>  &p_condExp,
                        const std::shared_ptr<ParallelComputeGridSplitting> &p_paralStep,
                        const std::shared_ptr<FullGrid> &p_pGridPrevTransExtended)  const ;

    /// \brief Dump deterministic continuation  values
    /// \param p_phiInPrev   values to dump as functon value by regression
    /// \param p_condExp              conditional expectation operator
    /// \param p_iPeriod              deterministic period number
    void dumpContinuationDetValues(const std::vector< std::shared_ptr< Eigen::ArrayXXd > > &p_phiInPrev,
                                   const  std::shared_ptr<BaseRegression>    &p_condExp,
                                   const int &p_iPeriod) const ;
public :

    /// \brief default
    TransitionStepMultiStageRegressionDPDist() {}
    virtual ~TransitionStepMultiStageRegressionDPDist() {}

    /// \brief Constructor
    TransitionStepMultiStageRegressionDPDist(const  std::shared_ptr<FullGrid> &p_pGridCurrent,
            const  std::shared_ptr<FullGrid> &p_pGridPrevious,
            const  std::shared_ptr<OptimizerMultiStageDPBase > &p_pOptimize,
            const boost::mpi::communicator &p_world);

    /// \brief Constructor with archive dump
    TransitionStepMultiStageRegressionDPDist(const  std::shared_ptr<FullGrid> &p_pGridCurrent,
            const  std::shared_ptr<FullGrid> &p_pGridPrevious,
            const  std::shared_ptr<OptimizerMultiStageDPBase > &p_pOptimize,
            const  std::shared_ptr<gs::BinaryFileArchive>   &p_arGen,
            const  std::string &p_nameDump,
            const  bool   &p_bOneFileDet,
            const  boost::mpi::communicator &p_world);

    /// \brief One step for optimization
    /// \param p_phiIn      for each regime the function value  ( nb simulation, nb stocks )
    /// \param p_condExp    Conditional expectation objet
    /// \return     solution obtained after one step of dynamic programming (  nb simulation, nb stocks )
    std::vector< std::shared_ptr< Eigen::ArrayXXd > >  oneStep(const std::vector< std::shared_ptr< Eigen::ArrayXXd > > &p_phiIn,
            const std::shared_ptr< BaseRegression>  &p_condExp) const ;

    /// \brief Permits to dump continuation values on archive
    /// \param p_ar                   archive to dump in
    /// \param p_name                 name used for object
    /// \param p_iStep               Step number or identifier for time step
    /// \param p_phiInPrev            for each regime the function value  ( nb simulation, nb stocks )
    /// \param p_condExp              conditional expectation operator
    /// \param p_bOneFile             if true Bellman values are store in one file
    void dumpContinuationValues(std::shared_ptr<gs::BinaryFileArchive> p_ar, const std::string &p_name, const int &p_iStep,
                                const std::vector< std::shared_ptr< Eigen::ArrayXXd > > &p_phiInPrev,
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
#endif /* TRANSITIONSTEPMULTISTEPREGRESSIONDPDIST_H */

