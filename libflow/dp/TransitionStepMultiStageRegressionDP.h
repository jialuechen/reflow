

#ifndef TRANSITIONSTEPMULTISTAGEREGRESSIONDP_H
#define TRANSITIONSTEPMULTISTAGEREGRESSIONDP_H
#ifdef OMP
#include <omp.h>
#endif
#include <memory>
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "libflow/core/grids/FullGrid.h"
#include "libflow/regression/BaseRegression.h"
#include "libflow/regression/ContinuationValue.h"
#include "libflow/dp/OptimizerMultiStageDPBase.h"

/** \file TransitionStepMultiStageRegressionDP.h
 * \brief Solve the dynamic programming  problem on one time step by regression with multi thread and mpi without distribution of the data
 *        In this version, on each time time a multistage deterministic problem is solved using DP
 * \author Benoit Clair , Xavier Warin
  */


namespace libflow
{

/// \class TransitionStepMultiStageRegressionDP TransitionStepMultiStageRegressionDP.h
///        One step of dynamic programming without using mpi
class TransitionStepMultiStageRegressionDP
{
private :

    std::shared_ptr<FullGrid>  m_pGridCurrent ; ///< global grid at current time step
    std::shared_ptr<FullGrid>  m_pGridPrevious ; ///< global grid at previous time step
    std::shared_ptr<OptimizerMultiStageDPBase  >  m_pOptimize ; ///< optimizer solving the problem for one point and one step
    std::shared_ptr<gs::BinaryFileArchive> m_arGen ; ///< geners archive
    std::string  m_nameDump ; ///< name to dump deterministic values

#ifdef USE_MPI
    boost::mpi::communicator  m_world; ///< Mpi communicator
#endif


    /// \brief calculate one step period
    /// \param p_phiIn             cash values at the end of period (nb sim * nb stock points)
    /// \param p_phiOut            cash to be filled at the beginning of period  (nb sim * nb stock points)
    /// \param p_contVal           continuation object
    /// \param p_pGridCurTrans     current grid in transition
    /// \param p_pGridPrevTrans    previous grid in transition
    /// \param p_phiOutLoc        utility for MPI calculations
    /// \param p_ilocToGLobal     utility for MPI calculations
    /// \param p_ilocToGLobalGlob utility for MPI calculations
    /// \param p_storeGlob        utility for MPI calculations
    void oneStageInStep(const std::vector< std::shared_ptr< Eigen::ArrayXXd > > &p_phiIn,
                        std::vector< std::shared_ptr< Eigen::ArrayXXd > > &p_phiOut,
                        std::vector<std::shared_ptr<ContinuationValue> > &p_contVal,
                        const std::shared_ptr<FullGrid> &p_pGridCurTrans,
                        const std::shared_ptr<FullGrid> &p_pGridPrevTrans
#ifdef USE_MPI
                        , std::vector< Eigen::ArrayXXd >   &p_phiOutLoc,
                        Eigen::ArrayXi &p_ilocToGLobal,
                        Eigen::ArrayXi &p_ilocToGLobalGlob,
                        Eigen::ArrayXXd   &p_storeGlob
#endif
                       ) const;


public :

    /// \brief default
    TransitionStepMultiStageRegressionDP() {}
    virtual ~TransitionStepMultiStageRegressionDP() {}

    /// \brief Constructor without arxive dump
    TransitionStepMultiStageRegressionDP(const  std::shared_ptr<FullGrid> &p_pGridCurrent,
                                         const  std::shared_ptr<FullGrid> &p_pGridPrevious,
                                         const  std::shared_ptr<OptimizerMultiStageDPBase > &p_pOptimize
#ifdef USE_MPI
                                         , const boost::mpi::communicator &p_world
#endif
                                        );

    /// \brief Constructor without arxive dump
    TransitionStepMultiStageRegressionDP(const  std::shared_ptr<FullGrid> &p_pGridCurrent,
                                         const  std::shared_ptr<FullGrid> &p_pGridPrevious,
                                         const  std::shared_ptr<OptimizerMultiStageDPBase > &p_pOptimize,
                                         const  std::shared_ptr<gs::BinaryFileArchive>   &p_arGen,
                                         const  std::string &p_nameDump
#ifdef USE_MPI
                                         , const boost::mpi::communicator &p_world
#endif
                                        );

    /// \brief One step for dynamic programming in optimization
    /// \param p_phiIn             for each regime the function value ( nb simulation, nb stocks )
    /// \param p_condExp           Conditional expectation object
    std::vector< std::shared_ptr< Eigen::ArrayXXd > >  oneStep(const std::vector< std::shared_ptr< Eigen::ArrayXXd > > &p_phiIn,
            const std::shared_ptr< BaseRegression>     &p_condExp) const ;


    /// \brief Permits to dump stochastic continuation values on archive during time step optimization
    /// \param p_ar                   archive to dump in
    /// \param p_name                 name used for object
    /// \param p_iStep                 Step number or identifier for time step
    /// \param p_phiIn                for each regime the function value ( nb simulation ,nb stocks)
    /// \param p_condExp               conditional expectation operator
    void dumpContinuationValues(std::shared_ptr<gs::BinaryFileArchive> p_ar, const std::string &p_name, const int &p_iStep, const std::vector< std::shared_ptr< Eigen::ArrayXXd > > &p_phiIn, const  std::shared_ptr<BaseRegression>    &p_condExp) const;


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
#endif /* TRANSITIONSTEPMULTISTAGEREGRESSIONDP_H */

