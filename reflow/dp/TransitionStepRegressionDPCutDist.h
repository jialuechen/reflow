
#ifndef TRANSITIONSTEPREGRESSIONDPCUTDIST_H
#define TRANSITIONSTEPREGRESSIONDPCUTDIST_H
#include <functional>
#include <memory>
#include <boost/mpi.hpp>
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "geners/Record.hh"
#include "reflow/dp/TransitionStepBaseDist.h"
#include "reflow/core/grids/FullGrid.h"
#include "reflow/core/parallelism/ParallelComputeGridSplitting.h"
#include "reflow/regression/BaseRegression.h"
#include "reflow/dp/OptimizerDPCutBase.h"

/** \file TransitionStepRegressionDPCutDist.h
 * \brief Solve the dynamic programming  problem on one time step by regression with parallelization
 *  The transition problem is written with cuts  so that the transition problem is written with  LP solver.
 * \author Xavier Warin
 */
namespace reflow
{
/// \class TransitionStepRegressionDPCutDist TransitionStepRegressionDPCutDist.h
///        One step of dynamic programming using MPI
class TransitionStepRegressionDPCutDist :  public TransitionStepBaseDist
{

public :

    /// \brief Constructor
    TransitionStepRegressionDPCutDist(const  std::shared_ptr<FullGrid> &p_pGridCurrent,
                                      const  std::shared_ptr<FullGrid> &p_pGridPrevious,
                                      const  std::shared_ptr<OptimizerDPCutBase > &p_pOptimize,
                                      const boost::mpi::communicator &p_world);

    /// \brief One step for dynamic programming in optimization
    /// \param p_phiIn      for each regime the function cut value ( (nb simulation * nb cuts), nb stocks ) coming from next step
    /// \param p_condExp    Conditional expectation object
    /// \return     For each regime, vector contained the cut value for each ((simulation* nbcuts) * stock number)
    ///             each Eigen array has shape  (simulation* nbcuts) by  stock number
    std::vector<  std::shared_ptr< Eigen::ArrayXXd > >  oneStep(const std::vector< std::shared_ptr< Eigen::ArrayXXd > > &p_phiIn,
            const std::shared_ptr< BaseRegression>     &p_condExp) const ;

    /// \brief Permits to dump continuation values on archive
    /// \param p_ar                   archive to dump in
    /// \param p_name                 name used for object
    /// \param  p_iStep               Step number or identifier for time step
    /// \param p_phiInPrev              for each regime  the function value ( nb simulation* nb cuts ,nb stocks)
    /// \param p_condExp              conditional expectation operator
    /// \param p_bOneFile             if true Bellman values are store in one file
    void dumpContinuationCutsValues(std::shared_ptr<gs::BinaryFileArchive> p_ar, const std::string &p_name, const int &p_iStep,
                                    const std::vector< std::shared_ptr< Eigen::ArrayXXd > > &p_phiInPrev,
                                    const  std::shared_ptr<BaseRegression>    &p_condExp,
                                    const bool &p_bOneFile) const;

    /// \brief Permits to dump Bellman values on archive
    /// \param p_ar                   archive to dump in
    /// \param p_name                 name used for object
    /// \param  p_iStep               Step number or identifier for time step
    /// \param p_phiIn                for each regime  the function value ( nb simulation* nb cuts ,nb stocks)
    /// \param p_condExp              conditional expectation operator
    /// \param p_bOneFile             if true Bellman values are store in one file
    void dumpBellmanCutsValues(std::shared_ptr<gs::BinaryFileArchive> p_ar, const std::string &p_name, const int &p_iStep,
                               const std::vector< std::shared_ptr< Eigen::ArrayXXd > > &p_phiIn,
                               const  std::shared_ptr<BaseRegression>    &p_condExp,
                               const bool &p_bOneFile) const;
};
}
#endif /* TRANSITIONSTEPREGRESSIONDPCUTDIST_H */

