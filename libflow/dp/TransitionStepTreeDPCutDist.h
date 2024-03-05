
#ifndef TRANSITIONSTEPTREEDPCUTDIST_H
#define TRANSITIONSTEPTREEDPCUTDIST_H
#include <functional>
#include <memory>
#include <boost/mpi.hpp>
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "geners/Record.hh"
#include "libflow/dp/TransitionStepBaseDist.h"
#include "libflow/core/grids/FullGrid.h"
#include "libflow/core/parallelism/ParallelComputeGridSplitting.h"
#include "libflow/tree/Tree.h"
#include "libflow/dp/OptimizerDPCutTreeBase.h"

/** \file TransitionStepTreeDPCutDist.h
 * \brief Solve the dynamic programming  problem on one time step by tree with parallelization
 *  The transition problem is written with cuts  so that the transition problem is written with  LP solver.
 * \author Xavier Warin
 */
namespace libflow
{
/// \class TransitionStepTreeDPCutDist TransitionStepTreeDPCutDist.h
///        One step of dynamic programming using MPI
class TransitionStepTreeDPCutDist :  public TransitionStepBaseDist
{

public :

    /// \brief Constructor
    TransitionStepTreeDPCutDist(const  std::shared_ptr<FullGrid> &p_pGridCurrent,
                                const  std::shared_ptr<FullGrid> &p_pGridPrevious,
                                const  std::shared_ptr<OptimizerDPCutTreeBase > &p_pOptimize,
                                const boost::mpi::communicator &p_world);

    /// \brief One step for dynamic programming in optimization
    /// \param p_phiIn      for each regime the function cut value ( (nb nodes at next date * nb cuts), nb stocks ) coming from next step
    /// \param p_condExp    Conditional expectation object
    /// \return     For each regime, vector contained the cut value for each ((nb nodes current date * nbcuts) * stock number)
    ///             each Eigen array has shape  (nb nodes at current date* nbcuts) by  stock number
    std::vector<  std::shared_ptr< Eigen::ArrayXXd > >  oneStep(const std::vector< std::shared_ptr< Eigen::ArrayXXd > > &p_phiIn,
            const std::shared_ptr< Tree>     &p_condExp) const ;

    /// \brief Permits to dump continuation values on archive
    /// \param p_ar                   archive to dump in
    /// \param p_name                 name used for object
    /// \param  p_iStep               Step number or identifier for time step
    /// \param p_phiInPrev            for each regime  the function value ( nb nodes at current date* nb cuts ,nb stocks)
    /// \param p_condExp               Conditional expectation object (tree)
    /// \param p_bOneFile             if true Bellman values are store in one file
    void dumpContinuationCutsValues(std::shared_ptr<gs::BinaryFileArchive> p_ar, const std::string &p_name, const int &p_iStep,
                                    const std::vector< std::shared_ptr< Eigen::ArrayXXd > > &p_phiInPrev, const std::shared_ptr< Tree>     &p_condExp,
                                    const bool &p_bOneFile) const;
};
}
#endif /* TRANSITIONSTEPTREEDPCUTDIST_H */

