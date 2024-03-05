
#ifndef TRANSITIONSTEPTREEDPCUT_H
#define TRANSITIONSTEPTREEDPCUT_H
#ifdef OMP
#include <omp.h>
#endif
#include <memory>
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "libflow/core/grids/FullGrid.h"
#include "libflow/tree/Tree.h"
#include "libflow/tree/ContinuationCutsTree.h"
#include "libflow/dp/OptimizerDPCutTreeBase.h"

/** \file TransitionStepTreeDPCut.h
 * \brief Solve the dynamic programming  problem on one time step by tree with multi thread and mpi without distribution of the data
 *   The transition problem is written with cuts  so that the transition problem is written with  LP solver.
 * \author Xavier Warin
  */


namespace libflow
{

/// \class TransitionStepTreeDPCut TransitionStepTreeDPCut.h
///        One step of dynamic programming without using mpi
class TransitionStepTreeDPCut
{
private :


    std::shared_ptr<FullGrid>  m_pGridCurrent ; ///< global grid at current time step
    std::shared_ptr<FullGrid>  m_pGridPrevious ; ///< global grid at previous time step
    std::shared_ptr<OptimizerDPCutTreeBase  >  m_pOptimize ; ///< optimizer solving the problem for one point and one step
#ifdef USE_MPI
    boost::mpi::communicator  m_world; ///< Mpi communicator
#endif

public :

    /// \brief Constructor
    TransitionStepTreeDPCut(const  std::shared_ptr<FullGrid> &p_pGridCurrent,
                            const  std::shared_ptr<FullGrid> &p_pridPrevious,
                            const  std::shared_ptr<OptimizerDPCutTreeBase > &p_pOptimize
#ifdef USE_MPI
                            , const boost::mpi::communicator &p_world
#endif
                           );

    /// \brief One step for dynamic programming in optimization
    /// \param p_phiIn      for each regime the function cut value ( (nb nodes at next date * nb cuts), nb stocks ) coming from next step
    /// \param p_condExp    Conditional expectation object
    /// \return     For each regime, vector contained the cut value for each ((number of nodes at current date* nbcuts) * stock number)
    ///             each Eigen array has shape  (nb nodes at current date* nbcuts) by  stock number
    std::vector<  std::shared_ptr< Eigen::ArrayXXd > >  oneStep(const std::vector< std::shared_ptr< Eigen::ArrayXXd > > &p_phiIn,
            const std::shared_ptr< Tree>     &p_condExp) const ;


    /// \brief Permits to dump continuation values on archive
    /// \param p_ar                   archive to dump in
    /// \param p_name                 name used for object
    /// \param p_iStep                Step number or identifier for time step
    /// \param p_phiIn                for each regime  the function value ( nb nodes at current date* nb cuts ,nb stocks)
    /// \param p_condExp               Conditional expectation object (tree)
    void dumpContinuationCutsValues(std::shared_ptr<gs::BinaryFileArchive> p_ar, const std::string &p_name, const int &p_iStep, const std::vector<  std::shared_ptr< Eigen::ArrayXXd > > &p_phiIn, const std::shared_ptr< Tree>     &p_condExp) const;
};
}
#endif /* TRANSITIONSTEPTREEDPCUT_H */

