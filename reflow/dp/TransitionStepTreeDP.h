
#ifndef TRANSITIONSTEPTREEDP_H
#define TRANSITIONSTEPTREEDP_H
#ifdef OMP
#include <omp.h>
#endif
#include <memory>
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "reflow/dp/TransitionStepTreeBase.h"
#include "reflow/core/grids/FullGrid.h"
#include "reflow/tree/Tree.h"
#include "reflow/dp/OptimizerDPTreeBase.h"

/** \file TransitionStepTreeDP.h
 * \brief Solve the dynamic programming  problem on one time step by tree with multi thread and mpi without distribution of the data
 * \author Xavier Warin
  */


namespace reflow
{

/// \class TransitionStepRegressionDP TransitionStepRegressionDP.h
///        One step of dynamic programming without using mpi with tree method
class TransitionStepTreeDP  :  public TransitionStepTreeBase
{
private :


    std::shared_ptr<FullGrid>  m_pGridCurrent ; ///< global grid at current time step
    std::shared_ptr<FullGrid>  m_pGridPrevious ; ///< global grid at previous time step
    std::shared_ptr<OptimizerDPTreeBase  >  m_pOptimize ; ///< optimizer solving the problem for one point and one step
#ifdef USE_MPI
    boost::mpi::communicator  m_world; ///< Mpi communicator
#endif

public :

    /// \brief default
    TransitionStepTreeDP() {}

    virtual ~TransitionStepTreeDP() {}

    /// \brief Constructor
    TransitionStepTreeDP(const  std::shared_ptr<FullGrid> &p_pGridCurrent,
                         const  std::shared_ptr<FullGrid> &p_pridPrevious,
                         const  std::shared_ptr<OptimizerDPTreeBase > &p_pOptimize
#ifdef USE_MPI
                         , const boost::mpi::communicator &p_world
#endif
                        );

    /// \brief One step for dynamic programming in optimization
    /// \param p_phiIn      for each regime the function value ( nb node in tree at next date, nb stocks )
    /// \param p_condExp    Conditional expectation object
    /// \return     solution obtained after one step of dynamic programming and the optimal control
    std::pair< std::vector< std::shared_ptr< Eigen::ArrayXXd > >, std::vector<  std::shared_ptr<  Eigen::ArrayXXd > > >  oneStep(const std::vector< std::shared_ptr< Eigen::ArrayXXd > > &p_phiIn,
            const std::shared_ptr< Tree>     &p_condExp) const ;


    /// \brief Permits to dump continuation values on archive
    /// \param p_ar                   archive to dump in
    /// \param p_name                 name used for object
    /// \param p_iStep                Step number or identifier for time step
    /// \param p_phiIn                for each regime the function value at next date ( nb node at next  date ,nb stocks)
    /// \param p_control              Optimal control ( nb node at curretn date ,nb stocks) for each control
    /// \param p_tree                 Tree to calculate conditiona expectation
    void dumpContinuationValues(std::shared_ptr<gs::BinaryFileArchive> p_ar, const std::string &p_name, const int &p_iStep,
                                const std::vector< std::shared_ptr< Eigen::ArrayXXd > > &p_phiIn,
                                const std::vector< std::shared_ptr< Eigen::ArrayXXd > > &p_control,
                                const std::shared_ptr< Tree>      &p_tree) const;
};
}
#endif /* TRANSITIONSTEPTREEDP_H */

