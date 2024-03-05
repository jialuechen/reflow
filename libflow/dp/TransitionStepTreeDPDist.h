
#ifndef TRANSITIONSTEPTREEDPDIST_H
#define TRANSITIONSTEPTREEDP_H
#ifdef OMP
#include <omp.h>
#endif
#include <memory>
#include <boost/mpi.hpp>
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "libflow/dp/TransitionStepTreeBase.h"
#include "libflow/dp/TransitionStepBaseDist.h"
#include "libflow/core/grids/FullGrid.h"
#include "libflow/tree/Tree.h"
#include "libflow/dp/OptimizerDPTreeBase.h"

/** \file TransitionStepTreeDP.h
 * \brief Solve the dynamic programming  problem on one time step by tree with distribution of the data
 * \author Xavier Warin
  */


namespace libflow
{

/// \class TransitionStepRegressionDPDist TransitionStepRegressionDPDist.h
///        One step of dynamic programming  using mpi with tree method
class TransitionStepTreeDPDist  :  public TransitionStepTreeBase, public TransitionStepBaseDist
{

public :

    /// \brief default
    TransitionStepTreeDPDist() {}

    virtual ~TransitionStepTreeDPDist() {}

    /// \brief Constructor
    TransitionStepTreeDPDist(const  std::shared_ptr<FullGrid> &p_pGridCurrent,
                             const  std::shared_ptr<FullGrid> &p_pridPrevious,
                             const  std::shared_ptr<OptimizerDPTreeBase > &p_pOptimize,
                             const boost::mpi::communicator &p_world);

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
    /// \param p_phiIn                for each regime the function value at next date ( nb node at node date ,nb stocks)
    /// \param p_control              Optimal control ( nb node at current date ,nb stocks) for each control
    /// \param p_tree                 Tree to calculate conditiona expectation
    /// \param p_bOneFile             if true Bellman values are store in one file
    void dumpContinuationValues(std::shared_ptr<gs::BinaryFileArchive> p_ar,
                                const std::string &p_name, const int &p_iStep,
                                const std::vector< std::shared_ptr< Eigen::ArrayXXd > > &p_phiIn,
                                const std::vector< std::shared_ptr< Eigen::ArrayXXd > > &p_control,
                                const std::shared_ptr< Tree>      &p_tree,
                                const bool &p_bOneFile) const;
};
}
#endif /* TRANSITIONSTEPTREEDPDIST_H */

