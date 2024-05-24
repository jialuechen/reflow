
#ifndef TRANSITIONSTEPREGRESSIONDPSPARSE_H
#define TRANSITIONSTEPREGRESSIONDPSPARSE_H
#ifdef OMP
#include <omp.h>
#endif
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <memory>
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "reflow/dp/TransitionStepRegressionBase.h"
#include "reflow/core/grids/SparseSpaceGrid.h"
#include "reflow/regression/BaseRegression.h"
#include "reflow/dp/OptimizerDPBase.h"

/** \file TransitionStepRegressionDPSparse.h
 * \brief Solve the dynamic programming  problem on one time step by regression with sparse grids
 * \author Xavier Warin
  */


namespace reflow
{

/// \class TransitionStepRegressionDPSparse TransitionStepRegressionDPSparse.h
///        One step of dynamic programming using sparse  and mpi potentially
class TransitionStepRegressionDPSparse :  public TransitionStepRegressionBase
{
private :

    std::shared_ptr<SparseSpaceGrid>  m_pGridCurrent ; ///< global grid at current time step
    std::shared_ptr<SparseSpaceGrid>  m_pGridPrevious ; ///< global grid at previous time step
    std::shared_ptr<OptimizerDPBase  >  m_pOptimize ; ///< optimizer solving the problem for one point and one step
#ifdef USE_MPI
    boost::mpi::communicator  m_world; ///< Mpi communicator
#endif

public :

    /// \brief Constructor
    TransitionStepRegressionDPSparse(const  std::shared_ptr<SparseSpaceGrid> &p_pGridCurrent,
                                     const  std::shared_ptr<SparseSpaceGrid> &p_pridPrevious,
                                     const  std::shared_ptr<OptimizerDPBase > &p_pOptimize
#ifdef USE_MPI
                                     , const boost::mpi::communicator &p_world
#endif
                                    );

    /// \brief One step for
    /// \param p_phiIn      for each regime the function value ( nb simulation, nb stocks )
    /// \param p_condExp    Conditional expectation object
    /// \return     solution obtained after one step of dynamic programming and the optimal control
    std::pair< std::vector< std::shared_ptr< Eigen::ArrayXXd > >, std::vector<  std::shared_ptr<  Eigen::ArrayXXd > > >  oneStep(const std::vector< std::shared_ptr< Eigen::ArrayXXd > > &p_phiIn,
            const std::shared_ptr< BaseRegression>     &p_condExp) const ;


    /// \brief Permits to dump continuation values on archive
    /// \param p_ar                   archive to dump in
    /// \param p_name                 name used for object
    /// \param p_iStep                 Step number or identifier for time step
    /// \param p_phiIn                for each regime the function value ( nb simulation ,nb stocks) which has been hierarchized
    /// \param p_control              Optimal control ( nb simulation ,nb stocks) for each control
    /// \param p_condExp               conditional expectation operator
    void dumpContinuationValues(std::shared_ptr<gs::BinaryFileArchive> p_ar, const std::string &p_name, const int &p_iStep, const std::vector< std::shared_ptr< Eigen::ArrayXXd > > &p_phiIn,
                                const std::vector< std::shared_ptr< Eigen::ArrayXXd > > &p_control, const  std::shared_ptr<BaseRegression>    &p_condExp) const;
};
}
#endif /* TRANSITIONSTEPREGRESSIONDPSPARSE_H */

