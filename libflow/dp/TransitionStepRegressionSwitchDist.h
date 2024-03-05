// Copyright (C) 2021 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#ifndef TRANSITIONSTEPREGRESSIONSWITCHDIST_H
#define TRANSITIONSTEPREGRESSIONSWITCHDIST_H
#include <functional>
#include <memory>
#include <vector>
#include <boost/mpi.hpp>
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "geners/Record.hh"
#include "libflow/core/grids/RegularSpaceIntGrid.h"
#include "libflow/core/parallelism/ParallelComputeGridSplitting.h"
#include "libflow/regression/BaseRegression.h"
#include "libflow/dp/OptimizerSwitchBase.h"

/** \file TransitionStepRegressionSwitchDist.h
 * \brief Solve the dynamic programming  problem on one time step by regression with parallelization for switching problems with integer states
 * \author Xavier Warin
 * \todo Developp MPI for regimes too : parallelism should be applied to stock and regimes.
 */
namespace libflow
{
/// \class TransitionStepRegressionSwitchDist TransitionStepRegressionSwitchDist.h
///        One step of dynamic programming using MPI for switching problems with integer states
class TransitionStepRegressionSwitchDist
{
private :
    std::vector< std::shared_ptr<RegularSpaceIntGrid> >           m_pGridCurrent ; ///< global grid for each regime at current time step
    std::vector< std::shared_ptr<RegularSpaceIntGrid> >           m_pGridPrevious ; ///< global grid for each regime at previous time step
    std::shared_ptr<OptimizerSwitchBase >                         m_pOptimize ; ///< optimizer solving the problem for one point and one step
    std::vector< std::shared_ptr<ParallelComputeGridSplitting> >  m_paral ; ///< parallel object fro each regime
    std::vector< std::shared_ptr<RegularSpaceIntGrid> >    m_gridCurrentProc ; ///<  vector of local grid  treated by the processor (for each regime)
    std::vector< std::shared_ptr<RegularSpaceIntGrid> >    m_gridExtendPreviousStep; ///< vector of given extended grid at previous step  (for each regime)
    boost::mpi::communicator  m_world; ///< Mpi communicator
public :

    /// \brief default
    TransitionStepRegressionSwitchDist() {}

    /// \brief Destructor
    virtual ~TransitionStepRegressionSwitchDist() {}

    /// \brief Constructor
    TransitionStepRegressionSwitchDist(const  std::vector< std::shared_ptr<RegularSpaceIntGrid> >  &p_pGridCurrent,
                                       const  std::vector< std::shared_ptr<RegularSpaceIntGrid> >  &p_pGridPrevious,
                                       const  std::shared_ptr<OptimizerSwitchBase > &p_pOptimize,
                                       const boost::mpi::communicator &p_world);

    /// \brief One step for optimization
    /// \param p_phiIn      for each regime the function value  ( nb simulation, nb stocks )
    /// \param p_condExp    Conditional expectation objet
    /// \return     solution obtained after one step of dynamic programming (  nb simulation, nb non stochastic states  )
    std::vector< std::shared_ptr< Eigen::ArrayXXd > >  oneStep(const std::vector< std::shared_ptr< Eigen::ArrayXXd > > &p_phiIn, const std::shared_ptr< BaseRegression>  &p_condExp) const ;


    /// \brief Permits to dump continuation values on archive : for switching dump is realized on a single file
    /// \param p_ar                   archive to dump in
    /// \param p_name                 name used for object
    /// \param  p_iStep               Step number or identifier for time step
    /// \param p_phiInPrev            for each regime the function value  ( nb simulation, nb stocks )
    /// \param p_condExp              conditional expectation operator
    void dumpContinuationValues(std::shared_ptr<gs::BinaryFileArchive> p_ar, const std::string &p_name, const int &p_iStep,
                                const std::vector< std::shared_ptr< Eigen::ArrayXXd > > &p_phiInPrev,
                                const  std::shared_ptr<BaseRegression>    &p_condExp) const;

};
}
#endif /* TRANSITIONSTEPREGRESSIONSWITCHDIST_H */

