
#ifndef TRANSITIONSTEPDPDIST_H
#define TRANSITIONSTEPDPDIST_H
#include <functional>
#include <memory>
#include <boost/mpi.hpp>
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "geners/Record.hh"
#include "reflow/dp/TransitionStepBase.h"
#include "reflow/dp/TransitionStepBaseDist.h"
#include "reflow/core/parallelism/ParallelComputeGridSplitting.h"
#include "reflow/core/grids/FullGrid.h"
#include "reflow/regression/GridAndRegressedValue.h"
#include "reflow/regression/BaseRegression.h"
#include "reflow/dp/OptimizerNoRegressionDPBase.h"

/** \file TransitionStepDPDist.h
 * \brief  Base class for  the dynamic programming when  mpi with distribution of the data is used
  * \author Xavier Warin
 */
namespace reflow
{
/// \class TransitionStepDPDist TransitionStepDPDist.h
///        One step of dynamic programming using MPI
class TransitionStepDPDist :  public TransitionStepBase, public TransitionStepBaseDist
{
private :

    std::shared_ptr<BaseRegression> m_regressorPrevious; /// regressor object at the previous date
    std::shared_ptr<BaseRegression> m_regressorCurrent; /// regressor object at the current date

public :

    /// \brief Constructor
    /// \param p_pGridCurrent  grid (stock points) at the current time step
    /// \param p_pridPrevious  grid (stock points) at the previusly treated time step
    /// \param p_regressorCurrent   regressor (with respect to simulation) at the current time step
    /// \param p_regressorPrevious  regressor (with respect to simulation) at the previously treated time step
    /// \param p_pOptimize           optimizer object to optimizer the problem on one time step
    /// \param p_world               MPI communicator
    TransitionStepDPDist(const  std::shared_ptr<FullGrid> &p_pGridCurrent,
                         const  std::shared_ptr<FullGrid> &p_pridPrevious,
                         const  std::shared_ptr<BaseRegression> &p_regressorCurrent,
                         const  std::shared_ptr<BaseRegression> &p_regressorPrevious,
                         const  std::shared_ptr<OptimizerNoRegressionDPBase> &p_pOptimize,
                         const boost::mpi::communicator &p_world);

    /// \brief One step for optimization
    /// \param p_phiIn      for each regime the function value at the next time step : store as (regressed function by number of  grid points)
    /// \return     solution obtained after one step of dynamic programming and the optimal control  (regressed function by the number of grid points)
    virtual std::pair< std::shared_ptr< std::vector< Eigen::ArrayXXd > >, std::shared_ptr< std::vector< Eigen::ArrayXXd  > > >   oneStep(const std::vector< Eigen::ArrayXXd  > &p_phiIn)  const ;

    /// \brief Permits to dump the  control
    /// \param p_ar                   archive to dump in
    /// \param p_name                 name used for object
    /// \param p_iStep                 Step number or identifier for time step
    /// \param p_control              Optimal control ( nb of basis functions, number of points on the grid) for each control
    /// \param p_bOneFile              True if only one file is used to dump the controls
    void dumpValues(std::shared_ptr<gs::BinaryFileArchive> p_ar,
                    const std::string &p_name, const int &p_iStep,
                    const std::vector<  Eigen::ArrayXXd  > &p_control, const bool &p_bOneFile) const;
};
}
#endif /* TRANSITIONSTEPDPDIST_H */

