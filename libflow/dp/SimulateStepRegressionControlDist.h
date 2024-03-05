// Copyright (C) 2016 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#ifndef SIMULATESTEPREGRESSIONCONTROLDIST_H
#define SIMULATESTEPREGRESSIONCONTROLDIST_H
#include <boost/lexical_cast.hpp>
#include "geners/BinaryFileArchive.hh"
#include "libflow/dp/SimulateStepBase.h"
#include "libflow/core/grids/FullGrid.h"
#include "libflow/core/utils/StateWithStocks.h"
#include "libflow/regression/GridAndRegressedValue.h"
#include "libflow/regression/GridAndRegressedValueGeners.h"
#include "libflow/core/parallelism/ParallelComputeGridSplitting.h"
#include "libflow/dp/OptimizerBaseInterp.h"

/** SimulateStepRegressionControlDist.h
 *  In simulation part, permits to  use  continuation values stored in in optimization step and
 *  calculate optimal control
 */

namespace libflow
{
/// \class SimulateStepRegressionControlDist SimulateStepRegressionControlDist.h
/// One step in forward simulation using mpi
class SimulateStepRegressionControlDist : public SimulateStepBase
{
private :

    std::shared_ptr<FullGrid>  m_pGridCurrent ; ///< global grid at current time step
    std::shared_ptr<FullGrid>  m_pGridFollowing ; ///< global grid at following time step
    std::shared_ptr<OptimizerBaseInterp >          m_pOptimize ; ///< optimizer solving the problem for one point and one step
    std::vector< GridAndRegressedValue >  m_control ; ///< to store optimal control at the current step
    std::shared_ptr<BaseRegression>  m_regressor ; ///< Regressor used (read if multiple file used)
    std::vector< Eigen::ArrayXXd > m_contValue ; ///< to store control values split on current processor
    bool m_bOneFile ; /// do we use one file for continuation values
    std::shared_ptr<ParallelComputeGridSplitting>  m_parall  ; ///< parallel object for splitting and reconstruction
    boost::mpi::communicator  m_world; ///< Mpi communicator

public :

    /// \brief default
    SimulateStepRegressionControlDist() {}
    virtual ~SimulateStepRegressionControlDist() {}

    /// \brief Constructor
    /// \param p_ar               Archive where continuation values are stored
    /// \param p_iStep            Step number identifier
    /// \param p_nameCont         Name use to store continuation valuation
    /// \param p_pGridCurrent     grid at current  time step
    /// \param p_pGridFollowing   grid at following time step
    /// \param p_pOptimize        Optimize object defining the transition step
    /// \param p_bOneFile              do we store continuation values  in only one file
    /// \param p_world            MPI communicator
    SimulateStepRegressionControlDist(gs::BinaryFileArchive &p_ar,  const int &p_iStep,  const std::string &p_nameCont,
                                      const   std::shared_ptr<FullGrid> &p_pGridCurrent,
                                      const   std::shared_ptr<FullGrid> &p_pGridFollowing,
                                      const  std::shared_ptr<OptimizerBaseInterp > &p_pOptimize,
                                      const bool &p_bOneFile, const boost::mpi::communicator &p_world);

    /// \brief Define one step arbitraging between possible commands
    /// \param p_statevector    Vector of states (regime, stock descriptor, uncertainty)
    /// \param p_phiInOut       actual contract value modified at current time step by applying an optimal command
    void oneStep(std::vector<StateWithStocks > &p_statevector, Eigen::ArrayXXd  &p_phiInOut) const;


};
}
#endif /* SIMULATESTEPREGRESSIONCONTROLDIST_H */
