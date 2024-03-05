// Copyright (C) 2016 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#ifndef TRANSITIONSTEPSEMILAGRANGDIST_H
#define TRANSITIONSTEPSEMILAGRANGDIST_H
#ifdef OMP
#include <omp.h>
#endif
#include <boost/mpi.hpp>
#include <memory>
#include <functional>
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "libflow/core/grids/FullGrid.h"
#include "libflow/core/grids/InterpolatorSpectral.h"
#include "libflow/core/parallelism/ParallelComputeGridSplitting.h"
#include "libflow/semilagrangien/OptimizerSLBase.h"

/** \file TransitionStepSemilagrangDist.h
 * \brief Solve one step  of explicit semi Lagrangian scheme
 *        Mpi is used to split the  grid between the processors so calculations and data are spread on processors
 * \author Xavier Warin
 */


namespace libflow
{

/// \class TransitionStepSemilagrangDist TransitionStepSemilagrangDist.h
///        One step  of semi Lagrangian scheme
class TransitionStepSemilagrangDist
{
private :

    std::shared_ptr<FullGrid>  m_gridCurrent ; ///< global grid at current time step
    std::shared_ptr<FullGrid>  m_gridPrevious ; ///< global grid at previous time step
    std::shared_ptr<OptimizerSLBase  >  m_optimize ; ///< optimizer solving the problem for one point and one step
    std::shared_ptr<ParallelComputeGridSplitting> m_paral ; ///< parallel object
    std::shared_ptr<FullGrid>   m_gridCurrentProc ; ///< local grid  treated by the processor
    std::shared_ptr<FullGrid>   m_gridExtendPreviousStep; ///< give extended grid at previous step
    boost::mpi::communicator m_world ; ///< MPI communicator

public :

    /// \brief Constructor
    TransitionStepSemilagrangDist(const  std::shared_ptr<FullGrid> &p_gridCurrent,
                                  const  std::shared_ptr<FullGrid> &p_gridPrevious,
                                  const  std::shared_ptr<OptimizerSLBase > &p_optimize,
                                  const boost::mpi::communicator &p_world);

    /// \brief One step for
    /// \param p_phiIn         for each regime the function value ( on the grid)
    /// \param p_time          current date
    /// \param p_boundaryFunc  Function at the boundary to impose Dirichlet conditions (depending on regime and position)
    /// \return             solution obtained after one step of the explicit semi Lagrangian scheme
    std::pair< std::vector< std::shared_ptr< Eigen::ArrayXd > >, std::vector< std::shared_ptr< Eigen::ArrayXd >  >  >  oneStep(const std::vector< std::shared_ptr< Eigen::ArrayXd > > &p_phiIn,  const double &p_time,   const std::function<double(const int &, const Eigen::ArrayXd &)> &p_boundaryFunc) const;

    /// \brief Permits to dump continuation values on archive
    /// \param p_ar                   archive to dump in
    /// \param p_name                 name used for object
    /// \param p_iStep               Step number or identifier for time step
    /// \param p_phi                 for each regime the function value
    /// \param p_control              for each control, the optimal value
    /// \param p_bOneFile             if true Bellman values are store in one file
    void dumpValues(std::shared_ptr<gs::BinaryFileArchive> p_ar, const std::string &p_name, const int &p_iStep, const std::vector< std::shared_ptr< Eigen::ArrayXd > > &p_phi,
                    const std::vector< std::shared_ptr< Eigen::ArrayXd > > &p_control,  const bool &p_bOneFile) const;

    /// \brief get back local grid (to processor) associated to current step
    inline std::shared_ptr<FullGrid >   getGridCurrentProc() const
    {
        return m_gridCurrentProc ;
    }
    /// \brief Reconstruct on processor 0 on the current grid
    /// \param p_phiIn data owned by current processor
    /// \param p_phiOut data reconstructed
    void reconstructOnProc0(const std::vector< std::shared_ptr< Eigen::ArrayXd > > &p_phiIn, std::vector< std::shared_ptr< Eigen::ArrayXd > > &p_phiOut);
};
}
#endif /* TRANSITIONSTEPSEMILAGRANGDIST_H */

