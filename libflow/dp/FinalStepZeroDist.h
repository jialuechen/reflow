
#ifndef FINALSTEPZERODIST_H
#define FINALSTEPZERODIST_H
#include <functional>
#include <memory>
#include <vector>
#include <boost/mpi.hpp>
#include <Eigen/Dense>
#include "libflow/core/parallelism/ParallelComputeGridSplitting.h"


/** \file FinalStepZeroDist.h
 *  \brief permits to affect 0 to the final value
 * \author  Xavier Warin
 */
namespace libflow
{

/// \class FinalStepZeroDist FinalStepZeroDist.h
///   Last time in dynamic programming : set to 0
///   Each regime has a grid with same dimension
template< class grid>
class FinalStepZeroDist
{

private :

    std::vector<std::shared_ptr< grid> >  m_pGridCurrent; ///< grid at final time step
    std::vector<std::shared_ptr< grid> >    m_gridCurrentProc ; ///< local grid  treated by the processor
    int m_nbRegime ; ///< Number of regimes

public:

    /// \brief Contructor
    /// \param p_pGridCurrent   grids describing the whole problem for each regime
    /// \param p_bdimToSplit    Dimensions to split for parallelism ofr each regime
    /// \param p_world          MPI communicator
    FinalStepZeroDist(const  std::vector< std::shared_ptr< grid > >   &p_pGridCurrent,
                      const  std::vector< Eigen::Array< bool, Eigen::Dynamic, 1> >   &p_bdimToSplit,
                      const boost::mpi::communicator &p_world):
        m_pGridCurrent(p_pGridCurrent),
        m_gridCurrentProc(p_pGridCurrent.size()),
        m_nbRegime(p_pGridCurrent.size())
    {
        for (int iReg = 0;  iReg < m_nbRegime; ++iReg)
        {

            // initial dimension
            Eigen::ArrayXi initialDimension   = p_pGridCurrent[iReg]->getDimensions();
            // organize the hypercube splitting for parallel
            Eigen::ArrayXi splittingRatio = paraOptimalSplitting(initialDimension, p_bdimToSplit[iReg], p_world);
            // grid treated by current processor
            m_gridCurrentProc[iReg] = m_pGridCurrent[iReg]->getSubGrid(paraSplitComputationGridsProc(initialDimension, splittingRatio, p_world.rank()));
        }
    }
    /// \brief Fill array with 0 for each processor
    /// \param p_nbSimul   number of particles
    std::vector< std::shared_ptr< Eigen::ArrayXXd > > operator()(const int &p_nbSimul) const
    {
        std::vector<std::shared_ptr< Eigen::ArrayXXd > > finalValues(m_nbRegime);
        for (int iReg = 0; iReg < m_nbRegime; ++iReg)
        {
            if (m_gridCurrentProc[iReg]->getNbPoints() > 0)
            {
                finalValues[iReg] = std::make_shared<Eigen::ArrayXXd>(Eigen::ArrayXXd::Zero(p_nbSimul, m_gridCurrentProc[iReg]->getNbPoints()));
            }
            else
            {
                finalValues[iReg] = std::make_shared<Eigen::ArrayXXd>();
            }
        }
        return finalValues;
    }

    /// \brief get back local grid for each regime associated to current step
    inline std::vector< std::shared_ptr<grid> >    getGridCurrentProc()const
    {
        return m_gridCurrentProc ;
    }
};
}
#endif
