
#ifndef SDDPCUTBASE_H
#define SDDPCUTBASE_H
#include <vector>
#include <tuple>
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "reflow/sddp/SDDPVisitedStates.h"
#include "reflow/sddp/SDDPACut.h"
#include "reflow/sddp/SDDPCutOptBase.h"

/**  \file SDDPCutBase.h
 *   \brief Abstract class for cut by regressions
 *   \author Xavier Warin
 */

namespace reflow
{
/// \class SDDPCutBase SDDPCutBase.h
/// Abstract class
class SDDPCutBase: public SDDPCutOptBase
{
public :

    virtual ~SDDPCutBase() {}

    /// \brief Load already calculated cuts
    /// \param p_ar   archive to load cuts
    /// \param p_world  MPI communicator
    virtual void loadCuts(const std::shared_ptr<gs::BinaryFileArchive> &p_ar
#ifdef USE_MPI
                          , const boost::mpi::communicator &p_world
#endif
                         ) = 0;

    /// \brief create cuts using result of all  LP solved and store them adding them to an archive
    /// \param p_cutPerSim      cuts per simulation
    ///         - first dimension 1 + size of state X
    ///         - second dimension if the number of simulations
    /// \param p_states             visited states object
    /// \param p_vectorOfLp         vector of LP corresponding to cuts associated to p_visitedStates    : for each member of p_vectorOfLp, m_sample are generated in  p_cutPerSim
    /// \param p_ar                 binary archive used to store additional cuts
    /// \param p_world              MPI communicator
    virtual void createAndStoreCuts(const Eigen::ArrayXXd &p_cutPerSim, const SDDPVisitedStates &p_states,
                                    const std::vector< std::tuple< std::shared_ptr<Eigen::ArrayXd>, int, int >  > &p_vectorOfLp,
                                    const std::shared_ptr<gs::BinaryFileArchive>   &p_ar
#ifdef USE_MPI
                                    , const boost::mpi::communicator &p_world
#endif
                                   ) = 0;

    /// \brief create a vector of (stocks, particle) for LP to solve
    /// \param   p_states   visited states object
    /// \return  a vector  giving the state, the particle used for the LP, the mesh number associated
    virtual std::vector< std::tuple< std::shared_ptr<Eigen::ArrayXd>, int, int >  > createVectorStatesParticle(const SDDPVisitedStates &p_states) const  = 0;

    /// \brief get back members
    ///@{
    virtual const std::vector< std::vector<  std::shared_ptr<SDDPACut> > > &getCuts() const = 0 ;
    virtual int  getSample() const = 0 ;
    virtual int getUncertaintyDimension() const  = 0 ;
    ///@}

};
}
#endif /* SDDPCutBase.h */
