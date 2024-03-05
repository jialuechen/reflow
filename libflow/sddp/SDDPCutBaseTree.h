
#ifndef SDDPCUTBASETREE_H
#define SDDPCUTBASETREE_H
#include <vector>
#include <tuple>
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "libflow/sddp/SDDPVisitedStatesTree.h"
#include "libflow/sddp/SDDPACut.h"
#include "libflow/sddp/SDDPCutOptBase.h"

/**  \file SDDPCutBaseTree.h
 *   \brief Abstract class for cuts for tree
 *   \author Xavier Warin
 */

namespace libflow
{
/// \class SDDPCutBaseTree SDDPCutBaseTree.h
/// Abstract class
class SDDPCutBaseTree: public SDDPCutOptBase
{
public :

    virtual ~SDDPCutBaseTree() {}

    /// \brief Load already calculated cuts
    /// \param p_ar   archive to load cuts
    virtual void loadCuts(const std::shared_ptr<gs::BinaryFileArchive> &p_ar
#ifdef USE_MPI
                          , const boost::mpi::communicator &p_world
#endif
                         ) = 0;

    /// \brief create cuts using result of all  LP solved and store them adding them to an archive
    /// \param p_cutPerSim      cuts per simulation
    ///         - first dimension 1 + size of state X
    ///         - second dimension the number of simulations
    /// \param p_states            visited states object
    /// \param p_vectorOfLp        vector of LP corresponding to cuts associated to p_visitedStates    : for each member of p_vectorOfLp, m_sample are generated in  p_cutPerSim
    /// \param p_ar                  binary archive used to store additional cuts
    virtual void createAndStoreCuts(const Eigen::ArrayXXd &p_cutPerSim, const SDDPVisitedStatesTree &p_states,
                                    const std::vector< std::tuple< std::shared_ptr<Eigen::ArrayXd>, int, int >  > &p_vectorOfLp,
                                    const std::shared_ptr<gs::BinaryFileArchive>   &p_ar
#ifdef USE_MPI
                                    , const boost::mpi::communicator &p_world
#endif
                                   ) = 0;

    /// \brief create a vector of (stocks, particle) for LP to solve
    /// \param   p_states   visited states object
    /// \return  a vector  giving the state, the node reached at next date and  used for the LP, the current node number associated
    virtual std::vector< std::tuple< std::shared_ptr<Eigen::ArrayXd>, int, int >  > createVectorStatesParticle(const SDDPVisitedStatesTree &p_states) const  = 0;

    /// \brief get back members
    ///@{
    virtual const std::vector< std::vector<  std::shared_ptr<SDDPACut> > > &getCuts() const = 0 ;
    virtual int  getSample() const = 0 ;
    ///@}

};
}
#endif /* SDDPCutBaseTree.h */
