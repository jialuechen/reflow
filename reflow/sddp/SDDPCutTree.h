
#ifndef SDDPCUTTREE_H
#define SDDPCUTTREE_H
#include <tuple>
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "reflow/tree/Tree.h"
#include "reflow/sddp/SDDPACut.h"
#include "reflow/sddp/SDDPCutCommon.h"
#include "reflow/sddp/SDDPCutBaseTree.h"
#include "reflow/sddp/SDDPVisitedStatesTree.h"

/** \file SDDPCutTree.h
 *  \brief Create SDDP Cuts for tree methods
 *
 *  \author Xavier Warin
 */

namespace reflow
{
/// \class SDDPCutTree SDDPCutTree.h
/// Create SDDP cuts for the tree approach
/// the problem to be solve is at each stage is
/// \f{eqnarray*}{
///     Q_t(X_t,W_t) & =  & \min_{U_t} C_t^T U_t + \mathbb{E}( Q_{t+1}(X_{t+1},W_{t+1}) \\ 
///                  &    & X_{t+1} = E X_{t+1} + B(W_t) \\
///                  &    &    X_{min} \le X_{i+1} \le X_{max}
/// \f}
/// where $X_t$ is the state vector and  $W_t$ is a vector of uncertainty
/// at each node of the tree
class SDDPCutTree : public SDDPCutBaseTree, SDDPCutCommon
{
private :

    int m_date ; ///< date identifier
    std::vector<std::vector< std::shared_ptr<SDDPACut> > >  m_cuts; ///<vector af cuts : for each node in the tree, list of all cut associated
    Tree m_tree ; // to calculate conditional expectation
    Eigen::ArrayXXd   m_nodes ; ///< nodes coordinates in tree
    int m_sample ; ///< number of samples used for each particle for Monte Carlo

public :

    /// \brief Default constructor
    SDDPCutTree();

    /// \brief Constructor
    /// \param p_date       date identifier
    /// \param p_sample     number of sample for expectation
    /// \param p_proba      probability between nodes on two dates
    /// \param p_connected  connection between nodes
    /// \param p_nodes      nodes coordinates in tree
    SDDPCutTree(const int   &p_date, const int &p_sample, const std::vector<double>  &p_proba, const std::vector< std::vector< std::array<int, 2> > > &p_connected, const Eigen::ArrayXXd &p_nodes);

    /// \brief Constructor (used in forward part)
    /// \param p_date date identifier
    /// \param p_nodes      nodes coordinates in tree
    SDDPCutTree(const int   &p_date, const Eigen::ArrayXXd &p_nodes);

    /// \brief create a vector of (stocks, particle) for LP to solve
    /// \param   p_states   visited states object
    /// \return  a vector   giving the state , the arrival node, starting node
    std::vector< std::tuple< std::shared_ptr<Eigen::ArrayXd>, int, int >  > createVectorStatesParticle(const SDDPVisitedStatesTree &p_states) const  ;

    /// \brief create cuts using result of all  the LP solved and store them adding them to an archive
    /// \param p_cutPerSim      cuts per simulation
    ///         - first dimension  1 + size of state X
    ///         - second dimension the number of simulations
    /// \param p_states             visited states object associated to each simulation in p_cutPerSim
    /// \param p_vectorOfLp         vector of LP corresponding to cuts associated to p_visitedStates    : for each member of p_vectorOfLp, m_sample are generated in  p_cutPerSim
    /// \param p_ar                 binary archive used to store additional cuts
    /// \param p_world  MPI communicator
    void createAndStoreCuts(const Eigen::ArrayXXd &p_cutPerSim, const SDDPVisitedStatesTree &p_states,
                            const std::vector< std::tuple< std::shared_ptr<Eigen::ArrayXd>, int, int >  > &p_vectorOfLp,
                            const std::shared_ptr<gs::BinaryFileArchive>   &p_ar
#ifdef USE_MPI
                            , const boost::mpi::communicator &p_world
#endif
                           );

    /// \brief Load already calculated cuts
    /// \param p_ar   archive to load cuts
    /// \param p_world  MPI communicator
    inline void loadCuts(const std::shared_ptr<gs::BinaryFileArchive> &p_ar
#ifdef USE_MPI
                         , const boost::mpi::communicator &p_world
#endif
                        )
    {
        // number of node in tree at current date
#ifdef USE_MPI
        loadCutsByName(p_ar, "CutNode", m_nodes.cols(), m_date, m_cuts, p_world);
#else
        loadCutsByName(p_ar, "CutNode", m_nodes.cols(), m_date, m_cuts);
#endif
    }

    /// \brief get back all cuts associated to a node  in the tree
    inline const std::vector< std::shared_ptr< SDDPACut > >   &getCutsForAMesh(const int &p_mesh) const
    {
        return m_cuts[p_mesh];
    }

    inline const std::vector< std::vector<  std::shared_ptr<SDDPACut> > > &getCuts() const
    {
        return m_cuts;
    }
    inline int  getSample() const
    {
        return m_sample;
    }

    ///@}

    /// \brief get back all the cuts associated to a node number (state size by the number of cuts)
    /// \param p_node  node  number
    Eigen::ArrayXXd  getCutsAssociatedToTheParticle(int p_node) const;

    /// \brief get back all the cuts associated to a point (node in the tree)
    /// \param p_aParticle  a particle cooresponding to a node coordinates in the tree
    Eigen::ArrayXXd  getCutsAssociatedToAParticle(const Eigen::ArrayXd &p_aParticle) const;
};
}
#endif
