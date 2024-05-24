#ifndef SDDPFINALCUTTREE_H
#define  SDDPFINALCUTTREE_H

#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <Eigen/Dense>
#include "reflow/sddp/SDDPCutBaseTree.h"

/** \file SDDPFinalCutTree.h
 * \brief Create final cuts for SDDP with trees
 * \author Xavier Warin
 */

namespace reflow
{

/// \class SDDPFinalCutTree SDDPFinalCutTree.h
/// Defines cuts independent of uncertainties for final valorisation for tree methods
class SDDPFinalCutTree : public  SDDPCutBaseTree
{

private :
    Eigen::ArrayXXd m_cuts; ///< array of cuts (a cut is a column)
    std::vector< std::vector<  std::shared_ptr<SDDPACut> > >  m_useless;

public :

    /// \brief defaut constructor
    SDDPFinalCutTree() {}

    /// \brief Constructor
    /// \param  p_cuts  all cuts for final value
    SDDPFinalCutTree(const Eigen::ArrayXXd &p_cuts): m_cuts(p_cuts) {}

    /// \brief get back all the cuts associated to a node number (state size by the number of cuts)
    Eigen::ArrayXXd  getCutsAssociatedToTheParticle(int) const
    {
        return m_cuts;
    }
    /// \brief get back all the cuts to a given particle (corresponding to a node)   (state size by the number of cuts)
    Eigen::ArrayXXd  getCutsAssociatedToAParticle(const Eigen::ArrayXd &p_aParticle) const
    {
        return m_cuts;
    }
    /// \brief Useless but for genericity
    //@{
    void loadCuts(const std::shared_ptr< gs::BinaryFileArchive> &p_ar
#ifdef USE_MPI
                  , const boost::mpi::communicator &p_world
#endif
                 ) {}
    void createAndStoreCuts(const Eigen::ArrayXXd &p_cutPerSim, const SDDPVisitedStatesTree &p_states,
                            const std::vector< std::tuple< std::shared_ptr<Eigen::ArrayXd>, int, int >  > &p_vectorOfLp,
                            const std::shared_ptr<gs::BinaryFileArchive>   &p_ar
#ifdef USE_MPI
                            , const boost::mpi::communicator &p_world
#endif
                           ) {}
    std::vector< std::tuple< std::shared_ptr<Eigen::ArrayXd>, int, int >  > createVectorStatesParticle(const SDDPVisitedStatesTree &p_states) const
    {
        return std::vector< std::tuple< std::shared_ptr<Eigen::ArrayXd>, int, int >  >();
    }
    inline const std::vector< std::vector<  std::shared_ptr<SDDPACut> > > &getCuts() const
    {
        return m_useless;
    }
    inline int  getSample() const
    {
        return 0;
    }
    //@}
};
}
#endif /* SDDPFINALCUTTREE_H */
