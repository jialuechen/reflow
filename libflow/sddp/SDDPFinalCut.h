#ifndef SDDPFINALCUT_H
#define  SDDPFINALCUT_H
// Copyright (C) 2016 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <Eigen/Dense>
#include "libflow/sddp/SDDPCutBase.h"

/** \file SDDPFinalCut.h
 * \brief Create final cuts for SDDP
 * \author Xavier Warin
 */

namespace libflow
{

/// \class SDDPFinalCut SDDPFinalCut.h
/// Defines cuts independent of uncertainties for final valorisation
class SDDPFinalCut : public  SDDPCutBase
{

private :
    Eigen::ArrayXXd m_cuts; ///< array of cuts (a cut is a column)
    std::vector< std::vector<  std::shared_ptr<SDDPACut> > >  m_useless;

public :

    /// \brief defaut constructor
    SDDPFinalCut() {}

    /// \brief Constructor
    /// \param  p_cuts  all cuts for final value
    SDDPFinalCut(const Eigen::ArrayXXd &p_cuts): m_cuts(p_cuts) {}

    /// \brief get back all the cuts associated to a particle number (state size by the number of cuts)
    Eigen::ArrayXXd  getCutsAssociatedToTheParticle(int p_isim) const
    {
        return m_cuts;
    }
    /// \brief get back all the cuts to a given particle  (state size by the number of cuts)
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

    void createAndStoreCuts(const Eigen::ArrayXXd &p_cutPerSim, const SDDPVisitedStates &p_states,
                            const std::vector< std::tuple< std::shared_ptr<Eigen::ArrayXd>, int, int >  > &p_vectorOfLp,
                            const std::shared_ptr<gs::BinaryFileArchive>   &p_ar
#ifdef USE_MPI
                            , const boost::mpi::communicator &p_world
#endif
                           ) {}
    std::vector< std::tuple< std::shared_ptr<Eigen::ArrayXd>, int, int >  > createVectorStatesParticle(const SDDPVisitedStates &p_states) const
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
    inline int getUncertaintyDimension() const
    {
        return 0;
    }
    //@}
};
}
#endif /* SDDPFINALCUT_H */
