
#ifndef SDDPLOCALCUT_H
#define SDDPLOCALCUT_H
#include <tuple>
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "reflow/regression/LocalRegression.h"
#include "reflow/sddp/SDDPACut.h"
#include "reflow/sddp/SDDPVisitedStates.h"
#include "reflow/sddp/SDDPCutBase.h"
#include "reflow/sddp/SDDPCutCommon.h"

/** \file SDDPLocalCut.h
 *  \brief Create SDDP Cuts for Local  Regressor
 *
 *  \author Xavier Warin
 */

namespace reflow
{
/// \class SDDPLocalCut SDDPLocalCut.h
/// Create SDDP cuts
/// the problem to be solve is at each stage is
/// \f{eqnarray*}{
///     Q_t(X_t,W_t) & =  & \min_{U_t} C_t^T U_t + \mathbb{E}( Q_{t+1}(X_{t+1},W_{t+1}) \\ 
///                  &    & X_{t+1} = E X_{t+1} + B(W_t) \\
///                  &    &    X_{min} \le X_{i+1} \le X_{max}
/// \f}
/// where $X_t$ is the state vector and  $W_t$ is a vector of uncertainty
class SDDPLocalCut : public SDDPCutBase, SDDPCutCommon
{
private :

    int m_date ; ///< date identifier
    std::shared_ptr<LocalRegression> m_regressor ; ///< regressor object
    std::vector< std::vector<  std::shared_ptr<SDDPACut> > > m_cuts; ///< For each mesh of conditional expectation , give a list of all cuts
    int m_sample ; ///< number of samples used for each particle



public :

    /// \brief Default constructor
    SDDPLocalCut();

    /// \brief Constructor
    /// \param p_date date identifier
    /// \param p_sample  number of sample for expectation
    /// \param p_regressor regressor
    SDDPLocalCut(const int   &p_date, const int &p_sample, std::shared_ptr<LocalRegression> p_regressor);

    /// \brief Constructor (used in forward part)
    /// \param p_date date identifier
    /// \param p_regressor regressor
    SDDPLocalCut(const int   &p_date, std::shared_ptr<LocalRegression> p_regressor);


    /// \brief create a vector of (stocks, particle) for LP to solve
    /// \param   p_states   visited states object
    /// \return  a vector  giving the state, the particle used for the LP, the mesh number associated
    std::vector< std::tuple< std::shared_ptr<Eigen::ArrayXd>, int, int >  > createVectorStatesParticle(const SDDPVisitedStates &p_states) const  ;

    /// \brief create cuts using result of all  the LP solved and store them adding them to an archive
    /// \param p_cutPerSim      cuts per simulation
    ///         - first dimension 1 + size of state X
    ///         - second dimension if the number of simulations
    /// \param p_states             visited states object
    /// \param p_vectorOfLp         vector of LP corresponding to cuts associated to p_visitedStates    : for each member of p_vectorOfLp, m_sample are generated in  p_cutPerSim
    /// \param p_ar                 binary archive used to store additional cuts
    /// \param p_world              MPI communicator
    void createAndStoreCuts(const Eigen::ArrayXXd &p_cutPerSim, const SDDPVisitedStates &p_states,
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
#ifdef USE_MPI
        loadCutsByName(p_ar, "CutMesh", m_regressor->getNbMeshTotal(), m_date, m_cuts, p_world);
#else
        loadCutsByName(p_ar, "CutMesh", m_regressor->getNbMeshTotal(), m_date, m_cuts);
#endif
    }

    /// \brief get back all cuts associated to a mesh
    inline const std::vector< std::shared_ptr< SDDPACut > >   &getCutsForAMesh(const int &p_mesh) const
    {
        return m_cuts[p_mesh];
    }

    /// \brief get back members
    ///@{
    inline std::shared_ptr<LocalRegression>  getRegressor() const
    {
        return m_regressor ;
    }
    inline const std::vector< std::vector<  std::shared_ptr<SDDPACut> > > &getCuts() const
    {
        return m_cuts;
    }
    inline int  getSample() const
    {
        return m_sample;
    }
    inline int getUncertaintyDimension() const
    {
        return m_regressor-> getDimension();
    }
    ///@}

    /// \brief get back all the cuts associated to a particle number (state size by the number of cuts)
    /// \param p_isim  particle number
    Eigen::ArrayXXd  getCutsAssociatedToTheParticle(int p_isim) const;
    /// \brief get back all the cuts to a given particle  (state size by the number of cuts)
    /// \param p_aParticle  a particle
    Eigen::ArrayXXd  getCutsAssociatedToAParticle(const Eigen::ArrayXd &p_aParticle) const;

};
}
#endif
