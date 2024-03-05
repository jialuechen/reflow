
#ifndef SDDPVISITEDSTATE_H
#define SDDPVISITEDSTATE_H
#include <memory>
#include <Eigen/Dense>
#include "libflow/regression/LocalRegression.h"
#include "libflow/sddp/SDDPVisitedStatesBase.h"


/**  \file SDDPVisitedStates.h
 *   \brief Storing visited states during simulation
 *   \author Xavier Warin
 */
namespace libflow
{

/// \class SDDPVisitedStates
/// Permits to store visited cuts in SDDP forward resolution
class SDDPVisitedStates : public SDDPVisitedStatesBase
{

public :

    /// \brief Default constructor
    SDDPVisitedStates();

    /// \brief Constructor one
    SDDPVisitedStates(const int &p_nbNode);

    /// \brief Second constructor with all states
    SDDPVisitedStates(const std::vector< std::vector< int> >   &p_meshToState, const std::vector< std::shared_ptr< Eigen::ArrayXd >  > &p_stateVisited, const std::vector<int> &p_associatedMesh) ;

    /// \brief add a state
    /// \param  p_state state to add
    /// \param  p_particle particle  used for conditional cut
    /// \param  p_regressor     regressor used
    void addVisitedState(const std::shared_ptr< Eigen::ArrayXd > &p_state, const Eigen::ArrayXd &p_particle, const LocalRegression   &p_regressor);

    /// \brief add a state for particles
    /// \param  p_state state to add
    /// \param  p_regressor     regressor used
    void addVisitedStateForAll(const std::shared_ptr< Eigen::ArrayXd > &p_state,  const LocalRegression   &p_regressor);

};
}
#endif /* SDDPVISITEDSTATES_H */
