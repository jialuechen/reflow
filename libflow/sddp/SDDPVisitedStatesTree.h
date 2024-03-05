
#ifndef SDDPVISITEDSTATETREE_H
#define SDDPVISITEDSTATETREE_H
#include <memory>
#include <Eigen/Dense>
#include "libflow/sddp/SDDPVisitedStatesBase.h"


/**  \file SDDPVisitedStatesTree.h
 *   \brief Storing visited states during simulation with tree method in SDDP
 *   \author Xavier Warin
 */
namespace libflow
{

/// \class SDDPVisitedStatesTree
/// Permits to store visited cuts in SDDP forward resolution
/// These cuts are used in next backward resolutions
class SDDPVisitedStatesTree : public SDDPVisitedStatesBase
{

public :

    /// \brief Default constructor
    SDDPVisitedStatesTree();

    /// \brief Constructor one
    SDDPVisitedStatesTree(const int &p_nbNode);

    /// \brief Second constructor with all states
    SDDPVisitedStatesTree(const std::vector< std::vector< int> >   &p_meshToState, const std::vector< std::shared_ptr< Eigen::ArrayXd >  > &p_stateVisited, const std::vector<int> &p_associatedPoint) ;


    /// \brief add a state
    /// \param  p_state state to add
    /// \param  p_point number of the node in the tree at a given date where the state is added
    void addVisitedState(const std::shared_ptr< Eigen::ArrayXd > &p_state, const int   &p_nbNode);

    /// \brief add a state for all nodes
    /// \param  p_state state to add
    /// \param  p_nbNode number of nodes in the tree at a given date
    void addVisitedStateForAll(const std::shared_ptr< Eigen::ArrayXd > &p_state, const int &p_nbNode);

};
}
#endif /* SDDPVISITEDSTATESTREE_H */
