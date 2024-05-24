
#ifndef SDDPCUTOPTBASE_H
#define SDDPCUTOPTBASE_H
#include <Eigen/Dense>

/**  \file SDDPCutOptBase.h
 *   \brief Abstract class for all cuts  : regressions and trees
 *          Defines only methods used in  optimization  in a transition.
 *   \author Xavier Warin
 */
namespace reflow
{
/// \class SDDPCutOptBase SDDPCutOptBase.h
/// Abstract class
class SDDPCutOptBase
{
public :

    virtual ~SDDPCutOptBase() {}

    /// \brief Get back all the cuts associated to a particle number (state size by the number of cuts)
    /// \param p_isim  particle number  (or node number)
    virtual Eigen::ArrayXXd  getCutsAssociatedToTheParticle(int p_isim) const = 0;

    /// \brief Get back all the cuts to a given particle  (state size by the number of cuts)
    /// \param p_aParticle  a particle in regression or the coordinates of a node in the tree
    virtual Eigen::ArrayXXd  getCutsAssociatedToAParticle(const Eigen::ArrayXd &p_aParticle) const = 0;
};
}
#endif /* SDDPCutOptBase.h */
