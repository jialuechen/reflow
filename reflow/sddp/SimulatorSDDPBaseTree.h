
#ifndef SIMULATORSDDPBASETREE_H
#define SIMULATORSDDPBASETREE_H
#include <memory>
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "reflow/core/utils/comparisonUtils.h"
#include "reflow/dp/SimulatorDPBaseTree.h"
#include "reflow/sddp/SimulatorSDDPBase.h"

/* \file SimulatorBaseTree.h
 * \brief Base class for simulators for SDDP method with uncertainties breaking concavity/convexity in a tree
* \author Xavier Warin
*/
namespace reflow
{
/// \class SimulatorSDDPBaseTree SimulatorSDDPBaseTree.h
/// Base class for simulators used for SDDP  with uncertainties breaking concavity/convexity in a Tree
class SimulatorSDDPBaseTree : public SimulatorSDDPBase, public SimulatorDPBaseTree
{

public :

    /// \brief Constructor
    /// \param  p_binforTree  binary geners archive  with structure
    ///         - dates      ->  eigen array of dates, size ndate
    ///         - nodes     ->  nDate array , each array containing nodes coordinates  with size  (ndim, nbNodes)
    ///         - proba      -> for a point i at a given date and a point j at next date , prob(i,j) gives the probability to go from node i to node j.
    ///
    SimulatorSDDPBaseTree(const std::shared_ptr<gs::BinaryFileArchive>   &p_binForTree): SimulatorDPBaseTree(p_binForTree) {}


    /// \brief Destructor
    virtual ~SimulatorSDDPBaseTree() {}

    /// \brief
    /// \brief Get back the number of particles
    virtual int getNbSimul() const
    {
        return 0;
    }

    /// \brief Get back the number of sample used (simulation at each time step , these simulations are independent of the state)
    virtual int getNbSample() const
    {
        return 0 ;
    }


    /// \brief get one simulation
    /// \param p_isim  simulation number
    /// \return the particle associated to p_isim
    virtual Eigen::VectorXd getOneParticle(const int &p_isim) const
    {
        return  m_nodesCurr.col(getNodeAssociatedToSim(p_isim));
    }

    /// \brief get  current Markov state
    virtual  Eigen::MatrixXd getParticles() const
    {
        return Eigen::MatrixXd();
    }

    /// \brief Reset the simulator (to use it again for another SDDP sweep)
    virtual  void resetTime() {}

    /// \brief in simulation  part of SDDP reset  time  and reinitialize uncertainties
    /// \param p_nbSimul  Number of simulations to update
    virtual  void updateSimulationNumberAndResetTime(const int &p_nbSimul) {}

    /// \brief Update the simulator for the date :
    /// \param p_idateCurr   index in date array
    virtual void updateDateIndex(const int &p_idateCur)
    {
        load(p_idateCur);
    }


};
}

#endif /* SIMULATORSDDPBASETREE_H */
