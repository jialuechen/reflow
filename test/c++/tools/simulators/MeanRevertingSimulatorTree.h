
#ifndef MEANREVERTINGSIMULATORTREE_H
#define MEANREVERTINGSIMULATORTREE_H
#include <memory>
#include <boost/random.hpp>
#include "reflow/core/utils/constant.h"
#include "reflow/dp/SimulatorDPBase.h"
#include "reflow/sddp/SimulatorSDDPBaseTree.h"
#include "reflow/dp/SimulatorDPBaseTree.h"


/* \file MeanRevertingSimulatorTree.h
 * \brief Simulate a future deformation with
 *        a tree method. We suppose here that a  the number of uncertainties is equal to one.
 *        The tree describes the evolution of
 *        \f$ dY_t  =  -  a Y_t dt + \sigma dW_t \$
 *        and the future is
 *        \f$ dF(t,T) = F(t,T) ( e^{-a (T-t)} \sigma dW_t) \f$
 *        Such that
 *        \f$
 *             F(t,T)=  F(0,T) e^{ -\frac{1}{2} \frac{\sigma^2}{2a}(e^{-2a(T-t)}-e^{-2aT}) +  e^{-a (T-t)} Y_t}
 *        \f$
 *
 * \author Xavier Warin
 */


/// \class MeanRevertingSimulatorTree MeanRevertingSimulatorTree.h
/// Ornstein Uhlenbeck simulator with a tree
template< class Curve>
class MeanRevertingSimulatorTree: public reflow::SimulatorSDDPBaseTree
{
protected :
    double m_mr ; ///<  mean reverting
    double m_sigma ; ///< Volatility \f$\sigma\f$
    double m_trend ; ///< store  \f$ sigma^2/(2a) (1-exp(-2a t)) \f$
    std::shared_ptr< Curve> m_curve; ///< Future curve at initial date (0)
    bool m_bForward ; ///< true if it is a forward simulator
    boost::mt19937 m_generator;  ///< Boost random generator
    boost::random::uniform_real_distribution<double> m_uniformDistrib; ///< Uniform distribution
    boost::variate_generator<boost::mt19937 &, boost::random::uniform_real_distribution<double> > m_uniformRand ; ///< uniform generator
    int  m_nbSimul ; ///< number of simulations in forward
    Eigen::ArrayXi m_nodePos; ///< index in node position in forward
    double m_curveCurrent; ///< store curve at current date

    /// \brief Actualize trend
    inline void actualizeTrend()
    {
        m_trend =  0.5 * pow(m_sigma, 2.) / (2 * m_mr) * (1 - exp(-2 * m_mr * m_dates(m_idateCur)));
        m_curveCurrent = m_curve->get(m_dates(m_idateCur));
    }


public:

    /// \brief Constructor for backward simulator
    /// \param  p_binForTree  Geners archive to store the tree ( dates,   nodes in the node, probability transition)
    /// \param  p_curve   Initial forward curve
    /// \param  p_sigma   Volatility of each factor
    /// \param  p_mr      Mean reverting per factor
    MeanRevertingSimulatorTree(const std::shared_ptr<gs::BinaryFileArchive>   &p_binForTree,
                               const std::shared_ptr<Curve> &p_curve,
                               const double  &p_sigma,
                               const double  &p_mr):
        SimulatorSDDPBaseTree(p_binForTree), m_mr(p_mr), m_sigma(p_sigma), m_curve(p_curve), m_bForward(false),
        m_generator(), m_uniformDistrib(), m_uniformRand(m_generator, m_uniformDistrib)
    {
        updateDateIndex(m_dates.size() - 1);
    }

    /// \brief Constructor for forward  simulator
    /// \param  p_binForTree   Geners archive to store the tree ( dates,   nodes in the node, probability transition)
    /// \param  p_curve        Initial forward curve
    /// \param  p_sigma        Volatility of each factor
    /// \param  p_mr           Mean reverting per factor
    /// \param  p_nbSimul      Number of simulation used in SDDP forward
    MeanRevertingSimulatorTree(const std::shared_ptr<gs::BinaryFileArchive>   &p_binForTree,
                               const std::shared_ptr<Curve> &p_curve,
                               const double  &p_sigma,
                               const double    &p_mr, const int &p_nbSimul):
        SimulatorSDDPBaseTree(p_binForTree), m_mr(p_mr), m_sigma(p_sigma), m_curve(p_curve), m_bForward(true),
        m_generator(), m_uniformDistrib(), m_uniformRand(m_generator, m_uniformDistrib), m_nbSimul(p_nbSimul), m_nodePos(Eigen::ArrayXi::Zero(p_nbSimul))
    {
        updateDateIndex(0);
    }

    /// \brief Update the simulator for the date :
    /// \param p_idateCurr   index in date array
    void updateDateIndex(const int &p_idateCur)
    {
        SimulatorSDDPBaseTree::updateDateIndex(p_idateCur);
        actualizeTrend();
    }


    /// \brief get node associated to a simulation
    inline int getNodeAssociatedToSim(const int &p_isim) const
    {
        return  m_nodePos(p_isim);
    }

    /// \brief Get node number associated to a node
    /// \param p_nodeIndex  index of the node
    inline  Eigen::ArrayXd  getValueAssociatedToNode(const int &p_nodeIndex) const
    {
        return m_nodesCurr.col(p_nodeIndex);
    }

    ///
    /// \brief   From one particle simulation for an  OU process, get spot price
    ///          For the tree method it is the value at a node
    /// \param   p_oneParticle  One particle
    /// \return  spot value
    inline double fromOneParticleToSpot(const Eigen::VectorXd   &p_oneParticle)  const
    {
        return m_curveCurrent * exp(-m_trend +  p_oneParticle(0));
    }


    /// \brief a step forward for simulations
    void  stepForward()
    {
        // new nodes reached
        for (int inode = 0; inode < m_nodePos.size(); ++ inode)
        {
            double sample = m_uniformRand();
            m_nodePos(inode) = getNodeReachedInForward(m_nodePos(inode), sample);
        }
        updateDateIndex(m_idateCur + 1);
    }

    /// \brief a step forward for simulations
    void  stepBackward()
    {
        updateDateIndex(m_idateCur - 1);
    }

    ///
    /// \brief   For a node number, give the spot  value
    /// \param   p_iNode   node number
    /// \return  spot value
    inline double fromOneNodeToSpot(const int     &p_iNode)  const
    {
        return m_curveCurrent * exp(-m_trend +  m_nodesCurr(0, p_iNode));
    }

    /// \brief get all spot values from node values
    inline Eigen::ArrayXd  getSpotValues()const
    {
        return m_curveCurrent * (-m_trend +  m_nodesCurr.row(0)).exp().transpose();
    }

    /// \brief get back spot for nodes
    Eigen::ArrayXd getSpotAllNode() const
    {
        Eigen::ArrayXd spot(m_nodesCurr.cols());
        for (int i = 0; i < m_nodesCurr.cols(); ++i)
            spot(i) = m_curveCurrent * exp(-m_trend + m_nodesCurr(0, i));
        return spot;
    }


    /// \brief Analytical spot expectaton
    double getAnalyticalESpot() const
    {
        return m_curveCurrent;
    }

    /// \brief get analytical expectaton of spot^2
    double getAnalyticalESpot2() const
    {
        return m_curveCurrent * m_curveCurrent * exp(2 * m_trend);
    }

    ///@{
    /// Get back attribute
    inline double getSigma() const
    {
        return m_sigma ;
    }
    inline double getMr() const
    {
        return  m_mr    ;
    }
    inline int getNbSimul() const
    {
        return m_nbSimul;
    }
    inline int  getNbSample() const
    {
        return 1;
    }

    ///@}


    /// \brief forward or backward update for time
    inline void resetTime()
    {
        if (m_bForward)
        {
            updateDateIndex(0);
            m_nodePos = Eigen::ArrayXi::Zero(m_nbSimul);
        }
        else
        {
            updateDateIndex(m_dates.size() - 1);
        }
    }

    /// \brief  update the number of simulations (forward only)
    /// \param p_nbSimul  Number of simulations to update
    inline void updateSimulationNumberAndResetTime(const int &p_nbSimul)
    {
        assert(m_bForward);
        m_nbSimul = p_nbSimul;
        updateDateIndex(0);
        m_nodePos = Eigen::ArrayXi::Zero(p_nbSimul);
    }
};
#endif /* MEANREVERTINGSIMULATORTREE_H */
