
#ifndef SIMULATORDPBASETREE_H
#define SIMULATORDPBASETREE_H
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"

/* \file SimulatorDPBaseTree.h
 * \brief Abstract class for simulators for Dynamic Programming Programms with tree
 * \author Xavier Warin
 */

namespace libflow
{
/// \class SimulatorDPBaseTree SimulatorDPBaseTree.h
/// Abstract class for simulator used in dynamic programming with trees
class SimulatorDPBaseTree
{
protected :

    std::shared_ptr<gs::BinaryFileArchive> m_binForTree ; ///< archive for tree
    Eigen::ArrayXd m_dates ; ///< list of dates in the archive
    int m_idateCur ; ///< current date index
    Eigen::ArrayXXd  m_nodesCurr ; ///< storing coordinates of the nodes at current date  (dim, nbnodes)
    Eigen::ArrayXXd  m_nodesNext; ///< storing coordinates of the nodes at next date (dim, nbnodes)
    std::vector<double>  m_proba ; ///<  value stores probability to go from on node   at index m_dateCurc to node  at next date m_dateNext.
    std::vector< std::vector< std::array<int, 2> > >  m_connected ; ///<for each node at current  date, give a list of connected nodes at next date and index in probability vector
    /// \brief  load a date
    void load(const int &p_idateCur);

public :

    /// \brief Constructor
    SimulatorDPBaseTree() {}

    /// \brief Constructor : use in backward
    /// \param  p_binforTree  binary geners archive  with structure
    ///         - dates      ->  eigen array of dates, size ndate
    ///         - nodes     ->  nDate array , each array containing nodes coordinates  with size  (ndim, nbNodes)
    ///         - proba      ->  probabilities to go from node to another from a date to the next date
    ///         - connected  -> connecton matrix for a node at current date to go to a node at next date
    ///
    SimulatorDPBaseTree(const std::shared_ptr<gs::BinaryFileArchive>   &p_binForTree);

    /// \brief Destructor
    virtual ~SimulatorDPBaseTree() {}

    /// \brief a step forward for simulations
    virtual void  stepForward() = 0;

    /// \brief sample one simulation in forward mode
    /// \param  p_nodeStart starting node
    /// \param  p_randUni  uniform random in [0,1]
    /// \return node reached
    int getNodeReachedInForward(const int &p_nodeStart, const double &p_randUni) const ;


    /// \brief a step backward for simulations
    virtual void  stepBackward() = 0;
    /// \brief get back dimension of the problem
    virtual int getDimension() const
    {
        return m_nodesCurr.rows();
    }
    /// \brief get the number of steps
    virtual  int getNbStep() const
    {
        return  m_dates.size() - 1;
    }
    /// \brief Number of nodes at current date
    virtual int getNbNodes() const
    {
        return m_nodesCurr.cols();
    }
    /// \brief Number of nodes at next date
    virtual int getNbNodesNext() const
    {
        return m_nodesNext.cols();
    }

    /// \brief get back dates
    inline Eigen::ArrayXd getDates() const
    {
        return m_dates;
    }

    /// \brief get back the last date index
    inline int getBackLastDateIndex() const
    {
        return m_dates.size() - 1;
    }

    /// \brief get back connection matrix :for each node at current date, give the node connected
    std::vector< std::vector< std::array<int, 2 > > >  getConnected() const
    {
        return m_connected ;
    }

    /// \brief get back probabilities
    inline std::vector< double > getProba() const
    {
        return m_proba;
    }

    /// \brief get current nodes
    inline Eigen::ArrayXXd getNodes()  const
    {
        return m_nodesCurr ;
    }

    /// \brief get  nodes at next date
    inline Eigen::ArrayXXd getNodesNext()  const
    {
        return m_nodesNext ;
    }

    /// \brief Get number of simulations used in forward
    virtual inline int getNbSimul() const = 0;


    /// \brief Get node number associated to a node
    /// \param p_nodeIndex  index of the node
    virtual Eigen::ArrayXd  getValueAssociatedToNode(const int &p_nodeIndex) const = 0;

    /// \brief get node associated to a simulation
    /// \param p_isim  simulation number
    /// \return number of the node associated to a simulation
    virtual int getNodeAssociatedToSim(const int &p_isim) const  = 0;
};
}
#endif /* SIMULATORDPBASETREE_H */
