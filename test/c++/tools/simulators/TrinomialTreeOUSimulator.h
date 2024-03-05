
#ifndef TRINOMIALTREEOUSIMULATOR_H
#define TRINOMIALTREEOUSIMULATOR_H
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "geners/Record.hh"

/* \file TrinomialTreeSimulator.h
 *       Defines a trinomial tree for an Ornstein Uhlenbeck process
 *       \f$ dX_t =  - a X_t  dt + \sigma dW_t \f$
 *       Hull and White method
 *  \author Xavier Warin
 */

class TrinomialTreeOUSimulator
{
private :

    double m_mr ; ///< Mean reverting coefficient \$ a \$
    double m_sig ; ///< Volatility of the process
    Eigen::ArrayXd m_dates ; ///< dates for the tree
    Eigen::ArrayXd m_dx ; ///<  Size of the mesh in the tree at a given depth
    Eigen::ArrayXi m_nbNodeBelow0; ///< number of mesh with coordinates below 0
    std::vector <Eigen::ArrayXi> m_facing ; ///< For each date k  to a mesh i at date  k gives the "middle" mesh at date k+1
    std::vector< Eigen::ArrayXXd> m_proba; ///< For each date k gives the probability for a node i to go down middle and up  (size (3,nb node))

    /// \brief calculate prob along (k_1,..k_d) where k_i in [0,1, 2]
    /// \param  p_ipos         position in node number for given depth p_depthCur
    /// \param  p_depth        depth in tree
    /// \param  p_depthMax     maximal depth  : probability is calculated between node in depth  p_depth and p_depthMax
    /// \param  p_depthCur     current   depth
    /// \param  p_downMiddleUp  array of int with values 0, 1, 2 defining a path in the tree  (0: down, 1 middle and 2 up)
    /// \return  probablity value along path and arrival node
    std::pair<double, int>  probCal(const int &p_ipos, const  int &p_depth, const int &p_depthMax,  const int   &p_depthCur, const Eigen::ArrayXi &p_downMiddleUp) const ;

    /// \brief Get back mesh size
    /// \param p_depth  depth for meshing
    inline double getDx(const int &p_depth) const
    {
        return  m_dx(p_depth);
    }

    /// \brief Get facing node
    /// \param p_depth  depth for meshing
    /// \param p_inode  node number
    inline int facingNode(const int &p_depth, const int &p_inode) const
    {
        return  m_facing[p_depth](p_inode) ;
    }

public :

    /// \brief Constructor
    /// \param  p_mr    mean reverting
    /// \param  p_sig   volatility
    /// \param  p_dates dates for the tree
    TrinomialTreeOUSimulator(const double &p_mr, const double &p_sig, const Eigen::ArrayXd &p_dates);


    /// \brief Get back node coordinates at given time step index (dim=1, nbpoints)
    /// \param p_idate  index of the date in m_dates
    Eigen::ArrayXXd getPoints(const int &p_idate) const;

    /// \brief Get back probability between a given time step and a second one
    /// \param p_idateBeg index of the date in m_dates of starting pint
    /// \param p_idateEnd index of the date in m_dates of ending points
    Eigen::ArrayXXd getProbability(const int &p_idateBeg, const int &p_idateEnd) const ;

    /// \brief Dump
    /// \param p_name   archive name
    /// \param p_index  index in date to print in archive
    void dump(const std::string &p_name, const Eigen::ArrayXi &p_index);


    /// \brief calculate expectation on one time step
    /// \param  p_depth   index of the time step to calculate the conditional expectaton
    /// \param  p_inode   node value
    /// \param  p_value   values at each node of the tree
    double calculateStepCondExpectation(const int &p_depth, const int &p_inode, const Eigen::ArrayXd   &p_value) const;


    /// \brief calculate expectation
    /// \param  p_depth0  index of the time step to calculate the conditional expectaton
    /// \param  p_depth   current depth in tree
    /// \param  p_value   values at each node of the tree
    Eigen::ArrayXd calculateCondExpectation(const int &p_depth0, const int &p_depthLast, const Eigen::ArrayXd   &p_value) const;


    /// \brief calculate expectatiion
    /// \param  p_depth   current depth in tree
    /// \param  p_value   values at each node of the tree
    inline  double calculateExpectation(const int &p_depth, const Eigen::ArrayXd   &p_value) const
    {
        return calculateCondExpectation(0, p_depth,  p_value)(0);
    }
    /// \brief From a paprobility matrix, caculate a vector of probabiity with non 0 coefficients and conneciotn matrix
    /// \param p_proba probability matrix
    /// \return connection matrix and vector of probabilities
    std::pair< std::vector< std::vector< std::array<int, 2> > >, std::vector< double >  > calConnected(const Eigen::ArrayXXd &p_proba);
};

#endif
