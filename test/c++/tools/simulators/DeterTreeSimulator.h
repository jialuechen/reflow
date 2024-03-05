
#ifndef DETERTREESIMULATOR_H
#define DETERTREESIMULATOR_H
#include <vector>
#include <array>
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "geners/Record.hh"
#include "geners/vectorIO.hh"
#include "geners/arrayIO.hh"

/* \file DeterTreeSimulator.h
 *       Defines a  tree for with only one node and zero value
 *       For test purpose
 *  \author Xavier Warin
 */

class DeterTreeSimulator
{
private :

    Eigen::ArrayXd m_dates ; ///< dates for the tree
    std::vector< Eigen::ArrayXXd> m_proba; ///<  Equal to 1 fo all dates


public :

    /// \brief Constructor
    /// p_dates \param dates for the tree
    DeterTreeSimulator(const Eigen::ArrayXd &p_dates) : m_dates(p_dates)
    {
        m_proba.reserve(p_dates.size() - 1);
        for (int it = 1; it < p_dates.size() - 1; ++it)
        {
            Eigen::ArrayXXd proba = Eigen::ArrayXXd::Constant(1, 1, 1.);
            m_proba.push_back(proba);
        }
    }


    /// \brief Get back node coordinates at given time step index (dim=1, nbpoints)
    /// \param p_idate  index of the date in m_dates
    Eigen::ArrayXXd getPoints(const int &p_idate) const
    {
        Eigen::ArrayXXd ret = Eigen::ArrayXXd::Zero(1, 1);
        return ret;
    }

    /// \brief Get back probability betwwen a given time step and a second one
    /// \param p_idateBeg index of the date in m_dates of starting pint
    /// \param p_idateEnd index of the date in m_dates of ending points
    Eigen::ArrayXXd getProbability(const int &p_idateBeg, const int &p_idateEnd) const
    {
        Eigen::ArrayXXd ret = Eigen::ArrayXXd::Constant(1, 1, 1.);
        return ret;
    }

    /// \brief Dump
    /// \param p_name   archive name
    /// \param p_index  index in date to print in archive
    void dump(const std::string &p_name, const Eigen::ArrayXi &p_index)
    {
        gs::BinaryFileArchive binArxiv(p_name.c_str(), "w");
        Eigen::ArrayXd ddates(p_index.size());
        for (int i = 0; i < p_index.size(); ++i)
            ddates(i) = m_dates(p_index(i));
        binArxiv << gs::Record(ddates, "dates", "");
        for (int i = 0 ;  i < p_index.size(); ++i)
        {
            Eigen::ArrayXXd points = Eigen::ArrayXXd::Zero(1, 1);
            binArxiv << gs::Record(points, "points", "");
        }
        for (int i = 0 ;  i < p_index.size() - 1; ++i)
        {
            std::vector<double> proba(1);
            proba[0] = 1.;
            binArxiv << gs::Record(proba, "proba", "");
            std::vector< std::vector< std::array<int, 2> > >  connect(1);
            connect[0].resize(1);
            std::array<int, 2> two{{0, 0}};
            connect[0][0] = two;
            binArxiv << gs::Record(connect, "connection", "");
        }

    }


    /// \brief calculate expectation on one time step : here only one node...
    /// \param  p_depth   index of the time step to calculate the conditional expectaton
    /// \param  p_inode   node value
    /// \param  p_value   values at each node of the tree
    double calculateStepCondExpectation(const int &p_depth, const int &p_inode, const Eigen::ArrayXd   &p_value) const
    {
        return p_value(0);
    }


    /// \brief calculate expectation
    /// \param  p_depth0  index of the time step to calculate the conditional expectaton
    /// \param  p_depth   current depth in tree
    /// \param  p_value   values at each node of the tree
    Eigen::ArrayXd calculateCondExpectation(const int &p_depth0, const int &p_depthLast, const Eigen::ArrayXd   &p_value) const
    {
        return p_value;
    }


    /// \brief calculate expectatiion
    /// \param  p_depth   current depth in tree
    /// \param  p_value   values at each node of the tree
    inline  double calculateExpectation(const int &p_depth, const Eigen::ArrayXd   &p_value) const
    {
        return calculateCondExpectation(0, p_depth,  p_value)(0);
    }


};

#endif
