
#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include "geners/vectorIO.hh"
#include "geners/arrayIO.hh"
#include "libflow/core/utils/constant.h"
#include "libflow/core/utils/eigenGeners.h"
#include "libflow/core/utils/constant.h"
#include "test/c++/tools/simulators/TrinomialTreeOUSimulator.h"


using namespace std;
using namespace Eigen;

TrinomialTreeOUSimulator::TrinomialTreeOUSimulator(const double &p_mr, const double &p_sig, const ArrayXd &p_dates): m_mr(p_mr), m_sig(p_sig), m_dates(p_dates), m_dx(m_dates.size()), m_nbNodeBelow0(p_dates.size())
{
    assert(std::fabs(p_dates(0)) < libflow::tiny);
    // construct the tree
    m_facing.reserve(p_dates.size() - 1);
    m_proba.reserve(p_dates.size() - 1);

    // size of the mesh
    m_dx(0) = m_sig * sqrt((3. / (2.*m_mr)) * (1 - exp(-2 * m_mr * (p_dates(0)))));
    for (int it = 1;  it  < p_dates.size() ; ++it)
        m_dx(it) = m_sig * sqrt((3. / (2.*m_mr)) * (1 - exp(-2 * m_mr * (p_dates(it) - p_dates(it - 1)))));

    // to store the number of mesh below 0
    m_nbNodeBelow0(0) = 0;
    m_nbNodeBelow0(1) = 1;
    for (int it = 1; it < p_dates.size() - 1; ++it)
    {
        m_nbNodeBelow0(it + 1) = static_cast<int>(-sqrt(2. / 3.) +
                                 m_nbNodeBelow0(it) * (m_dx(it) / m_dx(it + 1) * exp(- m_mr * (p_dates(it + 1) - p_dates(it))) - 1))
                                 + m_nbNodeBelow0(it) + 1;
    }
    // probability and connections
    for (int id = 0;  id  < p_dates.size() - 1 ; ++id)
    {
        double expMean =  exp(- m_mr * (p_dates(id + 1) - p_dates(id))) * m_dx(id) / m_dx(id + 1) ;
        // number of points at given date is 2*m_nbNodeBelow0(id)+1
        ArrayXi facing(2 * m_nbNodeBelow0(id) + 1);
        facing(m_nbNodeBelow0(id)) = m_nbNodeBelow0(id + 1);
        // probability connection  (doxn, midle, up)
        ArrayXXd proba(3, 2 * m_nbNodeBelow0(id) + 1);
        proba(0, m_nbNodeBelow0(id)) = 1. / 6.   ; //down
        proba(1, m_nbNodeBelow0(id)) = 2. / 3.   ; // middle
        proba(2, m_nbNodeBelow0(id)) = 1. / 6.   ; // up
        // nest on nodes number belwo (or above 0)
        for (int it = 0; it < m_nbNodeBelow0(id); ++it)
        {
            double xut = -sqrt(2. / 3.) + (it + 1) * (expMean - 1) ;
            int idev = static_cast<int>(xut);
            facing(m_nbNodeBelow0(id) + it + 1) = m_nbNodeBelow0(id + 1) + idev + it + 1;
            facing(m_nbNodeBelow0(id) - it - 1) = m_nbNodeBelow0(id + 1) - idev - it - 1;
            // epsd
            double epsd = idev + it + 1 - (it + 1) * expMean ;
            double epsd2 = pow(epsd, 2.) ;
            // down
            proba(0, m_nbNodeBelow0(id) + it + 1) =  0.5 * (1. / 3. + epsd2 + epsd) ;
            proba(2, m_nbNodeBelow0(id) - it - 1) = proba(0, m_nbNodeBelow0(id) + it + 1);
            // middle
            proba(1, m_nbNodeBelow0(id) + it + 1) =  2. / 3. - epsd2 ;
            proba(1, m_nbNodeBelow0(id) - it - 1) =   proba(1, m_nbNodeBelow0(id) + it + 1);
            // up
            proba(2, m_nbNodeBelow0(id) + it + 1) =  0.5 * (1. / 3. + epsd2 - epsd) ;
            proba(0, m_nbNodeBelow0(id) - it - 1) = proba(2, m_nbNodeBelow0(id) + it + 1);
        }
        assert(proba.minCoeff() > 0.);
        assert(proba.maxCoeff() < 1);
        m_proba.push_back(proba);
        m_facing.push_back(facing);
    }
}



ArrayXd  TrinomialTreeOUSimulator::calculateCondExpectation(const int &p_depth0, const int &p_depthLast, const Eigen::ArrayXd   &p_value) const
{
    // backward
    ArrayXd valNext = p_value;
    for (int  id = p_depthLast - 1; id >= p_depth0; --id)
    {
        ArrayXd valCur(m_proba[id].cols());
        for (int i = 0; i < m_proba[id].cols(); ++i)
        {
            valCur(i) = m_proba[id](0, i) * valNext(m_facing[id](i) - 1) + m_proba[id](1, i) * valNext(m_facing[id](i)) + m_proba[id](2, i) * valNext(m_facing[id](i) + 1);
        }
        valNext =  valCur;
    }
    return valNext;
}

double  TrinomialTreeOUSimulator::calculateStepCondExpectation(const int &p_depth, const int &p_inode, const Eigen::ArrayXd   &p_value) const
{
    double valRet =  m_proba[p_depth](0, p_inode) * p_value(m_facing[p_depth](p_inode) - 1) +
                     m_proba[p_depth](1, p_inode) * p_value(m_facing[p_depth](p_inode)) + m_proba[p_depth](2, p_inode) * p_value(m_facing[p_depth](p_inode) + 1);
    return valRet;
}

ArrayXXd TrinomialTreeOUSimulator::getPoints(const int &p_idate) const
{
    ArrayXXd ret(1, 2 * m_nbNodeBelow0(p_idate) + 1);
    for (int it = 0; it < ret.size(); ++it)
        ret(0, it) = (-m_nbNodeBelow0(p_idate) + it) * m_dx(p_idate);
    return ret;
}

ArrayXXd TrinomialTreeOUSimulator::getProbability(const int &p_idateBeg, const int &p_idateEnd) const
{
    ArrayXXd proba = ArrayXXd::Zero(2 * m_nbNodeBelow0(p_idateBeg) + 1, 2 * m_nbNodeBelow0(p_idateEnd) + 1);
    int d = p_idateEnd - p_idateBeg;
    ArrayXi kp(p_idateEnd - p_idateBeg);
    for (long i = 0; i < static_cast<int>(pow(3., d)); ++i)
    {
        long ip = i;
        long idiv = static_cast<int>(pow(3., d - 1));
        for (int id = 0; id < p_idateEnd - p_idateBeg; ++id)
        {
            kp(id) = ip / idiv;
            ip = ip % idiv;
            idiv /= 3;
        }
        for (int inode = 0; inode < 2 * m_nbNodeBelow0(p_idateBeg) + 1; ++inode)
        {
            std::pair<double, int> probaAndNode = probCal(inode, p_idateBeg, p_idateEnd, p_idateBeg, kp);
            proba(inode, probaAndNode.second) += probaAndNode.first;
        }

    }
    return proba;
}


std::pair<double, int > TrinomialTreeOUSimulator::probCal(const int &p_ipos, const  int &p_depth, const int &p_depthMax,  const int   &p_depthCur, const Eigen::ArrayXi &p_downMiddleUp) const
{
    if (p_depthCur == p_depthMax)
        return make_pair(1., p_ipos);
    else
    {
        // next point
        int nextPt = m_facing[p_depthCur](p_ipos) + (p_downMiddleUp(p_depthCur - p_depth) - 1);
        std::pair<double, int >  probAndNode = probCal(nextPt,  p_depth, p_depthMax, p_depthCur + 1, p_downMiddleUp);
        return make_pair(m_proba[p_depthCur](p_downMiddleUp(p_depthCur - p_depth), p_ipos) * probAndNode.first, probAndNode.second);
    }
}

pair< vector< vector< array<int, 2> > >, vector< double >  > TrinomialTreeOUSimulator::calConnected(const ArrayXXd &p_proba)
{
    vector< vector< array<int, 2> > > connected;
    vector<double> probaRet ;
    connected.reserve(p_proba.rows());
    probaRet.reserve(p_proba.size());
    int iprob = 0;
    for (int i = 0; i < p_proba.rows(); ++i)
    {
        vector< array<int, 2> >  nodes;
        nodes.reserve(p_proba.cols());
        for (int j = 0; j < p_proba.cols(); ++j)
        {
            if (p_proba(i, j) > libflow::tiny)
            {
                array<int, 2 > two {{j, iprob++} };
                nodes.push_back(two);
                probaRet.push_back(p_proba(i, j));
            }
        }
        connected.push_back(nodes);
    }
    return make_pair(connected, probaRet) ;
}

void  TrinomialTreeOUSimulator::dump(const std::string &p_name, const Eigen::ArrayXi &p_index)
{
    gs::BinaryFileArchive binArxiv(p_name.c_str(), "w");
    ArrayXd ddates(p_index.size());
    for (int i = 0; i < p_index.size(); ++i)
        ddates(i) = m_dates(p_index(i));
    binArxiv << gs::Record(ddates, "dates", "");
    for (int i = 0 ;  i < p_index.size(); ++i)
    {
        ArrayXXd points = getPoints(p_index(i));
        binArxiv << gs::Record(points, "points", "");
    }
    for (int i = 0 ;  i < p_index.size() - 1; ++i)
    {
        ArrayXXd  proba = getProbability(p_index(i), p_index(i + 1));
        pair< vector< vector< array<int, 2> > >, vector< double >  > conAndProb = calConnected(proba);
        binArxiv << gs::Record(conAndProb.second, "proba", "");
        binArxiv << gs::Record(conAndProb.first, "connection", "");
    }
    binArxiv.flush() ;
}
