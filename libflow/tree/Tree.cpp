
#include <iostream>
#include "libflow/tree/Tree.h"

using namespace std;
using namespace Eigen;

namespace libflow
{

Tree::Tree() {}


Tree::Tree(const vector< double > &p_proba, const vector< std::vector< std::array<int, 2> > > &p_connected): m_proba(p_proba), m_connected(p_connected), m_nbNodeNextDate(0)
{

    for (size_t i = 0; i < m_connected.size(); ++i)
    {
        for (size_t j = 0; j < m_connected[i].size(); ++j)
        {
            m_nbNodeNextDate = std::max(m_nbNodeNextDate, static_cast<int>(m_connected[i][j][0]));
        }
    }
    m_nbNodeNextDate += 1;
}

void Tree::update(const vector< double > &p_proba,
                  const vector< std::vector< std::array<int, 2> > >  &p_connected)
{
    m_proba = p_proba;
    m_connected = p_connected;
    m_nbNodeNextDate = 0;
    for (size_t i = 0; i < m_connected.size(); ++i)
    {
        for (size_t j = 0; j < m_connected[i].size(); ++j)
            m_nbNodeNextDate = std::max(m_nbNodeNextDate, static_cast<int>(m_connected[i][j][0]));
    }
    m_nbNodeNextDate += 1;
}


ArrayXd  Tree::expCond(const ArrayXd &p_values) const
{
    ArrayXd ret =  ArrayXd::Zero(m_connected.size());
    for (size_t i = 0 ; i < m_connected.size(); ++i)
    {
        for (size_t j = 0; j < m_connected[i].size(); ++j)
        {
            ret(i) += m_proba[m_connected[i][j][1]] * p_values(m_connected[i][j][0]);
        }
    }
    return ret;
}

ArrayXXd  Tree::expCondMultiple(const ArrayXXd &p_values) const
{
    ArrayXXd ret =  ArrayXXd::Zero(p_values.rows(), m_connected.size());
    for (size_t i = 0 ; i < m_connected.size(); ++i)
    {
        for (size_t j = 0; j < m_connected[i].size(); ++j)
        {
            ret.col(i) += m_proba[m_connected[i][j][1]] * p_values.col(m_connected[i][j][0]);
        }
    }
    return ret;
}
}
