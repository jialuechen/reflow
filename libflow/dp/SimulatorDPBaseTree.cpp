
#include <memory>
#include <vector>
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "geners/Reference.hh"
#include "geners/vectorIO.hh"
#include "geners/arrayIO.hh"
#include "libflow/core/utils/eigenGeners.h"
#include "libflow/core/utils/constant.h"
#include "libflow/dp/SimulatorDPBaseTree.h"


using namespace Eigen ;
using namespace std ;

namespace libflow
{

SimulatorDPBaseTree::SimulatorDPBaseTree(const shared_ptr<gs::BinaryFileArchive> &p_binForTree): m_binForTree(p_binForTree)
{
    gs::Reference< ArrayXd >(*p_binForTree, "dates", "").restore(0, &m_dates);
}

void SimulatorDPBaseTree::load(const int &p_idateCur)
{
    m_idateCur = p_idateCur;
    //load nodes
    gs::Reference< ArrayXXd>(*m_binForTree, "points", "").restore(m_idateCur, &m_nodesCurr);
    //load nodes
    if (m_idateCur < m_dates.size() - 1)
    {
        gs::Reference< ArrayXXd>(*m_binForTree, "points", "").restore(m_idateCur + 1, &m_nodesNext);
        //load probability transition  and helper
        gs::Reference< vector< double > >(*m_binForTree, "proba", "").restore(m_idateCur, &m_proba);
        // connection matrix
        gs::Reference< vector<vector< array<int, 2 > > > >(*m_binForTree, "connection", "").restore(m_idateCur, &m_connected);
    }
}



int SimulatorDPBaseTree::getNodeReachedInForward(const int &p_nodeStart, const double &p_randUni) const
{
    double probSum = 0.;
    int iret = 0;
    for (size_t i = 0; i < m_connected[p_nodeStart].size(); ++i)
    {
        probSum += m_proba[m_connected[p_nodeStart][i][1]];
        if (p_randUni < probSum)
        {
            iret = m_connected[p_nodeStart][i][0];
            break;
        }
    }
    return iret;
}

}

