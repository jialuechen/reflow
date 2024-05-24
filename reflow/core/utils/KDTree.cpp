#include <iterator>
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "KDTree.h"

using namespace std;
using namespace Eigen;


namespace reflow
{

/// compare in a given dimension 2 points
class comparePt
{
public:
    size_t m_idx;
    explicit comparePt(size_t p_idx): m_idx(p_idx) {}
    // inline bool compareInDim(
    bool operator()(
        const pair< ArrayXd, size_t > &a,
        const pair< ArrayXd, size_t > &b
    )
    {
        return (a.first(m_idx) < b.first(m_idx));
    }
};


inline double dist2(const ArrayXd &a, const ArrayXd &b)
{
    double distc = 0;
    for (int i = 0; i < a.size(); i++)
    {
        double di = a(i) - b(i);
        distc += di * di;
    }
    return distc;
}

inline double dist2(const shared_ptr< KDNode >  &a, const shared_ptr< KDNode > &b)
{
    return dist2(a->getPoint(), b->getPoint());
}


KDTree::KDTree(const ArrayXXd &points)
{
    m_leaf = make_shared<KDNode>();

    vector< pair< ArrayXd, size_t> > vecPoints(points.cols());
    for (int i = 0; i < points.cols(); ++i)
        vecPoints[i] =  make_pair(points.col(i), i);

    auto beg = vecPoints.begin();
    auto end = vecPoints.end();
    int level = 0;
    m_root = createTree(beg, end, vecPoints.size(), level);


}


shared_ptr<KDNode> KDTree::createTree(const vector<pair< ArrayXd, size_t>>::iterator &p_beg,
                                      const vector<pair< ArrayXd, size_t>>::iterator   &p_end,
                                      const size_t &p_nbPoints,
                                      const size_t &p_level)
{
    if (p_beg == p_end)
    {
        return shared_ptr< KDNode >();  // empty tree
    }
    size_t dim = p_beg->first.size();

    if (p_nbPoints > 1)
    {
        sort(p_beg, p_end, comparePt(p_level));
    }

    auto middle = p_beg + (p_nbPoints / 2);

    auto lbeg = p_beg;
    auto lend = middle;
    auto rbeg = middle + 1;
    auto rend = p_end;

    size_t llen = p_nbPoints / 2;
    size_t rlen = p_nbPoints - llen - 1;

    shared_ptr< KDNode >  left;
    if (llen > 0 && dim > 0)
    {
        left = createTree(lbeg, lend, llen, (p_level + 1) % dim);
    }
    else
    {
        left = m_leaf;
    }
    shared_ptr< KDNode > right;
    if (rlen > 0 && dim > 0)
    {
        right = createTree(rbeg, rend, rlen, (p_level + 1) % dim);
    }
    else
    {
        right = m_leaf;
    }

    // KDNode result = KDNode();
    return make_shared< KDNode >(*middle, left, right);
}

shared_ptr< KDNode > KDTree::nearest(
    const shared_ptr< KDNode > &p_branch,
    const ArrayXd &p_pt,
    const size_t &p_level,
    const shared_ptr< KDNode > &p_best,
    const double &p_bestDist) const
{
    double d, dx, dx2;

    if (p_branch->isEmpty())
    {
        return make_shared<KDNode>();  // basically, null
    }

    ArrayXd  branchPt = p_branch->getPoint();
    size_t dim = branchPt.size();

    d = dist2(branchPt, p_pt);
    dx = branchPt(p_level) - p_pt(p_level);
    dx2 = dx * dx;

    shared_ptr< KDNode > bestLoc = p_best;
    double bestDistLoc = p_bestDist;

    if (d < p_bestDist)
    {
        bestDistLoc = d;
        bestLoc = p_branch;
    }

    size_t nextLevel = (p_level + 1) % dim;
    shared_ptr< KDNode > section;
    shared_ptr< KDNode > other;

    // select which p_branch  to check
    if (dx > 0)
    {
        section = p_branch->getLeft();
        other = p_branch->getRight();
    }
    else
    {
        section = p_branch->getRight();
        other = p_branch->getLeft();
    }

    // keep nearest neighbor from further down the tree
    shared_ptr< KDNode > further = nearest(section, p_pt, nextLevel, bestLoc, bestDistLoc);
    if (!further->isEmpty())
    {
        double dl = dist2(further->getPoint(), p_pt);
        if (dl < bestDistLoc)
        {
            bestDistLoc = dl;
            bestLoc = further;
        }
        // only check the other p_branch if it makes sense to do so
        if (dx2 < bestDistLoc)
        {
            further = nearest(other, p_pt, nextLevel, bestLoc, bestDistLoc);
            if (!further->isEmpty())
            {
                dl = dist2(further->getPoint(), p_pt);
                if (dl < bestDistLoc)
                {
                    bestDistLoc = dl;
                    bestLoc = further;
                }
            }
        }
    }

    return bestLoc;
}


shared_ptr< KDNode > KDTree::nearestNode(const ArrayXd   &p_pt) const
{
    size_t level = 0;
    double branchDist = dist2(m_root->getPoint(), p_pt);
    return nearest(m_root,          // beginning of tree
                   p_pt,        // point we are querying
                   level,         // start from level 0
                   m_root,          // best is the root
                   branchDist);  // best_dist = branch_dist
}

}
