#include <iostream>
#include "reflow/core/utils/constant.h"
#include "test/c++/tools/semilagrangien/OptimizeSLCase2.h"

using namespace reflow;
using namespace Eigen ;
using namespace std ;


vector< array< double, 2> >  OptimizerSLCase2::getCone(const  vector<  array< double, 2>  > &p_xInit) const
{
    // max of the vol
    vector< array< double, 2> > xReached(p_xInit.size());
    // all processor have the whole domain
    xReached[0][0] = -reflow::infty;
    xReached[0][1] = reflow::infty;
    xReached[1][0] = -reflow::infty;
    xReached[1][1] = reflow::infty;
    return xReached;
}

pair< ArrayXd, ArrayXd>  OptimizerSLCase2::stepOptimize(const ArrayXd   &p_point,
        const vector< shared_ptr<SemiLagrangEspCond> > &p_semiLag, const double &p_time, const Eigen::ArrayXd &) const
{
    pair< ArrayXd, ArrayXd> solutionAndControl;
    solutionAndControl.first.resize(1);
    solutionAndControl.second.resize(1);
    ArrayXd b = ArrayXd::Zero(2);
    ArrayXXd sig = ArrayXXd::Zero(2, 3);
    sig(0, 0) = SinSomme()(p_point);
    sig(1, 0) = CosSomme()(p_point);
    sig(0, 1) = m_beta * sqrt(2.);
    sig(1, 2) = m_beta * sqrt(2.);
    pair<double, bool> lagrang = p_semiLag[0]->oneStep(p_point, b, sig, m_dt);
    if (!lagrang.second)
    {
        cout << "Step too large" << endl ;
        abort();
    }
    solutionAndControl.first(0)  = SourceTerm(m_beta)(p_time, p_point) * m_dt + lagrang.first;
    return solutionAndControl;
}
