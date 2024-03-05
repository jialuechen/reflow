#include <iostream>
#include "test/c++/tools/semilagrangien/OptimizeSLCase1.h"

using namespace libflow;
using namespace Eigen ;
using namespace std ;


vector< array< double, 2> >  OptimizerSLCase1::getCone(const  vector<  array< double, 2>  > &p_xInit) const
{
    // max of the vol
    double volMax =  sqrt(2.*m_dt);
    vector< array< double, 2> > xReached(2);
    xReached[0][0] = p_xInit[0][0] - volMax;
    xReached[0][1] = p_xInit[0][1] + volMax;
    xReached[1][0] = p_xInit[1][0] - volMax;
    xReached[1][1] = p_xInit[1][1] + volMax;
    return xReached;
}

pair< ArrayXd, ArrayXd>  OptimizerSLCase1::stepOptimize(const ArrayXd   &p_point,
        const vector< shared_ptr<SemiLagrangEspCond> > &p_semiLag, const double &p_time, const Eigen::ArrayXd &) const
{
    // here no control
    pair< ArrayXd, ArrayXd> solutionAndControl;
    solutionAndControl.first.resize(1);
    solutionAndControl.second.resize(1);
    ArrayXd sol(1);
    ArrayXd b = ArrayXd::Zero(2);
    ArrayXXd sig = ArrayXXd::Zero(2, 2);
    sig(0, 0) = SinSomme1()(p_point);
    sig(1, 0) = SinSomme2()(p_point);
    pair<double, bool> lagrang = p_semiLag[0]->oneStep(p_point, b, sig, m_dt);
    if (!lagrang.second)
    {
        cout << "Step too large" << endl ;
        abort();
    }
    solutionAndControl.first(0) = SourceTerm()(p_time, p_point) * m_dt + lagrang.first;
    return solutionAndControl;
}
