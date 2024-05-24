
#include <iostream>
#include <memory>
#include "reflow/core/grids/FullGrid.h"
#include "reflow/core/utils/comparisonUtils.h"
#include "reflow/semilagrangien/SemiLagrangEspCond.h"

using namespace Eigen ;
using namespace reflow;
using namespace std;


SemiLagrangEspCond::SemiLagrangEspCond(const shared_ptr<InterpolatorSpectral> &p_interpolator, const vector <array< double, 2>  > &p_extremalValues, const bool &p_bModifVol):
    m_interpolator(p_interpolator), m_extremalValues(p_extremalValues), m_bModifVol(p_bModifVol) {}


pair<double, bool>  SemiLagrangEspCond::oneStep(const Eigen::ArrayXd   &p_x, const Eigen::ArrayXd &p_b, const ArrayXXd &p_sig, const double &p_dt) const
{
    // first store the points
    ArrayXXd pointReachedMax(p_x.size(), p_sig.cols());
    ArrayXXd pointReachedMin(p_x.size(), p_sig.cols());
    double sqrtdtRec = sqrt(p_dt * p_sig.cols());
    for (int i = 0; i < p_sig.cols(); ++i)
    {
        pointReachedMax.col(i)  =  p_x + p_b * p_dt + p_sig.col(i) * sqrtdtRec;
        pointReachedMin.col(i)  =  p_x + p_b * p_dt - p_sig.col(i) * sqrtdtRec;
    }
    double valRet = 0;
    //  test if the points are inside the domain
    for (int j = 0; j < p_sig.cols(); ++j)
    {
        double leftWeight  = 0.5;
        double rightWeight = 0.5 ;
        int idToMod = -1 ; // to modify
        int iLeftOrRight = 0 ;
        int iSignSigMod = 1 ;
        for (int i = 0; i < p_x.size(); ++i)
        {
            double pointLeft  ;
            double pointRight ;
            if (p_sig(i, j) > 0)
            {
                pointLeft = pointReachedMin(i, j);
                pointRight = pointReachedMax(i, j);
                iSignSigMod = 1; // keep in mind sign of the current vol
            }
            else
            {
                pointLeft = pointReachedMax(i, j);
                pointRight = pointReachedMin(i, j);
                iSignSigMod = -1; ; // keep in mind sign of the current vol
            }
            if (pointLeft < m_extremalValues[i][0])
            {
                idToMod = i ;
                iLeftOrRight = -1;
                break; // we work on this dimension (first with violation)
            }
            if (pointRight > m_extremalValues[i][1])
            {
                idToMod = i;
                iLeftOrRight = 1;
                break ; // we work on this dimension (first with violation)
            }
        }
        // first case : points inside the domain
        if (iLeftOrRight == 0)
        {
            valRet += leftWeight * m_interpolator->apply(pointReachedMin.col(j)) + rightWeight * m_interpolator->apply(pointReachedMax.col(j));
        }
        else
        {

            bool bRet = false;
            if (m_bModifVol)
            {
                double p_sigaSquare = pow(p_sig(idToMod, j), 2.) * p_dt * p_sig.cols();
                // define new weights
                if (iLeftOrRight == -1)
                {
                    double UMinus = p_x(idToMod) -  m_extremalValues[idToMod][0] + p_b(idToMod) * p_dt;
                    if (isLesserOrEqual(UMinus, 0.))
                        break;
                    double UPlusSquare = pow(p_sigaSquare / UMinus, 2.);
                    leftWeight = UPlusSquare / (UPlusSquare + p_sigaSquare);
                    rightWeight = p_sigaSquare / (UPlusSquare + p_sigaSquare);
                    bRet = true;
                }
                else
                {
                    double UPlus = m_extremalValues[idToMod][1] - p_x(idToMod) - p_b(idToMod) * p_dt;
                    if (isLesserOrEqual(UPlus, 0.))
                        break;
                    double UMinusSquare = pow(p_sigaSquare / UPlus, 2.);
                    leftWeight = p_sigaSquare / (UMinusSquare + p_sigaSquare);
                    rightWeight = UMinusSquare / (UMinusSquare + p_sigaSquare);
                    bRet = true;
                }
                if (bRet)
                {
                    // now recalculate arrival points
                    for (int  i = 0; i < p_x.size(); ++i)
                    {
                        double UPlus =  iSignSigMod * p_sig(i, j) * sqrt(p_dt * p_sig.cols() * leftWeight / (pow(rightWeight, 2.) + rightWeight * leftWeight));
                        double UMinus = iSignSigMod * p_sig(i, j) * sqrt(p_dt *  p_sig.cols() * rightWeight / (pow(leftWeight, 2.) + rightWeight * leftWeight));
                        pointReachedMin(i, j) =  p_x(i)  -  UMinus + p_b(i) * p_dt;
                        pointReachedMax(i, j) =  p_x(i) +  UPlus  +  p_b(i) * p_dt;
                        if (isStrictlyLesser(pointReachedMin(i, j), m_extremalValues[i][0]))
                        {
                            pointReachedMin(i, j) = m_extremalValues[i][0];
                        }
                        else if (isStrictlyMore(pointReachedMin(i, j), m_extremalValues[i][1]))
                        {
                            pointReachedMin(i, j) = m_extremalValues[i][1];
                        }
                        if (isStrictlyLesser(pointReachedMax(i, j), m_extremalValues[i][0]))
                        {
                            pointReachedMax(i, j)  = m_extremalValues[i][0];
                        }
                        else if (isStrictlyMore(pointReachedMax(i, j), m_extremalValues[i][1]))
                        {
                            pointReachedMax(i, j)  = m_extremalValues[i][1];
                        }
                    }
                    valRet += leftWeight * m_interpolator->apply(pointReachedMin.col(j)) + rightWeight * m_interpolator->apply(pointReachedMax.col(j));
                }
                // otherwise truncate:
                if (!bRet)
                {
                    ArrayXd ppointReachedMin = pointReachedMin.col(j);
                    ArrayXd ppointReachedMax = pointReachedMax.col(j);
                    const FullGrid *mgrid = static_cast<const FullGrid *>(m_interpolator->getGrid());
                    mgrid->truncatePoint(ppointReachedMin);
                    mgrid->truncatePoint(ppointReachedMax);
                    double leftWeight  = 0.5;
                    double rightWeight = 0.5 ;
                    valRet += leftWeight * m_interpolator->apply(ppointReachedMin) + rightWeight * m_interpolator->apply(ppointReachedMax);
                }
            }
            else
            {
                return  make_pair(0, false);
            }
        }
    }
    return make_pair(valRet /  p_sig.cols(), true);
}
