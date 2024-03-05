
#include <cmath>
#include "Polynomials1D.h"

using namespace std;

namespace libflow
{
double Canonical::F(double p_x, int p_n) const
{
    return pow(p_x, p_n);
}

double Canonical::DF(double p_x, int p_n) const
{
    if (p_n == 0) return 0.;
    return p_n * pow(p_x, p_n - 1);
}

double Canonical::D2F(double p_x, int p_n) const
{
    if (p_n <= 1) return 0.;
    return p_n * (p_n - 1) * pow(p_x, p_n - 2);
}

double Hermite::FRec(double p_x, int p_n, int p_n0, double &p_fn0, double &p_fn0_1) const
{
    if (p_n == p_n0)
    {
        return p_fn0;
    }
    else
    {
        double save = p_fn0;
        p_fn0 = p_x * p_fn0 - p_n0 * p_fn0_1;
        p_fn0_1 = save;
        return FRec(p_x, p_n, p_n0 + 1, p_fn0, p_fn0_1);
    }
}

double Hermite::F(double p_x, int p_n) const
{
    double val = p_x;
    double val2;
    switch (p_n)
    {
    case 0 :
        return 1;
    case 1 :
        return val;
    case 2 :
        return val * val - 1.;
    case 3 :
        return (val * val - 3.) * val;
    case 4 :
        val2 = val * val;
        return (val2 - 6.) * val2 + 3;
    case 5 :
        val2 = val * val;
        return ((val2 - 10) * val2 + 15.) * val;
    case 6 :
        val2 = val * val;
        return ((val2 - 15.) * val2 + 45.) * val2 - 15.;
    case 7:
        val2 = val * val;
        return (((val2 - 21.) * val2 + 105.) * val2 - 105) * val;
    default:
        double Fn0 = F(p_x, 7);
        double Fn0_1 = F(p_x, 6);
        return FRec(p_x, p_n, 7, Fn0, Fn0_1);
    }
}

double Hermite::DF(double p_x, int p_n) const
{
    if (p_n == 0) return 0.;
    else return p_n * F(p_x, p_n - 1);
}

double Hermite::D2F(double p_x, int p_n) const
{
    if (p_n == 0 || p_n == 1) return 0.;
    return p_n * (p_n - 1) * F(p_x, p_n - 2);
}

double Tchebychev::FRec(double p_x, int p_n, int p_n0, double &p_fn0, double &p_fn0_1) const
{
    if (p_n == p_n0)
    {
        return p_fn0;
    }
    else
    {
        double save = p_fn0;
        p_fn0 = 2 * p_x * p_fn0 - p_fn0_1;
        p_fn0_1 = save;
        return FRec(p_x, p_n, p_n0 + 1, p_fn0, p_fn0_1);
    }
}

double Tchebychev::F(double p_x, int p_n) const
{
    double val = p_x;
    double val2, val3, val4;
    switch (p_n)
    {
    case 0 :
        return 1.;
    case 1 :
        return val;
    case 2 :
        return 2. * val * val - 1.;
    case 3 :
        return (4. * val * val - 3.) * val;
    case 4 :
        val2 = val * val;
        return 8. * val2 * val2 - 8. * val2 + 1.;
        break;
    case 5 :
        val2 = val * val;
        val3 = val2 * val;
        return 16. * val3 * val2 - 20. * val3 + 5.* val;
    case 6 :
        val2 = val * val;
        val4 = val2 * val2;
        return 32. * val4 * val2 - 48. * val4 + 18. * val2 - 1;
    case 7 :
        val2 = val * val;
        val3 = val2 * val;
        val4 = val2 * val2;
        return (64. * val4 - 112. * val2 + 56) * val3 - 7. * val;
    default :
        double Fn0 = F(p_x, 7);
        double Fn0_1 = F(p_x, 6);
        return FRec(p_x, p_n, 7, Fn0, Fn0_1);
    }
}

double Tchebychev::DFRec(double p_x, int p_n, int p_n0, double &p_dfn0, double &p_dfn0_1) const
{
    if (p_n == p_n0)
    {
        return p_dfn0;
    }
    else
    {
        double save = p_dfn0;
        p_dfn0 = 2 * p_x * double(p_n0 + 1.0) / double(p_n0) * p_dfn0 -   double(p_n0 + 1.) / double(p_n0 - 1.) * p_dfn0_1;
        p_dfn0_1 = save;
        return DFRec(p_x, p_n, p_n0 + 1, p_dfn0, p_dfn0_1);
    }
}

double Tchebychev::DF(double p_x, int p_n) const
{
    double val = p_x;
    double val2, val4;
    switch (p_n)
    {
    case 0 :
        return 0.;
    case 1 :
        return 1.;
    case 2 :
        return 4. * val;
    case 3 :
        return (12. * val * val - 3.);
    case 4 :
        return (32. * val * val - 16.) * val;
    case 5 :
        val2 = val * val;
        return 80. * val2 * val2 - 60. * val2 + 5.;
    case 6 :
        val2 = val * val;
        val4 = val2 * val2;
        return (192. * val4 - 192. * val2 + 36.) * val;
    case 7 :
        val2 = val * val;
        val4 = val2 * val2;
        return (448. * val4 - 560. * val2 + 168) * val2 - 7.;
    default :
        double Dfn0 = DF(p_x, 7);
        double Dfn0_1 = DF(p_x, 6);
        return DFRec(p_x, p_n, 7, Dfn0, Dfn0_1);
    }
}

double Tchebychev::D2FRec(double p_x, int p_n, int p_n0, double &D2fn0, double &D2fn0_1) const
{
    if (p_n == p_n0)
    {
        return D2fn0;
    }
    else
    {
        double save = D2fn0;
        D2fn0 = 2 * p_x * D2fn0 - D2fn0_1 + 4 * DF(p_x, p_n0);
        D2fn0_1 = save;
        return D2FRec(p_x, p_n, p_n0 + 1, D2fn0, D2fn0_1);
    }
}

double Tchebychev::D2F(double p_x, int p_n) const
{
    double val = p_x;
    double val2, val4;
    switch (p_n)
    {
    case 0 :
        return 0.;
    case 1 :
        return 0.;
    case 2 :
        return 4.;
    case 3 :
        return 24. * val;
    case 4 :
        return (96. * val * val - 16.);
    case 5 :
        val2 = val * val;
        return 320. * val2 * val - 120. * val;
    case 6 :
        val2 = val * val;
        val4 = val2 * val2;
        return (960. * val4 - 576. * val2 + 36.);
    case 7 :
        val2 = val * val;
        val4 = val2 * val2;
        return (2688. * val4 - 2240. * val2 + 336) * val;
    default :
        double D2fn0_1 = D2F(p_x, 6);
        double D2fn0 = D2F(p_x, 7);
        return D2FRec(p_x, p_n, 7, D2fn0, D2fn0_1);
    }
}
}
