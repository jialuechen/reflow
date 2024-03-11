
#ifndef ANALYTICLEGENDRE_H
#define ANALYTICLEGENDRE_H

namespace libflow
{
/**
* \defgroup Legendre Defines Legendre polynomial till degree 8
*@{
*/

/// \class Legendre0 AnalyticLegendre.h
/// Polynomial degree 0
class Legendre0
{
public :
    Legendre0() {}
    /// \brief functor () operator
    double  operator()(const double &)
    {
        return 1;
    }
};

/// \class Legendre1 AnalyticLegendre.h
/// Polynomial degree 1
class Legendre1
{
public :
    Legendre1() {}
    /// \brief functor () operator
    /// \param x          value where to calculate the polynomial value
    double  operator()(const double &x)
    {
        return x;
    }
};

/// \class Legendre2 AnalyticLegendre.h
/// Polynomial degree 2
class Legendre2
{
public :
    Legendre2() {}
    /// \brief functor () operator
    /// \param x          value where to calculate the polynomial value
    double  operator()(const double &x)
    {
        return (1.5 * x * x - 0.5);
    }
};


/// \class Legendre3 AnalyticLegendre.h
/// Polynomial degree 3
class Legendre3
{
public :
    Legendre3() {}
    /// \brief functor () operator
    /// \param x          value where to calculate the polynomial value
    double  operator()(const double &x)
    {
        return 0.5 * x * (5 * x * x - 3.);
    }
};

/// \class Legendre4 AnalyticLegendre.h
/// Polynomial degree 4
class Legendre4
{
public :
    Legendre4() {}
    /// \brief functor () operator
    /// \param x          value where to calculate the polynomial value
    double  operator()(const double &x)
    {
        return (35.*pow(x, 4.) - 30 * x * x + 3) / 8.;
    }
};

/// \class Legendre5 AnalyticLegendre.h
/// Polynomial degree 5
class Legendre5
{
public :
    Legendre5() {}
    /// \brief functor () operator
    /// \param x          value where to calculate the polynomial value
    double  operator()(const double &x)
    {
        return  63. / 8.*pow(x, 5.) - 35. / 4.*pow(x, 3) + 15. / 8.*x ;
    }
};

/// \class Legendre6 AnalyticLegendre.h
/// Polynomial degree 6
class Legendre6
{
public :
    Legendre6() {}
    /// \brief functor () operator
    /// \param x          value where to calculate the polynomial value
    double  operator()(const double &x)
    {
        return  231. / 16.*pow(x, 6.) - 315. / 16.*pow(x, 4.) + 105. / 16.*pow(x, 2.) - 5. / 16.;
    }
};

/// \class Legendre7 AnalyticLegendre.h
/// Polynomial degree 7
class Legendre7
{
public :
    Legendre7() {}
    /// \brief functor () operator
    /// \param x          value where to calculate the polynomial value
    double  operator()(const double &x)
    {
        return  429. / 16.*pow(x, 7.) - 693. / 16.*pow(x, 5.) + 315. / 16.*pow(x, 3.) - 35. / 16.*x;
    }
};

/// \class Legendre8 AnalyticLegendre.h
/// Polynomial degree 8
class Legendre8
{
public :
    Legendre8() {}
    /// \brief functor () operator
    /// \param x          value where to calculate the polynomial value
    double  operator()(const double &x)
    {
        return  6435. / 128.*pow(x, 8.) - 3003. / 32.*pow(x, 6.) + 3465. / 64.*pow(x, 4.) - 315. / 32.*x * x + 35. / 128.;
    }
};

/// \class Legendre9 AnalyticLegendre.h
/// Polynomial degree 9
class Legendre9
{
public :
    Legendre9() {}
    /// \brief functor () operator
    /// \param x          value where to calculate the polynomial value
    double  operator()(const double &x)
    {
        return (1. / 128.) * (315 * x - 4620 * pow(x, 3.)  + 18018 * pow(x, 5.) - 25740 * pow(x, 7.) + 12155 * pow(x, 9.));
    }
};

/// \class Legendre10 AnalyticLegendre.h
/// Polynomial degree 10
class Legendre10
{
public :
    Legendre10() {}
    /// \brief functor () operator
    /// \param x          value where to calculate the polynomial value
    double  operator()(const double &x)
    {
        return (1. / 256.) * (-63 + 3465 * pow(x, 2.) - 30030 * pow(x, 4) + 90090 * pow(x, 6.) - 109395 * pow(x, 8.) + 46189 * pow(x, 10.));
    }
};

/**@}*/
}
#endif /* ANALYITICLEGENDRE_H */
