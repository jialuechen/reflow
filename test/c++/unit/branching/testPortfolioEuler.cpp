
#include <functional>
#include <boost/mpi.hpp>
#include <boost/timer/timer.hpp>
#include <boost/chrono.hpp>
#include <Eigen/Dense>
#include <trng/yarn4.hpp>
#include "libflow/branching/ExpDist.h"
#include "libflow/branching/GammaDist.h"
#include "libflow/branching/solvePDEDY2MC.h"
#include "estimatePortfolio.h"


/** \file mainPortfolioEuler.cpp
 * \brief solve the portfolio ptimization problem in dimension 6 using Euler Scheme routine
 *        Taken from 'Monte Carlo for high-dimensional degenerated Semi Linear and Full Non Linear PDEs' by X Warin
 *
 *         Equation
 *         \f[
 *             (-\partial_t u-{\cal L} u)(t,x)  = f(u,Du(t,x),D2u(t,x))
 *         \f]
 *         with
 *         \f[
 *            \mu =  ( 0, k^1 (m^1-y^1), ...,k^d (m^d-y^d) )^{\top},
 *         \f]
 *         \f[
 *             \sigma = \left(  \begin{array}{lllll}
 *                            \bar \sigma & 0 & ... & ...&  0 \	\
 *                            0 & c \sqrt{m^1} & 0 & ... & 0 \		\
 *                            0  & \dotsb &  \ddots & \dotsb  & 0 \	\
 *                            0  & \dotsb &  \dotsb & \ddots  & 0 \	\
 *                            0 & ... & ... & 0 & c \sqrt{m^d}
 *                      \end{array} \right)
 *         \f]
 *         \f[
 *             g(x) = - e^{-\eta x}
 *         \f]
 *         \f[
 *            f(y,z,\theta)=
 -\frac{1}{2} \bar{\sigma}^2 \theta_{11}  +\frac{1}{2} \sum_{i=1}^d (c^i)^2 ((y^i)^2-m^i) \theta_{i+1,i+1} -  \sum_{i=1}^d  \frac{\mu^i z_1 }{2 y^i \theta_{11}}.
 *         \f]
 *          Semi analytical solution given by
 *         \f[
 *            v(t,x,y^1,.., y^d)=-e^{-\eta x} \E[\prod_{i=1}^d \exp\left(-\frac{1}{2}\int_t^T \frac{(\mu^i)^2}{\tilde Y^i_s}ds \right) ]
 *         \f]
 *         Truncaton of $f$ to be Lipschitz
 *         \f[
 *            \begin{array}{ll}
 *             f_{M}(y,z,\theta) = &-\frac{1}{2} \bar \sigma^2 \theta_{11}  +\frac{1}{2} \sum_{i=1}^d (c^i)^2 ((y^i)^2-m^i) \theta_{2,2} + \\
 *                                 &  \sup_{ \begin{array}{c} \eta = (\eta^1,...,\eta^d) \\ 0 \le \eta^i\le M, i=1,d \end{array}}
 *                                      \sum_{i=1}^d \left(\frac{1}{2}(\eta^i)^2 y^i \theta_{11}+(\eta^i) \mu^i z_1\right).
 *            \end{array}
 *         \f]
 * \author Xavier Warin
 */

using namespace std;
using  namespace Eigen;



#if defined   __linux
#include <fenv.h>
#define enable_abort_on_floating_point_exception() feenableexcept(FE_DIVBYZERO | FE_INVALID)
#endif

int main(int argc, char *argv[])
{

#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif

    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;

    // Asset and volatility
    double muSj1 = 0.05 ; // trend asset
    double sigS1 = 1.; // vol asset
    double revertSj1 =  0.1 ; // mean reverting of volatility
    double muVolSj1 = 0.3 ; // assymptotic value of volalility
    double sigVolSj1 = 0.2 ; // volatility of volatility


    // Initial condition
    double x0 = 1;
    double x1 =  muVolSj1 ; // initial vol

    // vol of the portfolio
    double sigma1 = 0.1 ;
    // vol for OU replacing CIT
    double sigma2 = sigVolSj1 * sqrt(muVolSj1);

    double mat = 1.;
    // risk aversion
    double eta = 1 ;
    double trunk = 4 ; //truncaton of wealth portfolio

    // g function
    auto  fGFunc([eta](const Matrix<double, 5, 1> &x)
    {
        return -exp(-eta * x(0)) ;
    });
    function< double (const Matrix<double, 5, 1>&)>   g(std::cref(fGFunc));


    // initial point
    Eigen::Matrix<double, 5, 1> point = Eigen::Matrix<double, 5, 1>::Constant(x1);
    point(0) = x0 ;

    // trend
    auto muFunc([revertSj1, muVolSj1](const double &, const Eigen::Matrix<double, 5, 1> &p_x)
    {
        Eigen::Matrix<double, 5, 1> ret = Eigen::Matrix<double, 5, 1>::Zero();
        for (int id = 0; id < 4; ++id)
        {
            ret(id + 1) = revertSj1 * (muVolSj1 - p_x(id + 1));
        }
        return ret;
    });

    std::function<  Eigen::Matrix<double, 5, 1> (const double &, const Eigen::Matrix<double, 5, 1> &) >  mu(std::cref(muFunc));

    // volat
    auto volFunc([sigma1, sigma2](const double &, const Eigen::Matrix<double, 5, 1> &)
    {
        Eigen::Matrix<double, 5, 5> ret = Eigen::Matrix<double, 5, 5>::Identity();
        ret(0, 0) = sigma1;
        for (int id = 0; id < 4; ++id)
        {
            ret(id + 1, id + 1) = sigma2;
        }
        return ret;
    });
    std::function<  Eigen::Matrix<double, 5, 5>(const double &, const Eigen::Matrix<double, 5, 1> &) >  vol(std::cref(volFunc));



    auto  fSFunc([ sigma1, sigma2, sigVolSj1, muSj1, trunk, sigS1](const double &,
                 const Eigen::Matrix<double, 5, 1> &x,
                 const double &, const Eigen::Matrix<double, 5, 1> &p_z,
                 const Eigen::Matrix<double, 15, 1> &p_gam)
    {
        Eigen::Matrix<double, 5, 1> hessDiag ;
        int idec = 0;
        for (int id = 0; id  < 5; ++id)
        {
            hessDiag(id) = p_gam(idec);
            idec += id + 1;
        }
        double ret =  - 0.5 * sigma1 * sigma1 * hessDiag(0) -
                      0.5 * (sigma2 * sigma2 * hessDiag.tail(4).sum() - sigVolSj1 * sigVolSj1 * x.tail(4).transpose() * hessDiag.tail(4)) ;
        for (int id = 0; id < 4; ++id)
        {
            double control1 = -muSj1 * p_z(0) / (2.*sigS1 * sigS1 * x(id + 1) * hessDiag(0));
            if ((control1 > 0) && (control1 < trunk))
                ret += (control1 * muSj1 * p_z(0) + 0.5 * control1 * control1 * sigS1 * sigS1 * x(id + 1) * hessDiag(0));
            else
                ret += std::max(0., trunk * muSj1 * p_z(0) + 0.5 * trunk * trunk * sigS1 * sigS1 * x(id + 1) * hessDiag(0));
        }
        return ret;
    });

    function< double (const double &, const Matrix<double, 5, 1>&, const double &,  const Eigen::Matrix<double, 5, 1> &, const Eigen::Matrix<double, 15, 1> &) > fNonLin(std::cref(fSFunc));


    double solRef = libflow::estimatePortfolio<4>(muSj1, muVolSj1, revertSj1, sigVolSj1, sigS1, mat, eta, x1, x0, 1000000, 200);

    if (world.rank() == 0)
        cout << " Analytical sol " << solRef << endl ;



    // nest on numb switches
    Array2i  nbSwitch;
    nbSwitch << 1,  2 ;

    // intensity for switches
    double lambda = 0.1;

    // nest for particle number
    int nbpart = 7;

    for (int isw = 0; isw < nbSwitch.size(); ++isw)
    {

        if (world.rank() == 0)
        {
            cout << " SWITCH NB " <<  nbSwitch(isw) << endl ;
            cout << "****************" << endl ;
        }

        libflow::ExpDist law(lambda);
        if (world.rank() == 0)
        {
            std::cout << " Lambd " << lambda << std::endl ;
            cout << " -----------------" << endl ;
        }

        // nest on particles
        for (int ipart = 0; ipart < nbpart; ++ipart)
        {
            std::vector<int > nbSim(nbSwitch(isw));
            nbSim[0] = 1000 * pow(2, ipart);
            nbSim[1] = 100 * pow(2, ipart);
            if (world.rank() == 0)
            {
                cout << " NBPART " << nbSim[0] <<  "  " << nbSim[1] << endl ;
            }

            // no Euler
            double stepEuler = 0.05;

            trng::yarn4 gen;

            boost::timer::cpu_timer tt;
            std::tuple< double, Eigen::Array<double, 5, 1>, Eigen::Array<double, 15, 1>, double,  Eigen::Array<double, 5, 1>, Eigen::Array<double, 15, 1>   >   val = libflow::solvePDEDY2MCEuler<5, libflow::ExpDist, trng::yarn4>(mu, vol, fNonLin, point, 0., mat, law,  g, nbSim, stepEuler, gen);

            if (world.rank() == 0)
            {
                boost::chrono::duration<double> seconds = boost::chrono::nanoseconds(tt.elapsed().user);
                cout << " Value  PDE " << std::get<0>(val) <<   "  std " << std::get<3>(val) <<   " TIME " << seconds.count() <<   endl ;
            }
        }
    }
}
