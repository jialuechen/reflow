
#include <functional>
#include <boost/mpi.hpp>
#include <boost/timer/timer.hpp>
#include <boost/chrono.hpp>
#include <Eigen/Dense>
#include <trng/mrg4.hpp>
#include "libflow/branching/ExpDist.h"
#include "libflow/branching/GammaDist.h"
#include "libflow/branching/solvePDEDYMC.h"
#include "estimateHJB.h"


/** \file mainBSCVA.cpp
 * \brief solve the Jentzen HJB  problem in dimension 6 suppose a classical SDE with constant coefficients
 *
 *         Equation
 *         \f[
 *             (-\partial_t u-{\cal L} u)(t,x)  = f(Du(t,x))
 *         \f]
 *         \f$ \mu=0 \f$, \f$\sigma = \sqrt{2} I_d \f$
 *         \f$ f(z) = -\theta ||z||^2_2 \f$
 *         \f $g(x)= \log(\frac{1 + ||x||_2^2}{2} ) \f$
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

    // sigma
    double sig = sqrt(2);
    // g function
    auto  fGFunc([](const Matrix<double, 6, 1> &x)
    {
        return log(0.5 * (1 + x.squaredNorm()))  ;
    });
    function< double (const Matrix<double, 6, 1>&)>   g(std::cref(fGFunc));

    // initial position
    Eigen::Matrix<double, 6, 1> point = Eigen::Matrix<double, 6, 1>::Constant(0.);
    // trend
    Eigen::Matrix<double, 6, 1> mu = Eigen::Matrix<double, 6, 1>::Constant(0);
    // volat
    Eigen::Matrix<double, 6, 6> vol =  Eigen::Matrix<double, 6, 6>::Identity() * sig;
    // inverse
    Eigen::Matrix<double, 6, 6> volInv = vol.inverse();

    //  source fonction
    double mat = 1.; // maturity

    auto  fSFunc([ ](const double &,  const Eigen::Matrix<double, 6, 1> &, const double &, const Eigen::Matrix<double, 6, 1> &p_z)
    {
        return  - std::min(p_z.squaredNorm(), 1.);
        ;
    });
    function< double (const double &, const Matrix<double, 6, 1>&, const double &,  const Eigen::Matrix<double, 6, 1> &) > fNonLin(std::cref(fSFunc));


    int nbSimAnal = 10000000;
    pair<double, double > anal = libflow::estimateHJB(nbSimAnal, 1., point, mat, g);
    if (world.rank() == 0)
        cout << "  ANAL " << anal.first << endl ;


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
            nbSim[1] = 10 * pow(2, ipart);
            if (world.rank() == 0)
            {
                cout << " NBPART " << nbSim[0] <<  "  " << nbSim[1] << endl ;
            }


            trng::mrg4 gen;

            boost::timer::cpu_timer tt;
            std::tuple< double, Eigen::Array<double, 6, 1>, double,  Eigen::Array<double, 6, 1> >    val = libflow::solvePDEDYMCConst<6, libflow::ExpDist, trng::mrg4>(mu, vol, volInv, fNonLin, point, 0., mat, law,  g, nbSim, gen);

            if (world.rank() == 0)
            {
                boost::chrono::duration<double> seconds = boost::chrono::nanoseconds(tt.elapsed().user);
                cout << " Value  PDE " << std::get<0>(val) <<   "  std " << std::get<1>(val) <<   " TIME " << seconds.count() <<   endl ;
            }
        }
    }
}
