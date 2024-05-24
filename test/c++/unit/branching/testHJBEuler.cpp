
#include <functional>
#include <boost/mpi.hpp>
#include <boost/timer/timer.hpp>
#include <boost/chrono.hpp>
#include <Eigen/Dense>
#include <trng/yarn5.hpp>
#include "reflow/branching/ExpDist.h"
#include "reflow/branching/GammaDist.h"
#include "reflow/branching/solvePDEDYMC.h"
#include "estimateHJB.h"



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
    auto muFunc([](const double &, const Eigen::Matrix<double, 6, 1> &)
    {
        return Eigen::Matrix<double, 6, 1>::Constant(0);
    });
    std::function<  Eigen::Matrix<double, 6, 1> (const double &, const Eigen::Matrix<double, 6, 1> &) >  mu(std::cref(muFunc));
    // volat
    auto volFunc([sig](const double &, const Eigen::Matrix<double, 6, 1> &)
    {
        return Eigen::Matrix<double, 6, 6>::Identity() * sig;
    });
    std::function<  Eigen::Matrix<double, 6, 6>(const double &, const Eigen::Matrix<double, 6, 1> &) >  vol(std::cref(volFunc));

    //  source fonction
    double mat = 1.; // maturity

    auto  fSFunc([ ](const double &,  const Eigen::Matrix<double, 6, 1> &, const double &, const Eigen::Matrix<double, 6, 1> &p_z)
    {
        return  - std::min(p_z.squaredNorm(), 1.);
        ;
    });
    function< double (const double &, const Matrix<double, 6, 1>&, const double &,  const Eigen::Matrix<double, 6, 1> &) > fNonLin(std::cref(fSFunc));


    int nbSimAnal = 10000000;
    pair<double, double > anal = reflow::estimateHJB(nbSimAnal, 1., point, mat, g);
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

        reflow::ExpDist law(lambda);
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

            // no Euler
            double stepEuler = 10;

            trng::yarn5 gen;

            boost::timer::cpu_timer tt;
            std::tuple< double, Eigen::Array<double, 6, 1>, double,  Eigen::Array<double, 6, 1> >    val = reflow::solvePDEDYMCEuler<6, reflow::ExpDist, trng::yarn5>(mu, vol, fNonLin, point, 0., mat, law,  g, nbSim, stepEuler, gen);

            if (world.rank() == 0)
            {
                boost::chrono::duration<double> seconds = boost::chrono::nanoseconds(tt.elapsed().user);
                cout << " Value  PDE " << std::get<0>(val) <<   "  std " << std::get<1>(val) <<   " TIME " << seconds.count() <<   endl ;
            }
        }
    }
}
