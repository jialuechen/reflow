
#include <functional>
#include <boost/mpi.hpp>
#include <boost/timer/timer.hpp>
#include <boost/chrono.hpp>
#include <Eigen/Dense>
#include <trng/mrg5.hpp>
#include "reflow/branching/ExpDist.h"
#include "reflow/branching/GammaDist.h"
#include "reflow/branching/solvePDEDY2MC.h"



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
    double vol = 0.2 / sqrt(2);
    // mu
    double tmu = 0.1 / 2;
    // alpha
    double alpha = 0.1 ;
    // g function
    auto  fGFunc([](const Matrix<double, 2, 1> &x)
    {
        return cos(x.sum())  ;
    });
    function< double (const Matrix<double, 2, 1>&)>  g(std::cref(fGFunc));
    // initial position
    Eigen::Matrix<double, 2, 1> initPosition = Eigen::Matrix<double, 2, 1>::Constant(0.5);
    // trend
    Eigen::Matrix<double, 2, 1>  mu = Eigen::Matrix<double, 2, 1>::Constant(tmu);

    // volat
    Eigen::Matrix<double, 2, 2>  sig  = Eigen::Matrix<double, 2, 2>::Identity() * vol ;
    // inverse
    Eigen::Matrix<double, 2, 2> sigInv = sig.inverse();
    // inverse and tranpose
    Eigen::Matrix<double, 2, 2> sigInvTrans = sigInv.transpose();

    double mat = 1; // maturity

    // source function
    double rescale = 0.1 / sqrt(2) ;

    auto  fSFunc([alpha, vol,  rescale, mat, tmu](const double & t,
                 const Eigen::Matrix<double, 2, 1> &x,
                 const double &  p_y, const Eigen::Matrix<double, 2, 1> &,
                 const Eigen::Matrix<double, 3, 1> &p_gam)
    {
        double xsum = x.sum();
        double expMat = exp(alpha * mat);
        double cosum =  cos(xsum);
        double sinSum = sin(xsum);
        double expMatT = exp(alpha * (mat - t));
        double gamDiagSum =  p_gam(0) + p_gam(2);
        return cosum * (alpha + 0.5 * vol * vol * 2) * expMatT  +  sinSum * tmu * 2 * expMatT  + rescale * cosum * cosum * expMatT * expMatT * 2
               + rescale * std::max(-expMat * expMat * 2, std::min(p_y * gamDiagSum, expMat * expMat * 2))
               ;
    });

    function< double (const double &, const Matrix<double, 2, 1>&, const double &,  const Eigen::Matrix<double, 2, 1> &, const Eigen::Matrix<double, 3, 1> &) > fNonLin(std::cref(fSFunc));


    // nest on numb switches
    ArrayXd  nbSwitch(3);
    nbSwitch << 1, 2, 3 ;

    double   lambda = 0.075;

    if (world.rank() == 0)
    {
        cout << " PDE2Y  with GHOST  ND"  << 2 <<  endl ;
        cout << "********************************" << endl ;
    }

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
        for (int ipart = 0; ipart < 6; ++ipart)
        {
            std::vector<int > yVal(nbSwitch(isw));
            yVal[0] = 1000 * pow(2, ipart);
            yVal[1] = 40 * pow(2, ipart);
            if (nbSwitch(isw) > 2)
                yVal[2] = 40 * pow(2, ipart);
            if (nbSwitch(isw) > 3)
                yVal[3] = 20 * pow(2, ipart);
            if (nbSwitch(isw) > 4)
                yVal[4] = 10 * pow(2, ipart);

            if (world.rank() == 0)
            {
                cout << " NBPART " << yVal[0] <<  "  "  ;
                if (nbSwitch(isw) > 1)
                    cout <<  yVal[1] << " " ;
                if (nbSwitch(isw) > 2)
                    cout <<  yVal[2] << "  " ;
                if (nbSwitch(isw) > 3)
                    cout <<  yVal[3] << " " ;
                if (nbSwitch(isw) > 4)
                    cout <<  yVal[4] << " " ;
                cout << endl ;
            }

            trng::mrg5 gen;

            boost::timer::cpu_timer tt;

            std::tuple< double, Eigen::Array<double, 2, 1>, Eigen::Array<double, 3, 1>, double,  Eigen::Array<double, 2, 1>, Eigen::Array<double, 3, 1>   > val =
                reflow::solvePDEDY2MCConst<2, reflow::ExpDist, trng::mrg5>(mu, sig, sigInv, sigInvTrans, fNonLin, initPosition, 0., mat, law,  g, yVal, gen);

            if (world.rank() == 0)
            {
                boost::chrono::duration<double> seconds = boost::chrono::nanoseconds(tt.elapsed().user);
                cout << " Value  PDE " << std::get<0>(val)  ;
                cout << " REAL " << cos(initPosition.sum())*exp(alpha * mat)  ;
                cout <<  " TIME " << seconds.count() << endl ;
            }

        }
    }
    return 0;
}
