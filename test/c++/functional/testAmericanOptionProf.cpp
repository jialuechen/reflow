

#include <boost/timer/timer.hpp>
#include <Eigen/Dense>
#include "test/c++/tools/simulators/BlackScholesSimulator.h"
#include "test/c++/tools/BasketOptions.h"
#include "libflow/core/utils/types.h"
#include "libflow/regression/LocalLinearRegression.h"

using namespace std;
using namespace Eigen ;
using namespace libflow;


// american option by LongStaff Schwarz
//  p_sim        Monte Carlo simulator
//  p_payOff     Option pay off
//  p_regressor  regressor object
template < class Simulator, class PayOff, class Regressor   >
double resolutionAmericanRegression(Simulator &p_sim, const PayOff &p_payOff,  Regressor &p_regressor)
{
    double step = p_sim.getStep();
    // asset simulated under the neutral risk probability : get the trend of first asset to get interest rate
    double expRate = exp(-step * p_sim.getMu()(0));
    // Terminal
    VectorXd Cash(p_payOff(p_sim.getParticles()));
    for (int iStep = 0; iStep < p_sim.getNbStep(); ++iStep)
    {
        shared_ptr<ArrayXXd> asset(new ArrayXXd(p_sim.stepBackwardAndGetParticles()));
        VectorXd payOffLoc = p_payOff(*asset);
        // conditional espectation
        p_regressor.updateSimulations(((iStep == (p_sim.getNbStep() - 1)) ? true : false), asset);
        // condition espection
        VectorXd condEspec = p_regressor.getAllSimulations(Cash) * expRate;
        // arbitrage
        Cash = (condEspec.array() < payOffLoc.array()).select(payOffLoc, Cash * expRate);
    }
    return Cash.mean();
}


void testAmerican(const int &p_nDim, const int &p_nbSimul, const int &p_nMesh)
{
    VectorXd initialValues = ArrayXd::Constant(p_nDim, 1.);
    VectorXd sigma  = ArrayXd::Constant(p_nDim, 0.2);
    VectorXd mu  = ArrayXd::Constant(p_nDim, 0.05);
    MatrixXd corr = MatrixXd::Zero(p_nDim, p_nDim);
    double T = 1. ;
    int nDate = 10 ;
    corr.diagonal().setConstant(1.);
    double strike = 1.;
    // simulator
    BlackScholesSimulator simulator(initialValues, sigma, mu, corr, T, nDate, p_nbSimul, false);
    // payoff
    BasketPut payoff(strike);
    // mesh
    ArrayXi nbMesh = ArrayXi::Constant(p_nDim, p_nMesh);
    // regressor
    LocalLinearRegression regressor(nbMesh);
    // bermudean value
    double value = resolutionAmericanRegression(simulator, payoff, regressor);
    std::cout << " Value " << value <<  std::endl ;
}

int main()
{
    {
        // REFERENCE 0.06031 0.03882 0.02947 0.02404 0.02046 0.01831
        int nDim = 5 ;
        int nbSimul = 40000000;
        int nbMesh = 8;
        testAmerican(nDim, nbSimul, nbMesh);
    }
    {
        // REFERENCE 0.06031 0.03882 0.02947 0.02404 0.02046 0.01831
        int nDim = 6 ;
        int nbSimul = 40000000 * 8;
        int nbMesh = 8;
        testAmerican(nDim, nbSimul, nbMesh);
    }
    // {
    // 	// REFERENCE 0.06031 0.03882 0.02947 0.02404 0.02046 0.01831
    // 	int nDim =7 ;
    // 	int nbSimul = 40000000*64;
    // 	int nbMesh = 8;
    //     testAmerican(nDim, nbSimul, nbMesh);
    // }


    return 0;
}
