
#ifndef USE_MPI
#define BOOST_TEST_MODULE testGasStorageTree
#endif
#define BOOST_TEST_DYN_LINK
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <memory>
#include <boost/test/unit_test.hpp>
#include <Eigen/Dense>
#include "reflow/core/grids/OneDimRegularSpaceGrid.h"
#include "reflow/core/grids/OneDimData.h"
#include "reflow/tree/TreeGeners.h"
#include "test/c++/tools/simulators/TrinomialTreeOUSimulator.h"
#include "test/c++/tools/simulators/MeanRevertingSimulatorTree.h"


using namespace std;
using namespace Eigen ;
using namespace reflow;


#if defined   __linux
#include <fenv.h>
#define enable_abort_on_floating_point_exception() feenableexcept(FE_DIVBYZERO | FE_INVALID)
#endif

BOOST_AUTO_TEST_CASE(testAmericanTree)
{
    // define  a time grid
    shared_ptr<OneDimRegularSpaceGrid> timeGrid = make_shared<OneDimRegularSpaceGrid>(0., 2., 1);
    // future values
    shared_ptr<vector< double > > futValues = make_shared<vector<double> >(2);
    (*futValues)[0] = 50.;
    (*futValues)[1] = 50.;
    // define the future curve
    shared_ptr<OneDimData<OneDimRegularSpaceGrid, double> > futureGrid = make_shared<OneDimData< OneDimRegularSpaceGrid, double> >(timeGrid, futValues);

    // degenerated trinomial tree
    double  sigma =  0.2;
    double  mr =  0.000001; // nearly 0

    // interest rate
    double r = 0.05;

    // number of time step for tree
    int nstep = 50 ;

    // maturity
    double T = 1.;


    // dates for tree
    ArrayXd dates =  ArrayXd::LinSpaced(nstep + 1, 0., T);

    TrinomialTreeOUSimulator tree(mr, sigma, dates);

    // number of time steps
    int nbDtStep = 10;
    int nInc = nstep / nbDtStep;

    // sub array for dates
    ArrayXi indexT(nbDtStep + 1);
    for (int i = 0; i < nbDtStep + 1; ++i)
        indexT(i) = i * nInc;

    // create archive
    string nameTree = "TreeAmericanOption";
#ifdef USE_MPI
    boost::mpi::communicator world;
    if (world.rank() == 0)
#endif
        tree.dump(nameTree, indexT);
#ifdef USE_MPI
    world.barrier();
#endif

    // open archive to create a simualtor
    shared_ptr<gs::BinaryFileArchive> binArxiv = make_shared<gs::BinaryFileArchive>(nameTree.c_str(), "r");

    // FIRST VALORIZATION
    /////////////////////

    // a backward simulator
    MeanRevertingSimulatorTree< OneDimData<OneDimRegularSpaceGrid, double> > backSimulator1(binArxiv, futureGrid, sigma, mr);

    // strike of put
    double strike = 50.;

    // actualization
    double actu = exp(r * dates(dates.size() - 1));
    // spot provided by simulator
    ArrayXd spot = backSimulator1.getSpotValues() * actu;
    // actualized value for payoff
    ArrayXd val1 = (strike - spot).cwiseMax(0.) / actu;
    for (int istep = 0; istep < nbDtStep; ++istep)
    {
        // one step backward to update probabilities and connectons between nodes
        backSimulator1.stepBackward();
        // probabilities
        std::vector<double>  proba = backSimulator1.getProba();
        // get connection between nodes
        std::vector< std::vector<std::array<int, 2>  > >  connected = backSimulator1.getConnected();
        // conditional expectation operator
        reflow::Tree tree(proba, connected);
        //interest rates
        actu = exp(r * dates(dates.size() - 1 - (istep + 1) * nInc));
        // spot : add interest rate
        spot = backSimulator1.getSpotValues() * actu;
        // pay off
        ArrayXd payOff = (strike - spot).cwiseMax(0.) / actu;
        //actualize value
        val1 = tree.expCond(val1);
        // arbitrage
        val1 = (val1 > payOff).select(val1, payOff);
    }

    double finalValue = val1(0);

    // sub array for dates
    ArrayXi indexTT(nstep + 1);
    for (int i = 0; i < nstep + 1; ++i)
        indexTT(i) = i ;

    // create archive
#ifdef USE_MPI
    if (world.rank() == 0)
#endif
        tree.dump(nameTree, indexTT);
#ifdef USE_MPI
    world.barrier();
#endif

    // open archive to create a simualtor
    binArxiv = make_shared<gs::BinaryFileArchive>(nameTree.c_str(), "r");


    // a backward simulator
    MeanRevertingSimulatorTree< OneDimData<OneDimRegularSpaceGrid, double> > backSimulator2(binArxiv, futureGrid, sigma, mr);

    // second  VALORIZATION
    /////////////////////
    // actu
    actu = exp(r * dates(dates.size() - 1));
    // spot
    spot = backSimulator2.getSpotValues() * actu;
    // actualized value for payoff
    ArrayXd val2 = (strike - spot).cwiseMax(0.) / actu;
    for (int istep = 0; istep < nstep; ++istep)
    {
        backSimulator2.stepBackward();
        // probabilities
        std::vector<double>  proba = backSimulator2.getProba();
        // get connection between nodes
        std::vector< std::vector<std::array<int, 2>  > >  connected = backSimulator2.getConnected();
        // conditional expectation operator
        reflow::Tree tree(proba, connected);
        //actualize value
        val2 = tree.expCond(val2);
        if ((istep + 1) % nInc == 0)
        {
            //interest rates
            actu = exp(r * dates(dates.size() - 2 - istep));
            // spot : add interest rate
            spot = backSimulator2.getSpotValues() * actu;
            // pay off
            ArrayXd payOff = (strike - spot).cwiseMax(0.) / actu;
            // arbitrage
            val2 = (val2 > payOff).select(val2, payOff);
        }
    }
    BOOST_CHECK_CLOSE(finalValue, val2(0), 0.00001);
}


#ifdef USE_MPI
// (empty) Initialization function. Can't use testing tools here.
bool init_function()
{
    return true;
}

int main(int argc, char *argv[])
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    boost::mpi::environment env(argc, argv);
    return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}

#endif
