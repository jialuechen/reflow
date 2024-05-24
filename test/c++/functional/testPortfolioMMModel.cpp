#ifndef USE_MPI
#define BOOST_TEST_MODULE testPortfolioMMM
#endif
#define BOOST_TEST_DYN_LINK
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <Eigen/Dense>
#include <functional>
#include <memory>
#include <boost/test/unit_test.hpp>
#include <boost/timer/timer.hpp>
#include "reflow/core/grids/RegularLegendreGrid.h"
#include "test/c++/tools/simulators/MMMSimulator.h"
#include "test/c++/tools/dp/DynamicProgrammingPortfolio.h"
#include "test/c++/tools/dp/DynamicProgrammingPortfolioDist.h"
#include "test/c++/tools/dp/OptimizePortfolioDP.h"
#include "test/c++/tools/dp/SimulateDPPortfolio.h"
#include "test/c++/tools/dp/SimulateDPPortfolioDist.h"

using namespace Eigen ;
using namespace std;
using namespace reflow ;

double accuracyClose =  5.;

#if defined   __linux
#include <fenv.h>
#define enable_abort_on_floating_point_exception() feenableexcept(FE_DIVBYZERO | FE_INVALID)
#endif


/// For Clang < 3.7 (and above ?) to be compatible GCC 5.1 and above
namespace boost
{
namespace unit_test
{
namespace ut_detail
{
std::string normalize_test_case_name(const_string name)
{
    return (name[0] == '&' ? std::string(name.begin() + 1, name.size() - 1) : std::string(name.begin(), name.size()));
}
}
}
}


/// test the valorization of an option on a portfolio value
/// with investement in bond or risky assets modeled by Platen MMM Model
/// This is a case of a controlled process so usual regression methods can not be used
///
BOOST_AUTO_TEST_CASE(testMMMModelInOptimizationSimulation)
{
#ifdef USE_MPI
    boost::mpi::communicator world;
#endif
    // initial value for the market index (MSCI world february 1975)
    double initialValue =  687.017162081;
    int nbSimul = 50000;
    /// parameter of the MMM model (see Platen-Heath)
    double alpha0 = 21.3600578812 ; // scaling parameter
    double eta =  0.0541753484637; //
    double T = 1.; // number of years
    size_t nbStep = T * 12; // number of optimization date (equal time step)
    std::string nameArch = "StoreMPIMMM";
#ifdef USE_MPI
    nameArch += to_string(world.size());
#endif
    // forward
    bool bForward = false;
    std::shared_ptr< MMMSimulator>  sim = std::make_shared<MMMSimulator>(initialValue, alpha0, eta, T, nbStep, nbSimul, bForward, nameArch
#ifdef USE_MPI
                                          , world
#endif
                                                                        );

    // Determine max of the diffusion cone
    double maxCone = 0. ;
    double minCone = 0. ;
    {
        ArrayXXd partLast = sim->getParticles();
        maxCone = partLast.maxCoeff();
        minCone = partLast.minCoeff();
    }
    // mesh
    int nMesh = 50;
    ArrayXi nbMesh = ArrayXi::Constant(1, nMesh);
    // grid
    double lowValue = static_cast<int>(minCone / 100) * 100.;
    ArrayXd lowValues = ArrayXd::Constant(1, lowValue);
    double highValue = static_cast<int>(maxCone / 100 + 1) * 100;
    double stepForGrid = 40.;
    ArrayXd stepGrid = ArrayXd::Constant(1, stepForGrid);
    int nbSGrid = static_cast<int>((highValue - lowValue) / stepForGrid) + 1;
    ArrayXi nbStepGrid = ArrayXi::Constant(1, nbSGrid);
#ifdef USE_MPI
    if (world.rank() == 0)
#endif
        std::cout << " Low value for mesh " << lowValue << " nbStep for grid " << nbSGrid << " highValue for grid " << highValue << std::endl ;
    ArrayXi p_poly =  ArrayXi::Constant(1, 2);
    shared_ptr<RegularLegendreGrid> grid = make_shared<RegularLegendreGrid>(lowValues, stepGrid, nbStepGrid, p_poly);
    // initial portfolio value
    ArrayXd initialPortfolio = ArrayXd::Constant(1, initialValue);
    std::string fileToDump = "CondExpMMM";
#ifdef USE_MPI
    fileToDump += to_string(world.size());
#endif
    // discretization of portfolio composition  (0,1)
    // with 1  at each time invest all in risky asset or bond
    int nbStepPort = 1;
    shared_ptr< OptimizePortfolioDP >optimizer = make_shared< OptimizePortfolioDP>(nbStepPort);

    // define final pay off
    // here a put on the portfolio value
    double strike = 800 ;
    auto payOff([strike](const int &, const Eigen::ArrayXd & p_asset, const Eigen::ArrayXd &)
    {
        return std::max(strike - p_asset(0), 0.);
    });

    std::function<double(const int &, const Eigen::ArrayXd &, const Eigen::ArrayXd &)>   vFunction(std::cref(payOff)) ;

    // link the simulations to the optimizer
    optimizer->setSimulator(sim);

    double valueOptimDist = 0 ;
    {
        std::shared_ptr<boost::timer::auto_cpu_timer> t;
#ifdef USE_MPI
        if (world.rank() == 0)
#endif
            t = std::make_shared<boost::timer::auto_cpu_timer>();
        valueOptimDist =  DynamicProgrammingPortfolio(grid, optimizer, nbMesh, vFunction, initialPortfolio,  fileToDump
#ifdef USE_MPI
                          , world
#endif
                                                     );
#ifdef USE_MPI
        if (world.rank() == 0)
#endif
            std::cout << " Value in optimization " << valueOptimDist << std::endl ;
#ifdef USE_MPI
        world.barrier();
#endif
    }

    // New simulator to go forward (issuing more simulations...)
    int nbSimulForward = 200000;
    bForward = true;
    std::shared_ptr< MMMSimulator>  simForward = std::make_shared<MMMSimulator>(initialValue, alpha0, eta, T, nbStep, nbSimulForward, bForward, nameArch
#ifdef USE_MPI
            , world
#endif
                                                                               );
    // link the simulations to the optimizer
    optimizer->setSimulator(simForward);
    double valueSimuDist =  SimulateDPPortfolio(grid, optimizer, vFunction, initialPortfolio,  fileToDump
#ifdef USE_MPI
                            , world
#endif
                                               ) ;
#ifdef USE_MPI
    if (world.rank() == 0)
#endif
        std::cout << "Simulation " << valueSimuDist << std::endl ;

#ifdef USE_MPI
    if (world.rank() == 0)
#endif
        BOOST_CHECK_CLOSE(valueSimuDist, valueOptimDist, accuracyClose);
}


#ifdef USE_MPI

// The p_bOneFile is set to true for one file to dump the control
// If set to false, each processor dump its control on its own file
void testDistribution(const bool &p_bOneFile)
{
    boost::mpi::communicator world;
    world.barrier();
    // initial value for the market index (MSCI world february 1975)
    double initialValue =  687.017162081;
    int nbSimul = 50000;
    /// parameter of the MMM model (see Platen-Heath)
    double alpha0 = 21.3600578812 ; // scaling parameter
    double eta =  0.0541753484637; //
    double T = 1.; // number of years
    size_t nbStep = T * 12; // number of optimization date (equal time step)
    std::string nameArch = "StoreMMMMDI" +   to_string(world.size()) ;
    // forward
    bool bForward = false;
    std::shared_ptr< MMMSimulator>  sim = std::make_shared<MMMSimulator>(initialValue, alpha0, eta, T, nbStep, nbSimul, bForward, nameArch, world);

    // Determine max of the diffusion cone
    double maxCone = 0. ;
    double minCone = 0. ;
    {
        ArrayXXd partLast = sim->getParticles();
        maxCone = partLast.maxCoeff();
        minCone = partLast.minCoeff();
    }
    // mesh
    int nMesh = 50;
    ArrayXi nbMesh = ArrayXi::Constant(1, nMesh);
    // grid
    double lowValue = static_cast<int>(minCone / 100) * 100.;
    ArrayXd lowValues = ArrayXd::Constant(1, lowValue);
    double highValue = static_cast<int>(maxCone / 100 + 1) * 100;
    double stepForGrid = 40.;
    ArrayXd stepGrid = ArrayXd::Constant(1, stepForGrid);
    int nbSGrid = static_cast<int>((highValue - lowValue) / stepForGrid) + 1;
    ArrayXi nbStepGrid = ArrayXi::Constant(1, nbSGrid);
    ArrayXi p_poly =  ArrayXi::Constant(1, 2);
    shared_ptr<RegularLegendreGrid> grid = make_shared<RegularLegendreGrid>(lowValues, stepGrid, nbStepGrid, p_poly);
    // initial portfolio value
    ArrayXd initialPortfolio = ArrayXd::Constant(1, initialValue);
    std::string fileToDump = "CondExp" +   to_string(world.size());
    // discretization of portfolio composition  (0,1)
    // with 1  at each time invest all in risky asset or bond
    int nbStepPort = 1;
    shared_ptr< OptimizePortfolioDP >optimizer = make_shared< OptimizePortfolioDP>(nbStepPort);

    // define final pay off
    // here a put on the portfolio value
    double strike = 800 ;
    auto payOff([strike](const int &, const Eigen::ArrayXd & p_asset, const Eigen::ArrayXd &)
    {
        return std::max(strike - p_asset(0), 0.);
    });

    std::function<double(const int &, const Eigen::ArrayXd &, const Eigen::ArrayXd &)>   vFunction(std::cref(payOff)) ;

    // link the simulations to the optimizer
    optimizer->setSimulator(sim);

    // use a single file to dump
    double valueOptimDist = 0 ;
    {
        std::shared_ptr<boost::timer::auto_cpu_timer> t;
        if (world.rank() == 0)
            t = std::make_shared<boost::timer::auto_cpu_timer>();
        valueOptimDist =  DynamicProgrammingPortfolioDist(grid, optimizer, nbMesh, vFunction, initialPortfolio,  fileToDump, p_bOneFile, world);
        if (world.rank() == 0)
            std::cout << " Value in optimization " << valueOptimDist << std::endl ;
        world.barrier();
    }

    // New simulator to go forward (issuing more simulations...)
    int nbSimulForward = 200000;
    bForward = true;
    std::shared_ptr< MMMSimulator>  simForward = std::make_shared<MMMSimulator>(initialValue, alpha0, eta, T, nbStep, nbSimulForward, bForward, nameArch
#ifdef USE_MPI
            , world
#endif
                                                                               );
    // link the simulations to the optimizer
    optimizer->setSimulator(simForward);
    double valueSimuDist =  SimulateDPPortfolioDist(grid, optimizer, vFunction, initialPortfolio,  fileToDump, p_bOneFile, world) ;
    if (world.rank() == 0)
    {
        std::cout << "Simulation " << valueSimuDist << std::endl ;
        BOOST_CHECK_CLOSE(valueSimuDist, valueOptimDist, accuracyClose);
    }
    world.barrier();
}

/// same as before by distribution of the calculation grid
/// Dump on one file
BOOST_AUTO_TEST_CASE(testMMMModelInOptimizationSimulationMPIOneFile)
{
    testDistribution(true);
}

/// same as before by distribution of the calculation grid
/// Dump on multiple files : each processor has its own file
BOOST_AUTO_TEST_CASE(testMMMModelInOptimizationSimulationMPIMultipleFiles)
{
    testDistribution(false);
}


#endif



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
