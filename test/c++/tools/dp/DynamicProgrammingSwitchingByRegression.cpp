
#include <fstream>
#include <memory>
#include <functional>
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <boost/lexical_cast.hpp>
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "libflow/regression/BaseRegression.h"
#include "libflow/core/grids/RegularSpaceIntGrid.h"
#include "libflow/dp/TransitionStepRegressionSwitch.h"
#include "libflow/dp/OptimizerSwitchBase.h"


using namespace std;
using namespace Eigen;
using namespace  libflow;

double  DynamicProgrammingSwitchingByRegression(const vector<shared_ptr<RegularSpaceIntGrid> > &p_grid,
        const shared_ptr< OptimizerSwitchBase > &p_optimize,
        const shared_ptr<BaseRegression> &p_regressor,
        const ArrayXi &p_pointState,
        const int &p_initialRegime,
        const string   &p_fileToDump
#ifdef USE_MPI
        , const boost::mpi::communicator &p_world
#endif
                                               )
{
    // from the optimizer get back the simulator
    shared_ptr< SimulatorDPBase> simulator = p_optimize->getSimulator();
    int nbRegime = p_grid.size();
    int nbSimul = simulator->getNbSimul();
    // final values
    vector< shared_ptr< ArrayXXd > >  valuesNext(p_grid.size());
    for (int iReg = 0; iReg < nbRegime; ++iReg)
        valuesNext[iReg] = make_shared<ArrayXXd>(ArrayXXd::Zero(nbSimul, p_grid[iReg]->getNbPoints()));

    shared_ptr<gs::BinaryFileArchive> ar = make_shared<gs::BinaryFileArchive>(p_fileToDump.c_str(), "w");
    // name for object in archive
    string nameAr = "Continuation";
    // iterate on time steps
    for (int iStep = 0; iStep < simulator->getNbStep(); ++iStep)
    {
        ArrayXXd asset = simulator->stepBackwardAndGetParticles();
        // conditional expectation operator
        p_regressor->updateSimulations(((iStep == (simulator->getNbStep() - 1)) ? true : false), asset);
        // transition object
        TransitionStepRegressionSwitch transStep(p_grid, p_grid, p_optimize
#ifdef USE_MPI
                , p_world
#endif
                                                );
        vector< shared_ptr< ArrayXXd > > values = transStep.oneStep(valuesNext, p_regressor);
        // dump continuation values
        transStep.dumpContinuationValues(ar, nameAr, iStep, valuesNext,  p_regressor);
        valuesNext = values;
    }
    // interpolate at the initial stock point and initial regime
    return (*valuesNext[p_initialRegime]).col(p_grid[p_initialRegime]->globCoordPerDimToLocal(p_pointState)).mean();
}
