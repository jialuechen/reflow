
#ifdef USE_MPI
#include <fstream>
#include <memory>
#include <functional>
#include <boost/lexical_cast.hpp>
#include <boost/mpi.hpp>
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "libflow/core/grids/FullGrid.h"
#include "libflow/regression/BaseRegression.h"
#include "libflow/dp/FinalStepDPCutDist.h"
#include "libflow/dp/TransitionStepRegressionDPCutDist.h"
#include "libflow/core/parallelism/reconstructProc0Mpi.h"
#include "libflow/dp/OptimizerDPCutBase.h"
#include "libflow/dp/SimulatorDPBase.h"


using namespace std;
using namespace Eigen;

double  DynamicProgrammingByRegressionCutDist(const shared_ptr<libflow::FullGrid> &p_grid,
        const shared_ptr<libflow::OptimizerDPCutBase > &p_optimize,
        shared_ptr<libflow::BaseRegression> &p_regressor,
        const function< ArrayXd(const int &, const ArrayXd &, const ArrayXd &)>   &p_funcFinalValue,
        const ArrayXd &p_pointStock,
        const int &p_initialRegime,
        const string   &p_fileToDump,
        const bool &p_bOneFile,
        const boost::mpi::communicator &p_world)
{
    // from the optimizer get back the simulator
    shared_ptr< libflow::SimulatorDPBase> simulator = p_optimize->getSimulator();
    // final values
    vector< shared_ptr< ArrayXXd > >  valueCutsNext = libflow::FinalStepDPCutDist(p_grid, p_optimize->getNbRegime(), p_optimize->getDimensionToSplit(), p_world)(p_funcFinalValue, simulator->getParticles().array());
    // dump
    string toDump = p_fileToDump ;
    // test if one file generated
    if (!p_bOneFile)
        toDump +=  "_" + boost::lexical_cast<string>(p_world.rank());
    shared_ptr<gs::BinaryFileArchive> ar;
    if ((!p_bOneFile) || (p_world.rank() == 0))
        ar = make_shared<gs::BinaryFileArchive>(toDump.c_str(), "w");
    // name for object in archive
    string nameAr = "Continuation";
    for (int iStep = 0; iStep < simulator->getNbStep(); ++iStep)
    {
        ArrayXXd asset = simulator->stepBackwardAndGetParticles();
        // conditional expectation operator
        p_regressor->updateSimulations(((iStep == (simulator->getNbStep() - 1)) ? true : false), asset);
        // transition object
        libflow::TransitionStepRegressionDPCutDist transStep(p_grid, p_grid, p_optimize, p_world);
        vector< shared_ptr< ArrayXXd > > valueCuts  = transStep.oneStep(valueCutsNext, p_regressor);
        transStep.dumpContinuationCutsValues(ar, nameAr, iStep, valueCutsNext, p_regressor, p_bOneFile);
        valueCutsNext = valueCuts;
    }
    // reconstruct a small grid for interpolation
    ArrayXd  valSim = libflow::reconstructProc0Mpi(p_pointStock, p_grid, valueCutsNext[p_initialRegime], p_optimize->getDimensionToSplit(), p_world);
    return ((p_world.rank() == 0) ? valSim.head(simulator->getNbSimul()).mean() : 0.);

}
#endif
