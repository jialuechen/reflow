#ifdef USE_MPI
#include <fstream>
#include <memory>
#include <functional>
#include <boost/lexical_cast.hpp>
#include <boost/mpi.hpp>
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "reflow/core/grids/FullGrid.h"
#include "reflow/regression/BaseRegression.h"
#include "reflow/dp/FinalStepDPDist.h"
#include "reflow/dp/TransitionStepMultiStageRegressionDPDist.h"
#include "reflow/core/parallelism/reconstructProc0Mpi.h"
#include "reflow/dp/OptimizerMultiStageDPBase.h"
#include "reflow/dp/SimulatorMultiStageDPBase.h"


using namespace std;

double  DynamicProgrammingByRegressionMultiStageDist(const shared_ptr<reflow::FullGrid> &p_grid,
        const shared_ptr<reflow::OptimizerMultiStageDPBase > &p_optimize,
        shared_ptr<reflow::BaseRegression> &p_regressor,
        const function<double(const int &, const Eigen::ArrayXd &, const Eigen::ArrayXd &)>   &p_funcFinalValue,
        const Eigen::ArrayXd &p_pointStock,
        const int &p_initialRegime,
        const string   &p_fileToDump,
        const bool &p_bOneFile,
        const boost::mpi::communicator &p_world)
{
    // from the optimizer get back the simulator
    shared_ptr< reflow::SimulatorMultiStageDPBase> simulator = p_optimize->getSimulator();
    // final values
    vector< shared_ptr< Eigen::ArrayXXd > >  valuesNext = reflow::FinalStepDPDist(p_grid, p_optimize->getNbRegime(), p_optimize->getDimensionToSplit(), p_world)(p_funcFinalValue, simulator->getParticles().array());
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
    // Name for deterministic continuation in archive
    string nameArContValDet =  "ContinuationDet";
    for (int iStep = 0; iStep < simulator->getNbStep(); ++iStep)
    {
        Eigen::ArrayXXd asset = simulator->stepBackwardAndGetParticles();
        // conditional expectation operator
        p_regressor->updateSimulations(((iStep == (simulator->getNbStep() - 1)) ? true : false), asset);
        // transition object : constructor with dump on archive of deterministic continuation/Bellmna values
        string toStorBellDet = nameArContValDet + boost::lexical_cast<string>(iStep);
        // transition object
        reflow::TransitionStepMultiStageRegressionDPDist transStep(p_grid, p_grid, p_optimize, ar, toStorBellDet, p_bOneFile, p_world);
        // dump stochastic continuation value
        vector< shared_ptr< Eigen::ArrayXXd > >  values  = transStep.oneStep(valuesNext, p_regressor);
        transStep.dumpContinuationValues(ar, nameAr, iStep, valuesNext,  p_regressor, p_bOneFile);
        valuesNext = values;
    }
    // reconstruct a small grid for interpolation
    return reflow::reconstructProc0Mpi(p_pointStock, p_grid, valuesNext[p_initialRegime], p_optimize->getDimensionToSplit(), p_world).mean();

}
#endif
