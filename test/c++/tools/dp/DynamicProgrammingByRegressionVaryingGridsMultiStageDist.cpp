

#ifdef USE_MPI
#include <fstream>
#include <memory>
#include <functional>
#include <boost/lexical_cast.hpp>
#include <boost/mpi.hpp>
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "libflow/core/utils/comparisonUtils.h"
#include "libflow/core/grids/FullGrid.h"
#include "libflow/regression/BaseRegression.h"
#include "libflow/dp/FinalStepDPDist.h"
#include "libflow/dp/TransitionStepMultiStageRegressionDPDist.h"
#include "libflow/core/parallelism/reconstructProc0Mpi.h"
#include "libflow/dp/OptimizerDPBase.h"
#include "libflow/dp/SimulatorDPBase.h"

using namespace std;

double  DynamicProgrammingByRegressionVaryingGridsMultiStageDist(const vector<double>    &p_timeChangeGrid,
        const vector<shared_ptr<libflow::FullGrid> >   &p_grids,
        const shared_ptr<libflow::OptimizerMultiStageDPBase > &p_optimize,
        shared_ptr<libflow::BaseRegression> &p_regressor,
        const function<double(const int &, const Eigen::ArrayXd &, const Eigen::ArrayXd &)>   &p_funcFinalValue,
        const Eigen::ArrayXd &p_pointStock,
        const int &p_initialRegime,
        const string   &p_fileToDump,
        const bool &p_bOneFile,
        const boost::mpi::communicator &p_world)
{
    // from the optimizer get back the simulation
    shared_ptr< libflow::SimulatorMultiStageDPBase> simulator = p_optimize->getSimulator();
    // identify last grid
    double currentTime = simulator->getCurrentStep();
    int iTime = p_timeChangeGrid.size() - 1;
    while (libflow::isStrictlyLesser(currentTime, p_timeChangeGrid[iTime]))
        iTime--;
    shared_ptr<libflow::FullGrid>  gridCurrent = p_grids[iTime];
    // final values
    vector< shared_ptr< Eigen::ArrayXXd > >  valuesNext = libflow::FinalStepDPDist(gridCurrent, p_optimize->getNbRegime(), p_optimize->getDimensionToSplit(), p_world)(p_funcFinalValue, simulator->getParticles().array());
    shared_ptr<libflow::FullGrid> gridPrevious = gridCurrent;
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
        // update grid
        currentTime = simulator->getCurrentStep();
        while (libflow::isStrictlyLesser(currentTime, p_timeChangeGrid[iTime]))
            iTime--;       // conditional expectation operator
        gridCurrent = p_grids[iTime];
        // conditional expectation operator
        p_regressor->updateSimulations(((iStep == (simulator->getNbStep() - 1)) ? true : false), asset);
        // transition object : constructor with dump on archive of deterministic continuation/Bellmna values
        string toStorBellDet = nameArContValDet + boost::lexical_cast<string>(iStep);
        libflow::TransitionStepMultiStageRegressionDPDist transStep(gridCurrent, gridPrevious, p_optimize, ar, toStorBellDet, p_bOneFile, p_world);
        vector< shared_ptr< Eigen::ArrayXXd > >  values =  transStep.oneStep(valuesNext, p_regressor);
        transStep.dumpContinuationValues(ar, nameAr, iStep, valuesNext,  p_regressor, p_bOneFile);
        valuesNext = values;
        gridPrevious = gridCurrent;
    }
    // reconstruct a small grid for interpolation
    return libflow::reconstructProc0Mpi(p_pointStock, gridPrevious, valuesNext[p_initialRegime], p_optimize->getDimensionToSplit(), p_world).mean();

}
#endif
