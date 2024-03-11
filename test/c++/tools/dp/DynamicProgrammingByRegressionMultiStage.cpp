#include <fstream>
#include <memory>
#include <functional>
#include <boost/lexical_cast.hpp>
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <boost/lexical_cast.hpp>
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "libflow/regression/BaseRegression.h"
#include "libflow/core/grids/FullGrid.h"
#include "libflow/dp/FinalStepDP.h"
#include "libflow/dp/TransitionStepMultiStageRegressionDP.h"
#include "libflow/dp/OptimizerMultiStageDPBase.h"

using namespace std;


double  DynamicProgrammingByRegressionMultiStage(const shared_ptr<libflow::FullGrid> &p_grid,
        const shared_ptr<libflow::OptimizerMultiStageDPBase > &p_optimize,
        const shared_ptr<libflow::BaseRegression> &p_regressor,
        const function<double(const int &, const Eigen::ArrayXd &, const Eigen::ArrayXd &)>  &p_funcFinalValue,
        const Eigen::ArrayXd &p_pointStock,
        const int &p_initialRegime,
        const string   &p_fileToDump
#ifdef USE_MPI
        , const boost::mpi::communicator &p_world
#endif

                                                )
{
    // from the optimizer get back the simulator
    shared_ptr< libflow::SimulatorMultiStageDPBase> simulator = p_optimize->getSimulator();
    // final values
    vector< shared_ptr< Eigen::ArrayXXd > >  valuesNext = libflow::FinalStepDP(p_grid, p_optimize->getNbRegime())(p_funcFinalValue, simulator->getParticles().array());
    shared_ptr<gs::BinaryFileArchive> ar = make_shared<gs::BinaryFileArchive>(p_fileToDump.c_str(), "w");
    // name for object in archive
    string nameAr = "Continuation";
    // Name for deterministic continuation in archive
    string nameArContValDet =  "ContinuationDet";
    // iterate on time steps
    for (int iStep = 0; iStep < simulator->getNbStep(); ++iStep)
    {
        Eigen::ArrayXXd asset = simulator->stepBackwardAndGetParticles();
        // conditional expectation operator
        p_regressor->updateSimulations(((iStep == (simulator->getNbStep() - 1)) ? true : false), asset);
        // transition object : constructor with dump on archive of deterministic Bellman values
        string toStorBellDet = nameArContValDet + boost::lexical_cast<string>(iStep);
        libflow::TransitionStepMultiStageRegressionDP transStep(p_grid, p_grid, p_optimize, ar, toStorBellDet
#ifdef USE_MPI
                , p_world
#endif
                                                             );

        vector< shared_ptr< Eigen::ArrayXXd > > values = transStep.oneStep(valuesNext, p_regressor);
        // dump continuation values stochastic only
        transStep.dumpContinuationValues(ar, nameAr, iStep, valuesNext,  p_regressor);
        valuesNext = values;
    }
    // interpolate at the initial stock point and initial regime
    return (p_grid->createInterpolator(p_pointStock)->applyVec(*valuesNext[p_initialRegime])).mean();
}
