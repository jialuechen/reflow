#include <fstream>
#include <memory>
#include <functional>
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <boost/lexical_cast.hpp>
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "reflow/regression/BaseRegression.h"
#include "reflow/core/grids/FullGrid.h"
#include "reflow/dp/FinalStepDPCut.h"
#include "reflow/dp/TransitionStepRegressionDPCut.h"
#include "reflow/dp/OptimizerDPCutBase.h"

using namespace std;
using namespace Eigen;


double  DynamicProgrammingByRegressionCut(const shared_ptr<reflow::FullGrid> &p_grid,
        const shared_ptr<reflow::OptimizerDPCutBase > &p_optimize,
        const shared_ptr<reflow::BaseRegression> &p_regressor,
        const function< ArrayXd(const int &, const ArrayXd &, const ArrayXd &)>  &p_funcFinalValue,
        const ArrayXd &p_pointStock,
        const int &p_initialRegime,
        const string   &p_fileToDump
#ifdef USE_MPI
        , const boost::mpi::communicator &p_world
#endif
                                         )
{
    // from the optimizer get back the simulator
    shared_ptr< reflow::SimulatorDPBase> simulator = p_optimize->getSimulator();
    // final cut values
    vector< shared_ptr< ArrayXXd > >  valueCutsNext = reflow::FinalStepDPCut(p_grid, p_optimize->getNbRegime())(p_funcFinalValue, simulator->getParticles().array());
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
        reflow::TransitionStepRegressionDPCut  transStep(p_grid, p_grid, p_optimize
#ifdef USE_MPI
                , p_world
#endif
                                                       );
        vector< shared_ptr< ArrayXXd > > valueCuts = transStep.oneStep(valueCutsNext, p_regressor);
        // dump continuation values
        transStep.dumpContinuationCutsValues(ar, nameAr, iStep, valueCutsNext,  p_regressor);
        valueCutsNext = valueCuts;
    }
    // interpolate at the initial stock point and initial regime
    // Only keep the first nbSimul values  of the cuts  (other components are  for derivatives with respect to stock value)
    return (p_grid->createInterpolator(p_pointStock)->applyVec(*valueCutsNext[p_initialRegime]).head(simulator->getNbSimul())).mean();
}
