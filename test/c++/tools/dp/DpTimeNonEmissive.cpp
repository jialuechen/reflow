
#ifdef USE_MPI
#include <fstream>
#include <memory>
#include <functional>
#include <boost/lexical_cast.hpp>
#include <boost/mpi.hpp>
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "reflow/core/grids/FullGrid.h"
#include "reflow/core/parallelism/ParallelComputeGridSplitting.h"
#include "reflow/regression/BaseRegression.h"
#include "reflow/dp/FinalStepDPDist.h"
#include "reflow/dp/TransitionStepRegressionDPDist.h"
#include "reflow/core/parallelism/reconstructProc0Mpi.h"
#include "reflow/dp/OptimizerDPBase.h"
#include "reflow/dp/SimulatorDPBase.h"

using namespace std;
using namespace Eigen;

void  DpTimeNonEmissive(const shared_ptr<reflow::FullGrid> &p_grid,
                        const shared_ptr<reflow::OptimizerDPBase > &p_optimize,
                        const shared_ptr<reflow::BaseRegression> &p_regressor,
                        const std::function<double(const int &, const ArrayXd &, const ArrayXd &)>  &p_funcFinalValue,
                        const std::string   &p_fileToDump,
                        const boost::mpi::communicator &p_world)
{
    // from the optimizer get back the simulation
    shared_ptr< reflow::SimulatorDPBase> simulator = p_optimize->getSimulator();
    // final values
    vector< shared_ptr< ArrayXXd > >  valuesNext = reflow::FinalStepDPDist(p_grid, p_optimize->getNbRegime(), p_optimize->getDimensionToSplit(), p_world)(p_funcFinalValue, simulator->getParticles().array());
    // dump
    std::shared_ptr<gs::BinaryFileArchive> ar;
    if (p_world.rank() == 0)
        ar = std::make_shared<gs::BinaryFileArchive>(p_fileToDump.c_str(), "w");
    // store optimal control
    vector< shared_ptr< ArrayXXd > > optimalControl;
    vector< shared_ptr< ArrayXXd > > valuesRecons;
    // name for object in archive
    string nameAr = "Continuation";
    for (int iStep = 0; iStep < simulator->getNbStep(); ++iStep)
    {
        if (p_world.rank() == 0)
            std::cout << " Step " << iStep << std::endl ;
        ArrayXXd  asset = simulator->stepBackwardAndGetParticles();
        // conditional expectation operator
        p_regressor->updateSimulations(((iStep == (simulator->getNbStep() - 1)) ? true : false), asset);
        // transition object
        reflow::TransitionStepRegressionDPDist transStep(p_grid, p_grid, p_optimize, p_world);
        pair< vector< shared_ptr< ArrayXXd > >, vector< shared_ptr< ArrayXXd > > > valuesPrevAndControl  = transStep.oneStep(valuesNext, p_regressor);
        transStep.dumpContinuationValues(ar, nameAr, iStep, valuesNext, valuesPrevAndControl.second, p_regressor, true);
        valuesNext = valuesPrevAndControl.first;
        if (iStep == (simulator->getNbStep() - 1))
        {
            // store optimal control
            transStep.reconstructOnProc0(valuesNext, valuesRecons);
            transStep.reconstructOnProc0(valuesPrevAndControl.second, optimalControl);
        }
    }

    if (p_world.rank() == 0)
    {
        // DUMP
        fstream fileStream("SortieDP", ios::out);
        std::vector<std::array<double, 2> > extrem = p_grid->getExtremeValues();
        ArrayXd ptInterp(2);
        double step1 = (extrem[0][1] - extrem[0][0]) / 100.;
        double step2 = (extrem[1][1] - extrem[1][0]) / 100.;
        for (int j = 0; j <=  100; ++j)
            for (int i = 0; i <= 100; ++i)
            {
                ptInterp(0) = extrem[0][0] + step1 * j;
                ptInterp(1) = extrem[1][0] + step2 * i;
                // Interpolator, iterator
                shared_ptr<reflow::Interpolator> gridInterpol =  p_grid->createInterpolator(ptInterp);
                fileStream << ptInterp(0) << " " << ptInterp(1) <<  " " << gridInterpol->applyVec(*valuesRecons[0]).mean() <<  " " << gridInterpol->applyVec(*valuesRecons[1]).mean() <<  " " <<  gridInterpol->applyVec(*optimalControl[0]).mean() << std::endl ;
            }
        fileStream.close();
    }
}
#endif
