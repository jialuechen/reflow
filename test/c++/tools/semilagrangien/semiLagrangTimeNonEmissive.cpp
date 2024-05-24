
#ifdef USE_MPI
#include <fstream>
#include <memory>
#include <functional>
#include <boost/lexical_cast.hpp>
#include <boost/mpi.hpp>
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "reflow/core/grids/FullGrid.h"
#include "reflow/semilagrangien/InitialValueDist.h"
#include "reflow/semilagrangien/TransitionStepSemilagrangDist.h"
#include "OptimizeSLEmissive.h"

using namespace Eigen ;
using namespace std;
using namespace std::placeholders;


void  semiLagrangTimeNonEmissive(const shared_ptr<reflow::FullGrid> &p_grid,
                                 const shared_ptr<reflow::OptimizeSLEmissive > &p_optimize,
                                 const function<double(const int &, const ArrayXd &)>   &p_funcInitialValue,
                                 const function<double(const double &, const int &, const ArrayXd &)>   &p_timeBoundaryFunc,
                                 const double &p_step,
                                 const int &p_nStep,
                                 const string   &p_fileToDump,
                                 const boost::mpi::communicator &p_world)
{
    // final values
    vector< shared_ptr< ArrayXd > >  valuesNext = reflow::InitialValueDist(p_grid, p_optimize->getNbRegime(), p_optimize->getDimensionToSplit(), p_world)(p_funcInitialValue);
    // dump
    std::shared_ptr<gs::BinaryFileArchive> ar;
    if (p_world.rank() == 0)
        ar = std::make_shared<gs::BinaryFileArchive>(p_fileToDump.c_str(), "w");
    // name for object in archive
    string nameAr = "Continuation";
    // reconstruct on processor 0
    vector< shared_ptr< ArrayXd > >  valuesRecons;
    // store optimal control
    vector< shared_ptr< ArrayXd > > optimalControl;
    // iterate on time steps
    for (int iStep = 0; iStep < p_nStep ; ++iStep)
    {
        if (p_world.rank() == 0)
            std::cout << "Step " << iStep << std::endl ;
        // boundary function
        function<double(const int &, const ArrayXd &)> boundaryFunc = bind(p_timeBoundaryFunc, (iStep + 1) * p_step, _1, _2);
        // transition object
        reflow::TransitionStepSemilagrangDist transStep(p_grid, p_grid, static_pointer_cast<reflow::OptimizerSLBase>(p_optimize), p_world);
        pair< vector< shared_ptr< ArrayXd > >, vector< shared_ptr< ArrayXd > > > valuesPrevAndControl = transStep.oneStep(valuesNext, (iStep + 1) * p_step, boundaryFunc);
        // dump continuation values
        transStep.dumpValues(ar, nameAr, iStep, valuesNext, valuesPrevAndControl.second, true);
        valuesNext = valuesPrevAndControl.first;
        if (iStep == (p_nStep - 1))
        {
            // store optimal control
            transStep.reconstructOnProc0(valuesNext, valuesRecons);
            transStep.reconstructOnProc0(valuesPrevAndControl.second, optimalControl);
        }
    }

    // use the reconstructed solution
    if (p_world.rank() == 0)
    {
        // Spectral interpolator, iterator
        shared_ptr<reflow::InterpolatorSpectral> gridInterpol1 =  p_grid->createInterpolatorSpectral(*valuesRecons[0]);
        shared_ptr<reflow::InterpolatorSpectral> gridInterpol2 =  p_grid->createInterpolatorSpectral(*valuesRecons[1]);
        shared_ptr<reflow::InterpolatorSpectral> gridInterpControl = p_grid->createInterpolatorSpectral(*optimalControl[0]);
        // DUMP
        fstream fileStream("SortieSL", ios::out);
        std::vector<std::array<double, 2> > extrem = p_grid->getExtremeValues();
        ArrayXd ptInterp(3);
        ptInterp(0) = 1.4;
        double step1 = (extrem[1][1] - extrem[1][0]) / 100.;
        double step2 = (extrem[2][1] - extrem[2][0]) / 100.;
        for (int j = 0; j <=  100; ++j)
            for (int i = 0; i <= 100; ++i)
            {
                ptInterp(1) = extrem[1][0] + step1 * j;
                ptInterp(2) = extrem[2][0] + step2 * i;
                fileStream << ptInterp(1) << " " << ptInterp(2) <<  " " << gridInterpol1->apply(ptInterp) <<  " " << gridInterpol2->apply(ptInterp) <<  " " << gridInterpControl->apply(ptInterp) << std::endl ;
            }
        fileStream.close();
    }

}
#endif
