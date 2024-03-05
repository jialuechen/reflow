
#include <fstream>
#include <memory>
#include <functional>
#include <boost/lexical_cast.hpp>
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "libflow/core/grids/SparseSpaceGrid.h"
#include "libflow/semilagrangien/InitialValue.h"
#include "libflow/semilagrangien/TransitionStepSemilagrang.h"
#include "OptimizeSLEmissive.h"

using namespace Eigen ;
using namespace std;
using namespace std::placeholders;


void  semiLagrangTimeNonEmissiveSparse(const shared_ptr<libflow::SparseSpaceGrid> &p_grid,
                                       const shared_ptr<libflow::OptimizeSLEmissive > &p_optimize,
                                       const function<double(const int &, const ArrayXd &)>   &p_funcInitialValue,
                                       const function<double(const double &, const int &, const ArrayXd &)>   &p_timeBoundaryFunc,
                                       const double &p_step,
                                       const int &p_nStep,
                                       const string   &p_fileToDump
#ifdef USE_MPI
                                       , const boost::mpi::communicator &p_world
#endif
                                      )
{
    // final values
    vector< shared_ptr< ArrayXd > >  valuesNext = libflow::InitialValue(dynamic_pointer_cast<libflow::SpaceGrid>(p_grid), p_optimize->getNbRegime())(p_funcInitialValue);
    std::shared_ptr<gs::BinaryFileArchive> ar;
    // to store function value
    ArrayXd vFunc1, vFunc2;
    // to store control
    ArrayXd vControl;
#ifdef USE_MPI
    if (p_world.rank() == 0)
#endif
        ar = std::make_shared<gs::BinaryFileArchive>(p_fileToDump.c_str(), "w");
    // name for object in archive
    string nameAr = "Continuation";
    // iterate on time steps
    for (int iStep = 0; iStep < p_nStep ; ++iStep)
    {
        std::cout << " step " << iStep << std::endl  ;
        // boundary function
        function<double(const int &, const ArrayXd &)> boundaryFunc = bind(p_timeBoundaryFunc, (iStep + 1) * p_step, _1, _2);
        // transition object
#ifdef USE_MPI
        libflow::TransitionStepSemilagrang transStep(p_grid, p_grid, static_pointer_cast<libflow::OptimizerSLBase>(p_optimize), p_world);
#else
        libflow::TransitionStepSemilagrang transStep(p_grid, p_grid, static_pointer_cast<libflow::OptimizerSLBase>(p_optimize));
#endif
        pair< vector< shared_ptr< ArrayXd > >, vector< shared_ptr< ArrayXd > > > valuesPrevAndControl = transStep.oneStep(valuesNext, (iStep + 1) * p_step, boundaryFunc);
        // dump continuation values
        transStep.dumpValues(ar, nameAr, iStep, valuesNext, valuesPrevAndControl.second);
        valuesNext = valuesPrevAndControl.first;
        if (iStep == (p_nStep - 1))
        {
            vFunc1 =  *valuesPrevAndControl.first[0];
            vFunc2 =  *valuesPrevAndControl.first[1];
            vControl = *valuesPrevAndControl.second[0];
        }
    }

    // use the reconstructed solution
#ifdef USE_MPI
    if (p_world.rank() == 0)
    {
#endif
        // Spectral interpolator, iterator
        shared_ptr<libflow::InterpolatorSpectral> gridInterpol1 =  p_grid->createInterpolatorSpectral(vFunc1);
        shared_ptr<libflow::InterpolatorSpectral> gridInterpol2 =  p_grid->createInterpolatorSpectral(vFunc2);
        shared_ptr<libflow::InterpolatorSpectral> gridInterpControl = p_grid->createInterpolatorSpectral(vControl);
        // DUMP
        fstream fileStream("SortieSLSparse", ios::out);
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

#ifdef USE_MPI
    }
#endif

}
