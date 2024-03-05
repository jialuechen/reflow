// Copyright (C) 2016 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#include <fstream>
#include <memory>
#include <functional>
#ifdef USE_MPI
#include "boost/mpi.hpp"
#endif
#include <boost/lexical_cast.hpp>
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "libflow/core/utils/eigenGeners.h"
#include "libflow/core/grids/SpaceGrid.h"
#include "libflow/semilagrangien/InitialValue.h"
#include "libflow/semilagrangien/TransitionStepSemilagrang.h"
#include "libflow/semilagrangien/OptimizerSLBase.h"

using namespace Eigen ;
using namespace std;
using namespace std::placeholders;


pair< double, double>  semiLagrangianTime(const shared_ptr<libflow::SpaceGrid> &p_grid,
        const shared_ptr<libflow::OptimizerSLBase > &p_optimize,
        const function<double(const int &, const ArrayXd &)>   &p_funcInitialValue,
        const function<double(const double &, const int &, const ArrayXd &)>   &p_timeBoundaryFunc,
        const double &p_step,
        const int &p_nStep,
        const ArrayXd &p_point,
        const int &p_initialRegime,
        const function<double(const double &, const ArrayXd &)> &p_funcSolution,
        const string   &p_fileToDump
#ifdef USE_MPI
        , const boost::mpi::communicator &p_world
#endif
                                         )
{
    // final values
    vector< shared_ptr< ArrayXd > >  valuesNext = libflow::InitialValue(p_grid, p_optimize->getNbRegime())(p_funcInitialValue);
    std::shared_ptr<gs::BinaryFileArchive> ar;
    int rank = 0 ;
#ifdef USE_MPI
    rank = p_world.rank();
#endif
    if (rank == 0)
        ar = std::make_shared<gs::BinaryFileArchive>(p_fileToDump.c_str(), "w");
    // name for object in archive
    string nameAr = "Continuation";
    // iterate on tim steps
    for (int iStep = 0; iStep < p_nStep ; ++iStep)
    {
        // boundary function
        function<double(const int &, const ArrayXd &)> boundaryFunc = bind(p_timeBoundaryFunc, (iStep + 1) * p_step, _1, _2);
        // transition object
        libflow::TransitionStepSemilagrang transStep(p_grid, p_grid, p_optimize
#ifdef USE_MPI
                , p_world
#endif
                                                  );
        pair< vector< shared_ptr< ArrayXd > >, vector< shared_ptr< ArrayXd > > > valuesPrevAndControl = transStep.oneStep(valuesNext, (iStep + 1) * p_step, boundaryFunc);
        // dump continuation values
        transStep.dumpValues(ar, nameAr, iStep, valuesNext, valuesPrevAndControl.second);
        valuesNext = valuesPrevAndControl.first;
    }

    // Spectral interpolator, iterator
    shared_ptr<libflow::InterpolatorSpectral> gridInterpol =  p_grid->createInterpolatorSpectral(*valuesNext[p_initialRegime]);
    shared_ptr< libflow::GridIterator> iterGrid = p_grid->getGridIterator();
    double errMax = 0. ;
    function<double(const ArrayXd &)> fSolution = bind(p_funcSolution, p_nStep * p_step, _1);
    ArrayXd pointMax(p_grid->getDimension());
    while (iterGrid->isValid())
    {
        ArrayXd pointCoord =  iterGrid->getCoordinate();
        double val =  gridInterpol->apply(pointCoord);
        double errLoc = fabs(val - fSolution(pointCoord));
        if (errLoc > errMax)
        {
            errMax = errLoc;
            pointMax = pointCoord;
        }
        iterGrid->next();
    }
    // interpolate at the initial stock point and initial regime
    return make_pair(gridInterpol->apply(p_point), errMax);
}
