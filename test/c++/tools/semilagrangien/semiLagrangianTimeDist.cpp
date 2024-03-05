// Copyright (C) 2016 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#ifdef USE_MPI
#include <fstream>
#include <memory>
#include <functional>
#include <boost/lexical_cast.hpp>
#include <boost/mpi.hpp>
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "libflow/core/grids/FullGrid.h"
#include "libflow/semilagrangien/InitialValueDist.h"
#include "libflow/semilagrangien/TransitionStepSemilagrangDist.h"
#include "libflow/semilagrangien/OptimizerSLBase.h"

using namespace Eigen ;
using namespace std;
using namespace std::placeholders;


std::pair< double, double>  semiLagrangianTimeDist(const shared_ptr<libflow::FullGrid> &p_grid,
        const shared_ptr<libflow::OptimizerSLBase > &p_optimize,
        const function<double(const int &, const ArrayXd &)>   &p_funcInitialValue,
        const function<double(const double &, const int &, const ArrayXd &)>   &p_timeBoundaryFunc,
        const double &p_step,
        const int &p_nStep,
        const ArrayXd &p_point,
        const int &p_initialRegime,
        const function<double(const double &, const ArrayXd &)> &p_funcSolution,
        const string   &p_fileToDump,
        const bool &p_bOneFile,
        const boost::mpi::communicator &p_world)
{
    // final values
    vector< shared_ptr< ArrayXd > >  valuesNext = libflow::InitialValueDist(p_grid, p_optimize->getNbRegime(), p_optimize->getDimensionToSplit(), p_world)(p_funcInitialValue);
    // dump
    string toDump = p_fileToDump ;
    // test if one file generated
    if (!p_bOneFile)
        toDump +=  "_" + boost::lexical_cast<string>(p_world.rank());
    std::shared_ptr<gs::BinaryFileArchive> ar;
    if ((!p_bOneFile) || (p_world.rank() == 0))
        ar = std::make_shared<gs::BinaryFileArchive>(toDump.c_str(), "w");
    // name for object in archive
    string nameAr = "Continuation";
    // reconstruct on processor 0
    vector< shared_ptr< ArrayXd > >  valuesRecons;
    // iterate on tim steps
    for (int iStep = 0; iStep < p_nStep ; ++iStep)
    {
        // boundary function
        function<double(const int &, const ArrayXd &)> boundaryFunc = bind(p_timeBoundaryFunc, (iStep + 1) * p_step, _1, _2);
        // transition object
        libflow::TransitionStepSemilagrangDist transStep(p_grid, p_grid, p_optimize, p_world);
        pair< vector< shared_ptr< ArrayXd > >, vector< shared_ptr< ArrayXd > > > valuesPrevAndControl =  transStep.oneStep(valuesNext, (iStep + 1) * p_step, boundaryFunc);
        // dump continuation values
        transStep.dumpValues(ar, nameAr, iStep, valuesNext, valuesPrevAndControl.second, p_bOneFile);
        valuesNext =  valuesPrevAndControl.first;
        // last step reconstruct solution on one processor
        if (iStep == (p_nStep - 1))
        {
            transStep.reconstructOnProc0(valuesNext, valuesRecons);
        }
    }

    // use the reconstructed solution to calculate max.
    // another more effective way could have been to get back grids use by current processor, calculate the max error on each processor
    // and take the max of these values
    if (p_world.rank() == 0)
    {
        // Spectral interpolator, iterator
        shared_ptr<libflow::InterpolatorSpectral> gridInterpol =  p_grid->createInterpolatorSpectral(*valuesRecons[p_initialRegime]);
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
    else
    {
        return make_pair(0., 0.);
    }
}
#endif
