// Copyright (C) 2021 EDF
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
#include "libflow/core/grids/RegularSpaceIntGrid.h"
#include "libflow/regression/BaseRegression.h"
#include "libflow/dp/FinalStepZeroDist.h"
#include "libflow/dp/TransitionStepRegressionSwitchDist.h"
#include "libflow/core/parallelism/reconstructProc0ForIntMpi.h"
#include "libflow/dp/OptimizerSwitchBase.h"
#include "libflow/dp/SimulatorDPBase.h"


using namespace std;
using namespace Eigen;
using namespace libflow ;

double  DynamicProgrammingSwitchingByRegressionDist(const vector<shared_ptr<RegularSpaceIntGrid> > &p_grid,
        const shared_ptr< OptimizerSwitchBase > &p_optimize,
        const shared_ptr<BaseRegression> &p_regressor,
        const ArrayXi &p_pointState,
        const int &p_initialRegime,
        const string   &p_fileToDump,
        const boost::mpi::communicator &p_world
                                                   )
{
    // from the optimizer get back the simulator
    shared_ptr< SimulatorDPBase> simulator = p_optimize->getSimulator();
    // final values
    vector< shared_ptr< ArrayXXd > >  valuesNext = FinalStepZeroDist<RegularSpaceIntGrid>(p_grid,  p_optimize->getDimensionToSplit(), p_world)(simulator->getNbSimul());
    // dump
    // test if one file generated
    shared_ptr<gs::BinaryFileArchive> ar;
    if (p_world.rank() == 0)
        ar = make_shared<gs::BinaryFileArchive>(p_fileToDump.c_str(), "w");
    // name for object in archive
    string nameAr = "Continuation";
    for (int iStep = 0; iStep < simulator->getNbStep(); ++iStep)
    {
        ArrayXXd asset = simulator->stepBackwardAndGetParticles();
        // conditional expectation operator
        p_regressor->updateSimulations(((iStep == (simulator->getNbStep() - 1)) ? true : false), asset);
        // transition object
        TransitionStepRegressionSwitchDist transStep(p_grid, p_grid, p_optimize, p_world);
        vector< shared_ptr< ArrayXXd > > values  = transStep.oneStep(valuesNext, p_regressor);
        transStep.dumpContinuationValues(ar, nameAr, iStep, valuesNext, p_regressor);
        valuesNext = values;
    }
    // reconstruct a small grid for interpolation
    return reconstructProc0ForIntMpi(p_pointState, p_grid[p_initialRegime], valuesNext[p_initialRegime], p_optimize->getDimensionToSplit()[p_initialRegime], p_world).mean();

}
#endif
