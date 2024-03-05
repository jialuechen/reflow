
#include <fstream>
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <memory>
#include <functional>
#include <boost/lexical_cast.hpp>
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "libflow/core/grids/FullGrid.h"
#include "libflow/regression/LocalConstRegression.h"
#include "libflow/regression/GridAndRegressedValue.h"
#include "libflow/dp/FinalStepDP.h"
#include "libflow/dp/TransitionStepDP.h"
#include "test/c++/tools/dp/OptimizePortfolioDP.h"

using namespace std;
using namespace Eigen;

double  DynamicProgrammingPortfolio(const shared_ptr<libflow::FullGrid> &p_grid,
                                    const shared_ptr<OptimizePortfolioDP> &p_optimize,
                                    const ArrayXi &p_nbMesh,
                                    const function<double(const int &, const ArrayXd &, const ArrayXd &)>  &p_funcFinalValue,
                                    const ArrayXd &p_initialPortfolio,
                                    const string   &p_fileToDump
#ifdef USE_MPI
                                    , const boost::mpi::communicator &p_world
#endif
                                   )
{
    // initialize simulation
    p_optimize->initializeSimulation();
    // store regressor
    shared_ptr<libflow::LocalConstRegression> regressorPrevious;

    // store final regressed values in  object valuesStored
    shared_ptr< vector< ArrayXXd > > valuesStored = make_shared< vector<ArrayXXd> >(p_optimize->getNbRegime());
    {
        vector< shared_ptr< ArrayXXd > >  valuesPrevious = libflow::FinalStepDP(p_grid, p_optimize->getNbRegime())(p_funcFinalValue, *p_optimize->getCurrentSim());
        // regressor operator
        regressorPrevious = make_shared<libflow::LocalConstRegression>(false, *p_optimize->getCurrentSim(), p_nbMesh);
        for (int iReg = 0; iReg < p_optimize->getNbRegime(); ++iReg)
            (*valuesStored)[iReg] = regressorPrevious->getCoordBasisFunctionMultiple(valuesPrevious[iReg]->transpose()).transpose();
    }
    shared_ptr<gs::BinaryFileArchive> ar ;
#ifdef USE_MPI
    if (p_world.rank() == 0)
#endif
        ar = make_shared<gs::BinaryFileArchive>(p_fileToDump.c_str(), "w");
    // name for object in archive
    string nameAr = "OptimizePort";
    // iterate on time steps
    for (int iStep = 0; iStep < p_optimize->getNbStep(); ++iStep)
    {
        // step backward for simulations
        p_optimize->oneStepBackward();
        // create regressor at the given date
        bool bZeroDate = (iStep == p_optimize->getNbStep() - 1);
        shared_ptr<libflow::LocalConstRegression> regressorCur = make_shared<libflow::LocalConstRegression>(bZeroDate, *p_optimize->getCurrentSim(), p_nbMesh);
        // transition object
        libflow::TransitionStepDP transStep(p_grid, p_grid, regressorCur, regressorPrevious, p_optimize
#ifdef USE_MPI
                                          , p_world
#endif
                                         );
        pair< shared_ptr< vector< ArrayXXd> >, shared_ptr< vector< ArrayXXd > > > valuesAndControl = transStep.oneStep(*valuesStored);
        // dump control values
        transStep.dumpValues(ar, nameAr, iStep,  *valuesAndControl.second);
        valuesStored = valuesAndControl.first;
        // shift regressor
        regressorPrevious = regressorCur;
    }
    // interpolate at the initial stock point and initial regime( 0 here)  (take first particle)
    libflow::GridAndRegressedValue finalValue(p_grid, regressorPrevious);
    finalValue.setRegressedValues((*valuesStored)[0]);
    return finalValue.getValue(p_initialPortfolio, p_optimize->getCurrentSim()->col(0));
}
