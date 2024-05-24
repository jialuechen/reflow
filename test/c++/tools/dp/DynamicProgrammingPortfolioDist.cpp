
#ifdef USE_MPI
#include <fstream>
#include <boost/mpi.hpp>
#include <memory>
#include <functional>
#include <boost/lexical_cast.hpp>
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "reflow/core/grids/FullGrid.h"
#include "reflow/regression/LocalConstRegression.h"
#include "reflow/regression/GridAndRegressedValue.h"
#include "reflow/dp/FinalStepDPDist.h"
#include "reflow/dp/TransitionStepDPDist.h"
#include "reflow/core/parallelism/reconstructProc0Mpi.h"
#include "test/c++/tools/dp/OptimizePortfolioDP.h"

using namespace std;
using namespace Eigen;

double  DynamicProgrammingPortfolioDist(const shared_ptr<reflow::FullGrid> &p_grid,
                                        const shared_ptr<OptimizePortfolioDP> &p_optimize,
                                        const ArrayXi &p_nbMesh,
                                        const function<double(const int &, const ArrayXd &, const ArrayXd &)>  &p_funcFinalValue,
                                        const ArrayXd &p_initialPortfolio,
                                        const string   &p_fileToDump,
                                        const bool &p_bOneFile,
                                        const boost::mpi::communicator &p_world
                                       )
{
    // initialize simulation
    p_optimize->initializeSimulation();
    // store regressor
    shared_ptr<reflow::LocalConstRegression> regressorPrevious;

    // store final regressed values in  object valuesStored
    shared_ptr< vector< ArrayXXd > > valuesStored = make_shared< vector<ArrayXXd> >(p_optimize->getNbRegime());
    {
        vector< shared_ptr< ArrayXXd > >  valuesPrevious = reflow::FinalStepDPDist(p_grid, p_optimize->getNbRegime(), p_optimize->getDimensionToSplit(), p_world)(p_funcFinalValue, *p_optimize->getCurrentSim());
        // regressor operator
        regressorPrevious = make_shared<reflow::LocalConstRegression>(false, *p_optimize->getCurrentSim(), p_nbMesh);
        for (int iReg = 0; iReg < p_optimize->getNbRegime(); ++iReg)
            (*valuesStored)[iReg] = regressorPrevious->getCoordBasisFunctionMultiple(valuesPrevious[iReg]->transpose()).transpose();
    }
    string toDump = p_fileToDump ;
    // test if one file generated
    if (!p_bOneFile)
        toDump +=  "_" + boost::lexical_cast<string>(p_world.rank());
    shared_ptr<gs::BinaryFileArchive> ar;
    if ((!p_bOneFile) || (p_world.rank() == 0))
        ar = make_shared<gs::BinaryFileArchive>(toDump.c_str(), "w");
    // name for object in archive
    string nameAr = "OptimizePort";
    // iterate on time steps
    for (int iStep = 0; iStep < p_optimize->getNbStep(); ++iStep)
    {
        // step backward for simulations
        p_optimize->oneStepBackward();
        // create regressor at the given date
        bool bZeroDate = (iStep == p_optimize->getNbStep() - 1);
        shared_ptr<reflow::LocalConstRegression> regressorCur = make_shared<reflow::LocalConstRegression>(bZeroDate, *p_optimize->getCurrentSim(), p_nbMesh);
        // transition object
        reflow::TransitionStepDPDist transStep(p_grid, p_grid, regressorCur, regressorPrevious, p_optimize, p_world);
        pair< shared_ptr< vector< ArrayXXd> >, shared_ptr< vector< ArrayXXd > > > valuesAndControl = transStep.oneStep(*valuesStored);
        // dump control values
        transStep.dumpValues(ar, nameAr, iStep,  *valuesAndControl.second, p_bOneFile);
        valuesStored = valuesAndControl.first;
        // shift regressor
        regressorPrevious = regressorCur;
    }
    // interpolate at the initial stock point and initial regime( 0 here)  (take first particle)
    shared_ptr<ArrayXXd> topRows = make_shared<ArrayXXd>((*valuesStored)[0].topRows(1));
    return reflow::reconstructProc0Mpi(p_initialPortfolio, p_grid, topRows, p_optimize->getDimensionToSplit(), p_world).mean();
}
#endif
