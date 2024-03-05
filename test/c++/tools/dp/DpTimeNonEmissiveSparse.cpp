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
#include "libflow/core/grids/SparseSpaceGrid.h"
#include "libflow/core/parallelism/ParallelComputeGridSplitting.h"
#include "libflow/regression/BaseRegression.h"
#include "libflow/dp/FinalStepDP.h"
#include "libflow/dp/TransitionStepRegressionDPSparse.h"
#include "libflow/dp/OptimizerDPBase.h"
#include "libflow/dp/SimulatorDPBase.h"

using namespace std;
using namespace Eigen;

void  DpTimeNonEmissiveSparse(const shared_ptr<libflow::SparseSpaceGrid> &p_grid,
                              const shared_ptr<libflow::OptimizerDPBase > &p_optimize,
                              const shared_ptr<libflow::BaseRegression> &p_regressor,
                              const std::function<double(const int &, const ArrayXd &, const ArrayXd &)>  &p_funcFinalValue,
                              const std::string   &p_fileToDump,
                              const boost::mpi::communicator &p_world)
{
    // from the optimizer get back the simulation
    shared_ptr< libflow::SimulatorDPBase> simulator = p_optimize->getSimulator();
    // final values
    vector< shared_ptr< ArrayXXd > >  valuesNext = libflow::FinalStepDP(p_grid, p_optimize->getNbRegime())(p_funcFinalValue, simulator->getParticles().array());
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
        ArrayXXd asset = simulator->stepBackwardAndGetParticles();
        // conditional expectation operator
        p_regressor->updateSimulations(((iStep == (simulator->getNbStep() - 1)) ? true : false), asset);
        // transition object
        libflow::TransitionStepRegressionDPSparse transStep(p_grid, p_grid, p_optimize, p_world);
        pair< vector< shared_ptr< ArrayXXd > >, vector< shared_ptr< ArrayXXd > > > valuesPrevAndControl  = transStep.oneStep(valuesNext, p_regressor);
        transStep.dumpContinuationValues(ar, nameAr, iStep, valuesNext, valuesPrevAndControl.second, p_regressor);
        valuesNext = valuesPrevAndControl.first;
        if (iStep == (simulator->getNbStep() - 1))
        {
            // store optimal control
            valuesRecons = valuesNext;
            optimalControl = valuesPrevAndControl.second;
        }
    }
    if (p_world.rank() == 0)
    {
        // DUMP
        fstream fileStream("SortieDPSparse", ios::out);
        ArrayXXd valHierar1(simulator->getNbSimul(), valuesRecons[0]->cols());
        ArrayXXd valHierar2(simulator->getNbSimul(), valuesRecons[1]->cols());
        ArrayXXd optHierar(simulator->getNbSimul(), optimalControl[0]->cols());
        for (int is = 0; is < simulator->getNbSimul(); ++is)
        {
            ArrayXd val1 = valuesRecons[0]->row(is).transpose();
            p_grid->toHierarchize(val1);
            valHierar1.row(is) =  val1.transpose();
            ArrayXd val2 = valuesRecons[1]->row(is).transpose();
            p_grid->toHierarchize(val2);
            valHierar2.row(is) =  val2.transpose();
            ArrayXd opt  = optimalControl[0]->row(is).transpose();
            p_grid->toHierarchize(opt);
            optHierar.row(is) = opt.transpose();
        }
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
                shared_ptr<libflow::Interpolator> gridInterpol =  p_grid->createInterpolator(ptInterp);
                fileStream << ptInterp(0) << " " << ptInterp(1) <<  " " << gridInterpol->applyVec(valHierar1).mean() <<  " " << gridInterpol->applyVec(valHierar2).mean() <<  " " <<   gridInterpol->applyVec(optHierar).mean() << std::endl ;
            }
        fileStream.close();
    }
}
#endif
