// Copyright (C) 2016 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#ifdef USE_MPI
#include <memory>
#include <boost/random.hpp>
#include <boost/mpi.hpp>
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "libflow/core/grids/SparseSpaceGrid.h"
#include "libflow/semilagrangien/OptimizerSLBase.h"
#include "libflow/semilagrangien/SimulateStepSemilagrang.h"

using namespace std;
using namespace Eigen ;

double simuSLNonEmissiveSparse(const shared_ptr<libflow::SparseSpaceGrid> &p_grid,
                               const shared_ptr<libflow::OptimizerSLBase > &p_optimize,
                               const function<double(const int &, const ArrayXd &)>   &p_funcFinalValue,
                               const int &p_nbStep,
                               const ArrayXd &p_stateInit,
                               const int &p_nbSimul,
                               const string   &p_fileToDump,
                               const int &p_nbSimTostore,
                               const boost::mpi::communicator &p_world)
{
    // store states in a regime
    ArrayXXd states(p_stateInit.size(), p_nbSimul);
    for (int is = 0; is < p_nbSimul; ++is)
        states.col(is) = p_stateInit;
    // store the regime number (useless)
    ArrayXi regime = ArrayXi::Constant(p_nbSimul, 0);
    // test if one file generated
    string toDump = p_fileToDump ;
    gs::BinaryFileArchive ar(toDump.c_str(), "r");
    // name for continuation object in archive
    string nameAr = "Continuation";
    // cost function
    ArrayXXd costFunction =  ArrayXXd::Zero(p_optimize->getSimuFuncSize(), p_nbSimul);
    // random generator and Gaussian variables
    boost::mt19937 generator;
    boost::normal_distribution<double> normalDistrib;
    boost::variate_generator<boost::mt19937 &, boost::normal_distribution<double> > normalRand(generator, normalDistrib);
    ArrayXXd gaussian(p_optimize->getBrownianNumber(), p_nbSimul);
    std::shared_ptr<ofstream> fileInvest, fileDemand, fileQ, fileY;
    if (p_world.rank() == 0)
    {
        fileInvest = std::make_shared<ofstream>("InvestSparseSL");
        fileDemand = std::make_shared<ofstream>("DemandSparseSL");
        fileQ = std::make_shared<ofstream>("ProdSparseSL");
        fileY = std::make_shared<ofstream>("YSLSparse");
    }
    // iterate on time steps
    for (int istep = 0; istep < p_nbStep; ++istep)
    {
        for (int is = 0; is < gaussian.cols(); ++is)
            for (int id  = 0; id < gaussian.rows(); ++id)
                gaussian(id, is) = normalRand();

        if (p_world.rank() == 0)
            cout << "Step Simu " << istep << endl ;
        libflow::SimulateStepSemilagrang(ar, p_nbStep - 1 - istep, nameAr, p_grid, p_optimize, p_world).oneStep(gaussian, states, regime, costFunction);
        if (p_world.rank() == 0)
        {
            *fileInvest << istep << " "  ;
            *fileQ << istep << " ";
            *fileDemand << istep << " " ;
            *fileY << istep << " " ;
            for (int is = 0; is < p_nbSimTostore; ++is)
            {
                *fileInvest << states(2, is) << " " ;
                *fileQ << states(1, is) << " ";
                *fileDemand << states(0, is) << " ";
                *fileY << costFunction(1, is) << " " ;
            }
            *fileInvest << endl ;
            *fileQ << endl ;
            *fileDemand << endl ;
            *fileY << endl ;
        }
    }
    if (p_world.rank() == 0)
    {
        fileInvest->close();
        fileQ->close();
        fileDemand->close();
        fileY->close();
    }
    // final cost to add
    for (int is = 0; is < p_nbSimul; ++is)
        costFunction(0, is) += p_funcFinalValue(regime(is), states.col(is));
    // average gain/cost
    return costFunction.row(0).mean();
}
#endif
