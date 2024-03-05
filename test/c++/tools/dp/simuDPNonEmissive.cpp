// Copyright (C) 2016 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#ifdef USE_MPI
#include <functional>
#include <memory>
#include <fstream>
#include <Eigen/Dense>
#include <boost/mpi.hpp>
#include "geners/BinaryFileArchive.hh"
#include "libflow/core/grids/SpaceGrid.h"
#include "libflow/core/utils/StateWithStocks.h"
#include "libflow/dp/SimulateStepRegression.h"
#include "libflow/dp/OptimizerDPBase.h"
#include "libflow/dp/SimulatorDPBase.h"

using namespace Eigen ;
using namespace std;


double simuDPNonEmissive(const shared_ptr<libflow::SpaceGrid> &p_grid,
                         const shared_ptr<libflow::OptimizerDPBase > &p_optimize,
                         const function<double(const int &, const ArrayXd &, const ArrayXd &)>  &p_funcFinalValue,
                         const ArrayXd &p_pointStock,
                         const string   &p_fileToDump,
                         const int &p_nbSimTostore,
                         const boost::mpi::communicator &p_world)
{
    // from the optimizer get back the simulation
    shared_ptr< libflow::SimulatorDPBase> simulator = p_optimize->getSimulator();
    int nbStep = simulator->getNbStep();
    vector< libflow::StateWithStocks> states;
    states.reserve(simulator->getNbSimul());
    ArrayXXd  iniStoState = simulator->getParticles().array();
    for (int is = 0; is < simulator->getNbSimul(); ++is)
        states.push_back(libflow::StateWithStocks(0, p_pointStock, iniStoState.col(is)));
    string toDump = p_fileToDump ;
    gs::BinaryFileArchive ar(toDump.c_str(), "r");
    // name for continuation object in archive
    string nameAr = "Continuation";
    // cost function
    ArrayXXd costFunction = ArrayXXd::Zero(p_optimize->getSimuFuncSize(), simulator->getNbSimul());
    // to store control
    std::shared_ptr<ofstream> fileInvest, fileDemand, fileQ, fileY;
    if (p_world.rank() == 0)
    {
        fileInvest = std::make_shared<ofstream>("InvestDP");
        fileDemand = std::make_shared<ofstream>("DemandDP");
        fileQ = std::make_shared<ofstream>("ProdDP");
        fileY = std::make_shared<ofstream>("YDP");
    }
    for (int istep = 0; istep < nbStep; ++istep)
    {
        if (p_world.rank() == 0)
            cout << "Step simu " << istep << endl ;
        libflow::SimulateStepRegression(ar, nbStep - 1 - istep, nameAr, p_grid, p_optimize, p_world).oneStep(states, costFunction);
        // new stochastic state
        ArrayXXd particules =  simulator->stepForwardAndGetParticles();
        for (int is = 0; is < simulator->getNbSimul(); ++is)
            states[is].setStochasticRealization(particules.col(is));
        if (p_world.rank() == 0)
        {
            *fileInvest << istep + 1 << " "  ;
            *fileQ << istep + 1 << " ";
            *fileDemand << istep + 1 << " " ;
            *fileY << istep + 1 << " " ;
            for (int is = 0; is < p_nbSimTostore; ++is)
            {
                ArrayXd stochas = states[is].getStochasticRealization();
                ArrayXd stock = states[is].getPtStock();
                *fileInvest << stock(1) << " " ;
                *fileQ << stock(0) << " ";
                *fileDemand << stochas(0) << " ";
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
    // final : accept to exercise if not already done entirely (here suppose one function to follow)
    for (int is = 0; is < simulator->getNbSimul(); ++is)
        costFunction(0, is) += p_funcFinalValue(states[is].getRegime(), states[is].getPtStock(), states[is].getStochasticRealization()) * simulator->getActu();

    return costFunction.row(0).mean();
}
#endif
