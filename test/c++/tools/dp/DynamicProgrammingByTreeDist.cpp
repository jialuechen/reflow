
#ifdef USE_MPI
#include <fstream>
#include <memory>
#include <functional>
#include <boost/lexical_cast.hpp>
#include <boost/mpi.hpp>
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "libflow/core/grids/FullGrid.h"
#include "libflow/tree/Tree.h"
#include "libflow/dp/FinalStepDPDist.h"
#include "libflow/dp/TransitionStepTreeDPDist.h"
#include "libflow/core/parallelism/reconstructProc0Mpi.h"
#include "libflow/dp/OptimizerDPTreeBase.h"
#include "libflow/dp/SimulatorDPBaseTree.h"


using namespace std;

double  DynamicProgrammingByTreeDist(const shared_ptr<libflow::FullGrid> &p_grid,
                                     const shared_ptr<libflow::OptimizerDPTreeBase > &p_optimize,
                                     const function<double(const int &, const Eigen::ArrayXd &, const Eigen::ArrayXd &)>   &p_funcFinalValue,
                                     const Eigen::ArrayXd &p_pointStock,
                                     const int &p_initialRegime,
                                     const string   &p_fileToDump,
                                     const bool &p_bOneFile,
                                     const boost::mpi::communicator &p_world)
{
    // from the optimizer get back the simulator
    shared_ptr< libflow::SimulatorDPBaseTree> simulator = p_optimize->getSimulator();
    // final values
    vector< shared_ptr< Eigen::ArrayXXd > >  valuesNext = libflow::FinalStepDPDist(p_grid, p_optimize->getNbRegime(), p_optimize->getDimensionToSplit(), p_world)(p_funcFinalValue, simulator->getNodes());
    // dump
    string toDump = p_fileToDump ;
    // test if one file generated
    if (!p_bOneFile)
        toDump +=  "_" + boost::lexical_cast<string>(p_world.rank());
    shared_ptr<gs::BinaryFileArchive> ar;
    if ((!p_bOneFile) || (p_world.rank() == 0))
        ar = make_shared<gs::BinaryFileArchive>(toDump.c_str(), "w");
    // name for object in archive
    string nameAr = "Continuation";
    for (int iStep = 0; iStep < simulator->getNbStep(); ++iStep)
    {
        simulator->stepBackward();
        // probabilities
        std::vector<double>  proba = simulator->getProba();
        // get connection between nodes
        std::vector< std::vector<std::array<int, 2>  > >  connected = simulator->getConnected();
        // conditional expectation operator
        shared_ptr<libflow::Tree> tree = std::make_shared<libflow::Tree>(proba, connected);
        // transition object
        libflow::TransitionStepTreeDPDist transStep(p_grid, p_grid, p_optimize, p_world);
        pair< vector< shared_ptr< Eigen::ArrayXXd > >, vector< shared_ptr< Eigen::ArrayXXd > > > valuesAndControl  = transStep.oneStep(valuesNext, tree);
        // dump
        transStep.dumpContinuationValues(ar, nameAr, iStep, valuesNext, valuesAndControl.second,  tree, p_bOneFile);
        valuesNext = valuesAndControl.first;
    }
    // reconstruct a small grid for interpolation
    return libflow::reconstructProc0Mpi(p_pointStock, p_grid, valuesNext[p_initialRegime], p_optimize->getDimensionToSplit(), p_world).mean();

}
#endif
