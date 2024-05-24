
#ifdef USE_MPI
#include <fstream>
#include <memory>
#include <functional>
#include <boost/lexical_cast.hpp>
#include <boost/mpi.hpp>
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "reflow/core/grids/FullGrid.h"
#include "reflow/tree/Tree.h"
#include "reflow/dp/FinalStepDPCutDist.h"
#include "reflow/dp/TransitionStepTreeDPCutDist.h"
#include "reflow/core/parallelism/reconstructProc0Mpi.h"
#include "reflow/dp/OptimizerDPCutTreeBase.h"
#include "reflow/dp/SimulatorDPBaseTree.h"


using namespace std;
using namespace Eigen;

double  DynamicProgrammingByTreeCutDist(const shared_ptr<reflow::FullGrid> &p_grid,
                                        const shared_ptr<reflow::OptimizerDPCutTreeBase > &p_optimize,
                                        const function< ArrayXd(const int &, const ArrayXd &, const ArrayXd &)>   &p_funcFinalValue,
                                        const ArrayXd &p_pointStock,
                                        const int &p_initialRegime,
                                        const string   &p_fileToDump,
                                        const bool &p_bOneFile,
                                        const boost::mpi::communicator &p_world)
{
    // from the optimizer get back the simulator
    shared_ptr< reflow::SimulatorDPBaseTree> simulator = p_optimize->getSimulator();
    // final values
    vector< shared_ptr< ArrayXXd > >  valueCutsNext = reflow::FinalStepDPCutDist(p_grid, p_optimize->getNbRegime(), p_optimize->getDimensionToSplit(), p_world)(p_funcFinalValue, simulator->getNodes());
    // dump
    string toDump = p_fileToDump ;
    // test if one file generated
    if (!p_bOneFile)
        toDump +=  "_" + boost::lexical_cast<string>(p_world.rank());
    shared_ptr<gs::BinaryFileArchive> ar;
    if ((!p_bOneFile) || (p_world.rank() == 0))
        ar = make_shared<gs::BinaryFileArchive>(toDump.c_str(), "w");
    // name for object in archive
    string nameAr = "ContinuationTree";
    for (int iStep = 0; iStep < simulator->getNbStep(); ++iStep)
    {
        simulator->stepBackward();
        // probabilities
        std::vector<double>  proba = simulator->getProba();
        // get connection between nodes
        std::vector< std::vector<std::array<int, 2>  > >  connected = simulator->getConnected();
        // conditional expectation operator
        shared_ptr<reflow::Tree> tree = std::make_shared<reflow::Tree>(proba, connected);
        // transition object
        reflow::TransitionStepTreeDPCutDist transStep(p_grid, p_grid, p_optimize, p_world);
        vector< shared_ptr< ArrayXXd > > valueCuts  = transStep.oneStep(valueCutsNext, tree);
        transStep.dumpContinuationCutsValues(ar, nameAr, iStep, valueCutsNext, tree, p_bOneFile);
        valueCutsNext = valueCuts;
    }
    // reconstruct a small grid for interpolation
    ArrayXd  valSim = reflow::reconstructProc0Mpi(p_pointStock, p_grid, valueCutsNext[p_initialRegime], p_optimize->getDimensionToSplit(), p_world);
    return ((p_world.rank() == 0) ? valSim(0) : 0.);

}
#endif
