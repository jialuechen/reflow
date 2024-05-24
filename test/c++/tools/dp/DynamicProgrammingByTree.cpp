
#include <fstream>
#include <memory>
#include <functional>
#include <array>
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <boost/lexical_cast.hpp>
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "reflow/tree/Tree.h"
#include "reflow/core/grids/FullGrid.h"
#include "reflow/dp/FinalStepDP.h"
#include "reflow/dp/TransitionStepTreeDP.h"
#include "reflow/dp/OptimizerDPTreeBase.h"
#include "reflow/dp/SimulatorDPBaseTree.h"

using namespace std;


double  DynamicProgrammingByTree(const shared_ptr<reflow::FullGrid> &p_grid,
                                 const shared_ptr<reflow::OptimizerDPTreeBase > &p_optimize,
                                 const function<double(const int &, const Eigen::ArrayXd &, const Eigen::ArrayXd &)>  &p_funcFinalValue,
                                 const Eigen::ArrayXd &p_pointStock,
                                 const int &p_initialRegime,
                                 const string   &p_fileToDump
#ifdef USE_MPI
                                 , const boost::mpi::communicator &p_world
#endif
                                )
{
    // from the optimizer get back the simulator
    shared_ptr< reflow::SimulatorDPBaseTree> simulator = p_optimize->getSimulator();
    // final values
    vector< shared_ptr< Eigen::ArrayXXd > >  valuesNext = reflow::FinalStepDP(p_grid, p_optimize->getNbRegime())(p_funcFinalValue, simulator->getNodes());
    shared_ptr<gs::BinaryFileArchive> ar = make_shared<gs::BinaryFileArchive>(p_fileToDump.c_str(), "w");
    // name for object in archive
    string nameAr = "Continuation";
    // iterate on time steps
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
        reflow::TransitionStepTreeDP transStep(p_grid, p_grid, p_optimize
#ifdef USE_MPI
                                              , p_world
#endif

                                             );
        pair< vector< shared_ptr< Eigen::ArrayXXd > >, vector< shared_ptr< Eigen::ArrayXXd > > > valuesAndControl = transStep.oneStep(valuesNext, tree);
        // dump continuation values
        transStep.dumpContinuationValues(ar, nameAr, iStep, valuesNext, valuesAndControl.second, tree);
        valuesNext = valuesAndControl.first;

    }
    // interpolate at the initial stock point and initial regime
    return (p_grid->createInterpolator(p_pointStock)->applyVec(*valuesNext[p_initialRegime])).mean();
}
