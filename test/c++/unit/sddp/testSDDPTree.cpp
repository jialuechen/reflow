
#define BOOST_TEST_MODULE testSDDPTree
#define BOOST_TEST_DYN_LINK
#define _USE_MATH_DEFINES
#include <math.h>
#include <boost/test/unit_test.hpp>
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "geners/Record.hh"
#include "geners/Reference.hh"
#include "geners/vectorIO.hh"
#include "geners/arrayIO.hh"
#include "reflow/core/utils/constant.h"
#include "reflow/core/utils/eigenGeners.h"
#include "reflow/core/grids/OneDimRegularSpaceGrid.h"
#include "reflow/core/grids/OneDimData.h"
#include "reflow/sddp/SimulatorSDDPBaseTree.h"
#include "test/c++/tools/simulators/TrinomialTreeOUSimulator.h"
#include "test/c++/tools/simulators/MeanRevertingSimulatorTree.h"

using namespace std;
using namespace Eigen;
using namespace gs;
using namespace reflow;

#if defined   __linux
#include <fenv.h>
#define enable_abort_on_floating_point_exception() feenableexcept(FE_DIVBYZERO | FE_INVALID)
#endif

double accuracyEqual = 1e-10;
double accuracyNearlyEqual = 2e-2;
double accuracyNNearlyEqual = 3e-1;


// class created to test
class SimulatorSDDPBaseTreeTest: public SimulatorSDDPBaseTree
{
public :

    SimulatorSDDPBaseTreeTest(shared_ptr<gs::BinaryFileArchive>   &p_binForTree): SimulatorSDDPBaseTree(p_binForTree) {}

    int getNodeAssociatedToSim(const int &) const
    {
        return 0;
    }
    void  stepForward() {}
    void  stepBackward() {}
    Eigen::ArrayXXd getNodesValues() const
    {
        return Eigen::ArrayXXd();
    }
    int getDimension() const
    {
        return 1;
    }
    int getNbStep() const
    {
        return 0;
    }
    Eigen::ArrayXd  getValueAssociatedToNode(const int &) const
    {
        return Eigen::ArrayXd();
    }

};


/// Testtree creation ad lecture
BOOST_AUTO_TEST_CASE(testTrinomialTreeCreation)
{
    // mean reverting
    double mr = 0.3;
    // volatility
    double sig = 0.6;

    int nStep = 20;
    double T = 1;
    double dt = T / nStep;

    // simulation tree
    ArrayXd dates =  ArrayXd::LinSpaced(nStep + 1, 0., T);

    // simulaton dates
    TrinomialTreeOUSimulator tree(mr, sig, dates);

    // test probabiltu betwen two time step
    {
        int istepBeg = 0;
        int istepEnd = 4 ;
        ArrayXXd proba = tree.getProbability(istepBeg, istepEnd);
        for (int id = 0; id < proba.rows(); ++id)
        {
            double prob = proba.row(id).sum();
            BOOST_CHECK_CLOSE(prob, 1., tiny);
        }
    }
    {
        int istepBeg = 4;
        int istepEnd = 10 ;
        ArrayXXd proba = tree.getProbability(istepBeg, istepEnd);
        for (int id = 0; id < proba.rows(); ++id)
        {
            double prob = proba.row(id).sum();
            BOOST_CHECK_CLOSE(prob, 1., tiny);
        }
    }
    for (int i = 0; i < nStep - 1; ++i)
    {
        // points at date i+1
        ArrayXd pointNext = tree.getPoints(i + 1).row(0).transpose();
        ArrayXd pointNext2 = pointNext * pointNext;
        // poiny at date i
        ArrayXd pointCur = tree.getPoints(i).row(0).transpose();
        for (int j = 0; j < pointCur.size(); ++j)
        {
            double valExpCond = tree.calculateStepCondExpectation(i, j, pointNext);
            if (fabs(pointCur(j)) > tiny)
            {
                BOOST_CHECK_CLOSE(valExpCond - pointCur(j), (exp(-mr * dt) - 1)*pointCur(j), tiny);
            }
        }
    }
}

/// Test lecture of archive with  geners
BOOST_AUTO_TEST_CASE(testSimulatorcreation)
{
    // mean reverting
    double mr = 0.3;
    // volatility
    double sig = 0.6;

    // simulation dates
    ArrayXd dates =  ArrayXd::LinSpaced(20, 0., 1.);

    // simulaton dates
    TrinomialTreeOUSimulator tree(mr, sig, dates);

    // sub array for dates
    ArrayXi indexT(5);
    indexT << 0, 4, 10, 15, 19 ;

    ArrayXd ddates(indexT.size());
    for (int i = 0; i < indexT.size(); ++i)
        ddates(i) = dates(indexT(i));

    // create arxiv
    {
        BinaryFileArchive binArxiv("TestCreation", "w");
        binArxiv << Record(ddates, "dates", "");
        for (int i = 0 ;  i < indexT.size(); ++i)
        {
            ArrayXXd points = tree.getPoints(indexT(i));
            binArxiv << Record(points, "points", "");
        }
        for (int i = 0 ;  i < indexT.size() - 1; ++i)
        {
            ArrayXXd  proba = tree.getProbability(indexT(i), indexT(i + 1));
            pair< vector< vector< array<int, 2> > >, vector< double >  >   connectAndProba = tree.calConnected(proba);
            binArxiv << Record(connectAndProba.second, "proba", "");
            binArxiv << Record(connectAndProba.first, "connection", "");
        }
    }
    shared_ptr<BinaryFileArchive> binArxiv = make_shared<BinaryFileArchive>("TestCreation", "r");

    // create the tree for SDDP loading poinsta nd probability transitions
    SimulatorSDDPBaseTreeTest sddpTree(binArxiv);

    for (int i = 1; i < ddates.size(); ++i)
    {
        sddpTree.updateDateIndex(i);
        // get back probablity connections
        vector< vector< array<int, 2 > > >  connections = sddpTree.getConnected();
        // probabilities
        vector< double > proba = sddpTree.getProba();
        for (size_t ip  = 0; ip < connections.size(); ++ip)
        {
            double sProba = 0;
            // sum on all arrival from node  i
            for (size_t j = 0; j < connections[ip].size() ; ++j)
                sProba += proba[ connections[ip][j][1]] ;
            BOOST_CHECK_CLOSE(sProba, 1., tiny);
        }
    }

}


// test backward mean reverting simulator
BOOST_AUTO_TEST_CASE(testMeanRevertingSDDPBackward)
{
    size_t nbStep = 30;
    double step = 1. / nbStep;
    int nStepPerStepMax = 4;

    for (size_t iStep = 1; iStep < nbStep ; ++iStep)
    {
        // mean reverting parameters
        double sig =  0.94;
        double  mr = 0.29;

        // define a a time grid
        shared_ptr<OneDimRegularSpaceGrid> timeGrid(new OneDimRegularSpaceGrid(0., step, iStep));
        // future values
        shared_ptr<vector< double > > futValues(new vector<double>(iStep + 1));
        // periodicity factor
        int iPeriod = 52;
        for (size_t i = 0; i < iStep + 1; ++i)
            (*futValues)[i] = 50. + 20 * sin((M_PI * i * iPeriod) / iStep);

        // define the future curve
        shared_ptr<OneDimData<OneDimRegularSpaceGrid, double> > futureGrid(new OneDimData< OneDimRegularSpaceGrid, double> (timeGrid, futValues));

        for (int nbStepTreePerStep = 1; nbStepTreePerStep < nStepPerStepMax ; ++nbStepTreePerStep)
        {

            // simulation dates
            ArrayXd ddates =  ArrayXd::LinSpaced(iStep * nbStepTreePerStep + 1, 0., iStep * step);

            // simulaton dates
            TrinomialTreeOUSimulator tree(mr, sig, ddates);


            // sub array for dates
            ArrayXi indexT(iStep + 1);
            for (size_t i = 0; i < iStep; ++i)
                indexT(i) = i * nbStepTreePerStep;
            indexT(iStep) = ddates.size() - 1;

            // create arxiv
            string nameTree = "Tree";
            tree.dump(nameTree, indexT);

            shared_ptr<gs::BinaryFileArchive> binArxiv = make_shared<gs::BinaryFileArchive>(nameTree.c_str(), "r");

            // create mean reverting simulator (backward)
            MeanRevertingSimulatorTree<OneDimData<OneDimRegularSpaceGrid, double> >  simulator(binArxiv, futureGrid, sig, mr);

            // get spot at final date
            ArrayXd spot = simulator.getSpotAllNode();

            double exSpot = tree.calculateExpectation(ddates.size() - 1, spot);
            BOOST_CHECK_CLOSE(exSpot, simulator.getAnalyticalESpot(), accuracyNearlyEqual);

            double exSpot2 = tree.calculateExpectation(ddates.size() - 1, spot * spot);
            BOOST_CHECK_CLOSE(exSpot2, simulator.getAnalyticalESpot2(), accuracyNearlyEqual);

        }
    }
}

// test forward mean reverting simulator
BOOST_AUTO_TEST_CASE(testMeanRevertingSDDPForward)
{
    size_t nbStep = 60;
    double step = 1. / nbStep;
    int nbStepTreePerStep = 8;
    int nbSimul = 100000;

    // mean reverting parameters
    double sig =  0.94;
    double  mr = 0.29;

    // define a a time grid
    shared_ptr<OneDimRegularSpaceGrid> timeGrid(new OneDimRegularSpaceGrid(0., step, nbStep));
    // future values
    shared_ptr<vector< double > > futValues(new vector<double>(nbStep + 1));
    // periodicity factor
    int iPeriod = 52;
    for (size_t i = 0; i < nbStep + 1; ++i)
        (*futValues)[i] = 50. + 20 * sin((M_PI * i * iPeriod) / nbStep);

    // define the future curve
    shared_ptr<OneDimData<OneDimRegularSpaceGrid, double> > futureGrid(new OneDimData< OneDimRegularSpaceGrid, double> (timeGrid, futValues));

    // simulation dates
    ArrayXd ddates =  ArrayXd::LinSpaced(nbStep * nbStepTreePerStep + 1, 0., nbStep * step);

    // simulaton dates
    TrinomialTreeOUSimulator tree(mr, sig, ddates);


    // sub array for dates
    ArrayXi indexT(nbStep + 1);
    for (size_t i = 0; i < nbStep; ++i)
        indexT(i) = i * nbStepTreePerStep;
    indexT(nbStep) = ddates.size() - 1;

    // create arxiv
    string nameTree = "Tree";
    tree.dump(nameTree, indexT);

    shared_ptr<gs::BinaryFileArchive> binArxiv = make_shared<gs::BinaryFileArchive>(nameTree.c_str(), "r");

    // create mean reverting simulator (forward)
    MeanRevertingSimulatorTree<OneDimData<OneDimRegularSpaceGrid, double> >  simulator(binArxiv, futureGrid, sig, mr, nbSimul);

    for (size_t idate = 0; idate <  nbStep; ++idate)
    {
        simulator.stepForward();
    }

    // final node valeurs
    ArrayXd spot(nbSimul);
    for (int is = 0; is < nbSimul; ++is)
        spot(is) = simulator.fromOneNodeToSpot(simulator.getNodeAssociatedToSim(is));

    double exSpot = spot.mean();
    BOOST_CHECK_CLOSE(exSpot, simulator.getAnalyticalESpot(), accuracyNNearlyEqual);

    double exSpot2 = (spot * spot).mean();
    BOOST_CHECK_CLOSE(exSpot2, simulator.getAnalyticalESpot2(), accuracyNNearlyEqual);

}

