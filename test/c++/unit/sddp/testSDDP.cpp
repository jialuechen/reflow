
#ifndef USE_MPI
#define BOOST_TEST_MODULE testSDDP
#endif
#define BOOST_TEST_DYN_LINK
#include <tuple>
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <boost/test/unit_test.hpp>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "geners/Record.hh"
#include "geners/Reference.hh"
#include "libflow/sddp/LocalLinearRegressionForSDDPGeners.h"
#include "libflow/sddp/LocalConstRegressionForSDDPGeners.h"
#include "libflow/sddp/SDDPACut.h"
#include "libflow/sddp/SDDPACutGeners.h"
#include "libflow/sddp/SDDPVisitedStates.h"
#include "libflow/sddp/SDDPVisitedStatesGeners.h"
#include "libflow/sddp/SDDPLocalCut.h"

using namespace std;
using namespace Eigen;
using namespace gs;
using namespace libflow;

#if defined   __linux
#include <fenv.h>
#define enable_abort_on_floating_point_exception() feenableexcept(FE_DIVBYZERO | FE_INVALID)
#endif

double accuracyEqual = 1e-10;


/// For Clang < 3.7 (and above ?) to be compatible GCC 5.1 and above
namespace boost
{
namespace unit_test
{
namespace ut_detail
{
std::string normalize_test_case_name(const_string name)
{
    return (name[0] == '&' ? std::string(name.begin() + 1, name.size() - 1) : std::string(name.begin(), name.size()));
}
}
}
}


template< class LocalRegressionForSDDP>
void testNoCond()
{
#ifdef USE_MPI
    boost::mpi::communicator world;
    int nbTask = world.size();
    int iTask = world.rank();
#else
    int nbTask = 1;
    int iTask = 0;
#endif
    // generator
    boost::mt19937 generator;
    boost::normal_distribution<double> alea_n;
    boost::variate_generator<boost::mt19937 &, boost::normal_distribution<double> > normal_random(generator, alea_n) ;
    // no conditional regression but create regressor
    //************************************************
    ArrayXXd particles = ArrayXXd();
    ArrayXi mesh ;
    shared_ptr< LocalRegressionForSDDP > regressor = make_shared<LocalRegressionForSDDP>(false, particles, mesh);
    // for each particle assess the mesh it belongs to (useless here but should work without particle)
    regressor->evaluateSimulBelongingToCell();
    // Visited state object
    //************************
    SDDPVisitedStates states(regressor->getNbMeshTotal());
    // create state
    int nState = 10;
    for (int i = 0; i < nState; ++i)
    {
        shared_ptr<ArrayXd > tab = make_shared< ArrayXd>(1);
        (*tab)(0) = -1 + 2.*i / (nState * nbTask) + iTask * (2. / nbTask);
        // fictitious particle
        ArrayXd aParticle  ;
        states.addVisitedState(tab, aParticle, *regressor);
    }
    // root is collecting all states
#ifdef USE_MPI
    states.sendToRoot(world);
#endif
    // print an archive
    if (iTask == 0)
    {
        gs::BinaryFileArchive arState("ArchiveStates", "w");
        arState <<  Record(states, "States", "top") ;
    }
    // read the archive
    SDDPVisitedStates statesRecovery(regressor->getNbMeshTotal());
    if (iTask == 0)
    {
        gs::BinaryFileArchive arState("ArchiveStates", "r");
        Reference<SDDPVisitedStates >(arState, "States", "top").restore(0, &statesRecovery);
        BOOST_CHECK_EQUAL(statesRecovery.getStateSize(), states.getStateSize());
    }


#ifdef USE_MPI
    // all states are send to all processor
    statesRecovery.sendFromRoot(world);
    // To go on in mpi aux processors should have all states
    states.sendFromRoot(world);
#endif

    // To create conditional cuts
    //*************************
    int idate = 10; // date identifier
    int nbSample = 100;  // number of samples for expectation (AR models for example)
    SDDPLocalCut sddpCut(idate, nbSample, regressor);
    // add states, create conditional cuts
    ArrayXXd cutPerSim(2, states.getStateSize()*nbSample);
    for (int i = 0; i < states.getStateSize(); ++i)
    {
        shared_ptr<Eigen::ArrayXd > tab = states.getAState(i);
        for (int is = 0; is < nbSample; ++is)
        {
            cutPerSim(0, is + i * nbSample) = (*tab)(0) * (*tab)(0) + normal_random();
            cutPerSim(1, is + i * nbSample) = 2 * (*tab)(0) + normal_random();
        }
    }
    // create vector of LP (one for each sample)
    vector< tuple< shared_ptr<Eigen::ArrayXd>, int, int >  >  vecState = sddpCut.createVectorStatesParticle(states);
    // spread between processors
    int nbLPTotal = vecState.size() * nbSample;
    int nsimPProc = (int)(nbLPTotal / nbTask);
    int nRest = nbLPTotal % nbTask;
    int iLPFirst = iTask * nsimPProc + (iTask < nRest ? iTask : nRest);
    int iLPLast  = iLPFirst + nsimPProc + (iTask < nRest ? 1 : 0);
    ArrayXXd cutPerSimPerProc = cutPerSim.block(0, iTask * (iLPLast - iLPFirst), 2, iLPLast - iLPFirst);
    // archive to store cuts
    shared_ptr<gs::BinaryFileArchive> arCut;
    if (iTask == 0)
        arCut = make_shared<BinaryFileArchive>("archiveSDDPCut", "w+"); // write and read

    // conditional expectation and store
    sddpCut.createAndStoreCuts(cutPerSimPerProc, states, vecState, arCut
#ifdef USE_MPI
                               , world
#endif
                              );
    // get back cuts
    const vector< shared_ptr< SDDPACut > >   &cutRef = sddpCut.getCutsForAMesh(0);
    for (int i = 0; i < nState; ++i)
    {
        BOOST_CHECK_CLOSE(cutPerSim.block(0, i * nbSample, 1, nbSample).mean(), (*cutRef[i]->getCut())(0, 0), accuracyEqual);
        BOOST_CHECK_CLOSE(cutPerSim.block(1, i * nbSample, 1, nbSample).mean(), (*cutRef[i]->getCut())(1, 0), accuracyEqual);
    }
    // now create an object to load cuts
    SDDPLocalCut sddpCutRecover(idate, nbSample, regressor);
    // cut file read one time
    sddpCutRecover.loadCuts(arCut
#ifdef USE_MPI
                            , world
#endif
                           );
    const vector< shared_ptr< SDDPACut > >   &cutRefRecover = sddpCutRecover.getCutsForAMesh(0);
    for (size_t i = 0; i < cutRefRecover.size(); ++i)
    {
        BOOST_CHECK_CLOSE((*cutRef[i]->getCut())(0, 0), (*cutRefRecover[i]->getCut())(0, 0), accuracyEqual);
        BOOST_CHECK_CLOSE((*cutRef[i]->getCut())(1, 0), (*cutRefRecover[i]->getCut())(1, 0), accuracyEqual);
    }
    // Visited state object added
    //************************
    SDDPVisitedStates statesAdded(regressor->getNbMeshTotal());
    // create state
    for (int i = 0; i < nState; ++i)
    {
        shared_ptr<ArrayXd > tab = make_shared<ArrayXd>(1);
        (*tab)(0) = -1 + 2.*i / (nState * nbTask) + iTask * (2. / nbTask) + 1. / (nState * nbTask);
        // fictitious particle
        ArrayXd aParticle ;
        statesAdded.addVisitedState(tab, aParticle, *regressor);
    }
    // To create conditional cuts added
    //*********************************
    // add states, create conditional cuts
    ArrayXXd cutPerSimAdded(2, statesAdded.getStateSize()*nbSample);
    for (int i = 0; i < statesAdded.getStateSize(); ++i)
    {
        shared_ptr<Eigen::ArrayXd > tab = statesAdded.getAState(i);
        for (int is = 0; is < nbSample; ++is)
        {
            cutPerSimAdded(0, is + i * nbSample) = (*tab)(0) * (*tab)(0) + normal_random();
            cutPerSimAdded(1, is + i * nbSample) = 2 * (*tab)(0) + normal_random();
        }
    }
    // create vector of LP (one for each sample)
    vector< tuple< shared_ptr<Eigen::ArrayXd>, int, int >  >  vecStateAdded = sddpCutRecover.createVectorStatesParticle(statesAdded);
    // spread between processors
    int nbLPTotalAdded = vecStateAdded.size() * nbSample;
    int nsimPProcAdded = (int)(nbLPTotalAdded / nbTask);
    int nRestAdded = nbLPTotalAdded % nbTask;
    int iLPFirstAdded = iTask * nsimPProc + (iTask < nRestAdded ? iTask : nRestAdded);
    int iLPLastAdded  = iLPFirstAdded + nsimPProcAdded + (iTask < nRestAdded ? 1 : 0);
    ArrayXXd cutPerSimPerProcAdded = cutPerSimAdded.block(0, iTask * (iLPLastAdded - iLPFirstAdded), 2, iLPLastAdded - iLPFirstAdded);
    // conditional expectation and store
    sddpCutRecover.createAndStoreCuts(cutPerSimPerProcAdded, statesAdded, vecStateAdded, arCut
#ifdef USE_MPI
                                      , world
#endif
                                     );
}

BOOST_AUTO_TEST_CASE(testSDDPNoConditional)
{
    // constant  per mesh
    testNoCond<LocalConstRegressionForSDDP>();
    // linear per mesh
    testNoCond<LocalLinearRegressionForSDDP>();

}

template< class LocalRegressionForSDDP>
void testConditional()
{
#ifdef USE_MPI
    boost::mpi::communicator world;
    int nbTask = world.size();
    int iTask = world.rank();
#else
    int nbTask = 1;
    int iTask = 0;
#endif
    // generator
    boost::mt19937 generator;
    boost::normal_distribution<double> alea_n;
    boost::variate_generator<boost::mt19937 &, boost::normal_distribution<double> > normal_random(generator, alea_n) ;
    // no conditional regression but create regressor
    //************************************************
    int nbSimul = 10;
    ArrayXXd particles(1, nbSimul);
    for (int i = 0; i < nbSimul; ++i)
        particles(0, i) = i * (1. / nbSimul);
    int nbMesh = 2;
    ArrayXi mesh(1) ;
    mesh(0) = nbMesh;
    shared_ptr< LocalRegressionForSDDP > regressor = make_shared<LocalRegressionForSDDP>(false, particles, mesh);
    // for each particle assess the mesh it belongs to (useless here but should work without particle)
    regressor->evaluateSimulBelongingToCell();
    // Visited state object
    //************************
    SDDPVisitedStates states(regressor->getNbMeshTotal());
    // create state
    int nState = 10;
    for (int i = 0; i < nState; ++i)
    {
        shared_ptr<ArrayXd > tab = make_shared<ArrayXd>(1);
        (*tab)(0) = -1 + 2.*i / (nState * nbTask) + iTask * (2. / nbTask);
        // fictitious particle
        ArrayXd aParticle ;
        states.addVisitedState(tab, aParticle, *regressor);
    }
#ifdef USE_MPI
    // root is collecting all states
    states.sendToRoot(world);
    // To go on in mpi aux processors should have all states
    states.sendFromRoot(world);
#endif
    // To create conditional cuts
    //*************************
    int idate = 10; // date identifier
    int nbSample = 10;  // number of samples for expectation (AR models for example)
    SDDPLocalCut sddpCut(idate, nbSample, regressor);
    // add states, create conditional cuts
    ArrayXXd cutPerSim(2, states.getStateSize()*nbSample * nbSimul);
    for (int i = 0; i < states.getStateSize(); ++i)
    {
        shared_ptr<ArrayXd > tab = make_shared<ArrayXd>(1);
        (*tab)(0) = -1 + 2.*i / nState;
        for (int is = 0; is < particles.cols(); ++is)
            for (int isam = 0 ; isam < nbSample; ++isam)
            {
                cutPerSim(0, (i * nbSimul + is)*nbSample + isam) = (*tab)(0) * (*tab)(0) * (particles(0, is) + normal_random());
                cutPerSim(1, (i * nbSimul + is)*nbSample + isam) = 2 * (*tab)(0) * (particles(0, is)   + normal_random());
            }
    }
    // create vector of LP (one for each sample)
    vector< tuple< shared_ptr<Eigen::ArrayXd>, int, int >  >  vecState = sddpCut.createVectorStatesParticle(states);
    // spread between processors
    int nbLPTotal = vecState.size() * nbSample;
    int nsimPProc = (int)(nbLPTotal / nbTask);
    int nRest = nbLPTotal % nbTask;
    int iLPFirst = iTask * nsimPProc + (iTask < nRest ? iTask : nRest);
    int iLPLast  = iLPFirst + nsimPProc + (iTask < nRest ? 1 : 0);
    ArrayXXd cutPerSimPerProc = cutPerSim.block(0, iTask * (iLPLast - iLPFirst), 2, iLPLast - iLPFirst);
    // archive to store cuts
    shared_ptr<gs::BinaryFileArchive> arCut;
    if (iTask == 0)
        arCut = make_shared<BinaryFileArchive>("archiveSDDPCut", "w+"); // write and read

    // conditional expectation and store
    sddpCut.createAndStoreCuts(cutPerSimPerProc, states, vecState, arCut
#ifdef USE_MPI
                               , world
#endif
                              );
}

BOOST_AUTO_TEST_CASE(testSDDPConditional)
{
    // constant  per mesh
    testConditional<LocalConstRegressionForSDDP>();

    // linear per mesh
    testConditional<LocalLinearRegressionForSDDP>();

}



#ifdef USE_MPI

// (empty) Initialization function. Can't use testing tools here.
bool init_function()
{
    return true;
}

int main(int argc, char *argv[])
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    boost::mpi::environment env(argc, argv);
    return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
#endif
