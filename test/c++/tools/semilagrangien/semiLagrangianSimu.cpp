
#include <memory>
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <boost/random.hpp>
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "libflow/semilagrangien/OptimizerSLBase.h"
#include "libflow/semilagrangien/SimulateStepSemilagrang.h"

using namespace std;

double semiLagrangianSimu(const shared_ptr<libflow::SpaceGrid> &p_grid,
                          const shared_ptr<libflow::OptimizerSLBase > &p_optimize,
                          const function<double(const int &, const Eigen::ArrayXd &)>   &p_funcFinalValue,
                          const int &p_nbStep,
                          const Eigen::ArrayXd &p_stateInit,
                          const int &p_initialRegime,
                          const int &p_nbSimul,
                          const string   &p_fileToDump
#ifdef USE_MPI
                          , const boost::mpi::communicator &p_world
#endif
                         )
{
    // store states in a regime
    Eigen::ArrayXXd states(p_stateInit.size(), p_nbSimul);
    for (int is = 0; is < p_nbSimul; ++is)
        states.col(is) = p_stateInit;
    // sore the regime number
    Eigen::ArrayXi regime = Eigen::ArrayXi::Constant(p_nbSimul, p_initialRegime);
    gs::BinaryFileArchive ar(p_fileToDump.c_str(), "r");
    // name for continuation object in archive
    string nameAr = "Continuation";
    // cost function
    Eigen::ArrayXXd costFunction =  Eigen::ArrayXXd::Zero(p_optimize->getSimuFuncSize(), p_nbSimul);
    // random generator and Gaussian variables
    boost::mt19937 generator;
    boost::normal_distribution<double> normalDistrib;
    boost::variate_generator<boost::mt19937 &, boost::normal_distribution<double> > normalRand(generator, normalDistrib);
    Eigen::ArrayXXd gaussian(p_optimize->getBrownianNumber(), p_nbSimul);
    // iterate on time steps
    for (int istep = 0; istep < p_nbStep; ++istep)
    {
        for (int is = 0; is < gaussian.cols(); ++is)
            for (int id  = 0; id < gaussian.rows(); ++id)
                gaussian(id, is) = normalRand();
        libflow::SimulateStepSemilagrang(ar, p_nbStep - 1 - istep, nameAr, p_grid, p_optimize
#ifdef USE_MPI
                                       , p_world
#endif
                                      ).oneStep(gaussian, states, regime, costFunction);
    }
    // final cost to add (here suppose regime switching)
    for (int is = 0; is < p_nbSimul; ++is)
        costFunction(0, is) += p_funcFinalValue(regime(is), states.col(is));
    // average gain/cost
    return costFunction.mean();
}
