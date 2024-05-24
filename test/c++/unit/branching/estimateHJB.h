
#ifndef ESTIMATEHJB_H
#define ESTIMATEHJB_H
#include <functional>
#include <map>
#include <boost/mpi.hpp>
#include <Eigen/Dense>
#include <trng/yarn2.hpp>
#include <trng/normal_dist.hpp>




namespace reflow
{

/// \brief Semi analytical solution
///  \[
///     u(t,x) =- \frac{1}{\theta} \log \big(  \E[ e^{ - \theta g(x + \sqrt{2} W_{T-t})}]\big)
///  \]
/// \param p_nbSim  number of Monte Carlo simulations for this estimation
/// \param p_theta  theta parameter
/// \param p_xInit  initial point
/// \param p_T      matuirty
/// \param p_g      Terminal function
std::pair<double, double > estimateHJB(int p_nbSim, double  p_theta, const Eigen::Matrix<double, 6, 1> &p_xInit, const double &p_T,
                                       std::function< double (const Eigen::Matrix<double, 6, 1>&)>    &p_g)
{
    // mpi
    boost::mpi::communicator world;
    trng::yarn2 gen;
    gen.split(world.size(), world.rank());
    trng::normal_dist<double> normal(0., 1.);
    double retVal = 0;
    double retValSq = 0;
    double TSqrt = sqrt(p_T);
    int nbSimLoc = p_nbSim / world.size();
    p_nbSim = nbSimLoc * world.size();
    for (int i = 0; i < nbSimLoc; ++i)
    {
        Eigen::Matrix<double, 6, 1> x = p_xInit;
        for (int id = 0; id < 6; ++id)
        {
            x(id) += sqrt(2) * TSqrt * normal(gen);
        }
        double sol = exp(- p_theta * p_g(x));
        retVal += sol;
        retValSq += sol * sol;
    }
    double valRet = 0.;
    double valSqRet = 0.;
    boost::mpi::all_reduce(world, retVal, valRet, std::plus<double>());
    boost::mpi::all_reduce(world, retValSq, valSqRet, std::plus<double>());
    valRet /= p_nbSim;
    valSqRet /= p_nbSim   ;
    valSqRet = sqrt((valSqRet - valRet * valRet) / p_nbSim);
    return std::make_pair(-(1. / p_theta) * log(valRet), valSqRet);
}

}
#endif
