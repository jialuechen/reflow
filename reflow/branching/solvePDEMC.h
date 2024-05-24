
#ifndef SOLVEPDEMC_H
#define SOLVEPDEMC_H
#include <functional>
#include <boost/mpi.hpp>
#include <Eigen/Dense>
#include <trng/normal_dist.hpp>




/** \file SolvePDEMC.h
 *  \brief Permits to solve High dimensional PDE using methodology
 *         'Nesting Monte Carlo for high-dimensional Non Linear PDEs' by X Warin
 *         Equation
 *         \f[
 *             (-\partial_t u-{\cal L} u)(t,x)   = f(t,x,u(t,x)),
 *         \f]
 *         with
 *         \f[
 *             u_T =g
 *         \f]
 *         and
 *         \f[
 *           {\cal L} u(t,x) := \mu Du(t,x) +  \frac{1}{2} \sigma \sigma^{\top} \!:\! D^2 u(t,x)
 *         \f]
 *         Special case where the non linearity satisfies \f$ f(t,x, u) \f$
 *         Two versions
 *             - Euler scheme
 *             - exact simulation of the PDE simulated with :
 *              \f[
 *                   X_{t+dt} =  A(t,dt)  X_t + B(t,dt) + C(t,dt) g
 *               \f]
 *               where $g$ is a unit centered gaussian vector.
 *             .
 *   \author Xavier Warin
 */


namespace reflow
{

/// \brief PDE resolution for  one step and one simulation with an Euler scheme
///        Suppose that the drift and diffusion are given by functions
/// \param p_initVal           initial values for all points
/// \param p_mu                drift vector function
/// \param p_sigma             diffusion matrix function
/// \param p_f                 non linear function \f$ f(t,x,u) \f$
/// \param p_timeInit          initial time
/// \param p_T                 maturity
/// \param p_lawSwitch           time switch  generator
/// \param p_normal            N(0,1) generator
/// \param p_gen               parallel RNG
/// \param p_g                 Terminal function
/// \param p_stepEuler         Euler step
/// \param p_nbSim             for each nesting level,  give the number of simulations
/// \param p_depth             current nesting depth.
/// \param p_gen               TRNG random number generator
/// \return value at current date, current position
template< int N, class SwitchDistrib, class TRNGGenerator>
inline double  solveOneStepEuler(const Eigen::Matrix<double, N, 1>   &p_initVal,
                                 const std::function< Eigen::Matrix<double, N, 1> (const double &, const Eigen::Matrix<double, N, 1> &) >   &p_mu,
                                 const std::function< Eigen::Matrix<double, N, N> (const double &, const Eigen::Matrix<double, N, 1> &) >   &p_sigma,
                                 const std::function< double (const double &, const Eigen::Matrix<double, N, 1>&, const double &) > &p_f,
                                 const double   &p_timeInit,
                                 const double &p_T,
                                 SwitchDistrib &p_lawSwitch,
                                 trng::normal_dist<double> &p_normal,
                                 TRNGGenerator &p_gen,
                                 const std::function< double (const Eigen::Matrix<double, N, 1>&)> &p_g,
                                 const double &p_stepEuler,
                                 const std::vector<int> &p_nbSim,
                                 int p_depth)
{
    // one step for position
    double dt = std::max(std::min(p_lawSwitch(p_gen), p_T - p_timeInit), 1e-20);
    // number Euler steps
    int nbEulerSteps = std::max(static_cast<int>(dt / p_stepEuler), 1);
    double dtLoc = dt / nbEulerSteps;
    double sqrtdtLoc = sqrt(dtLoc);

    // store SDE solution
    Eigen::Matrix<double, N, 1>  xProc =  p_initVal;

    // nest on dates
    double tSuiv = p_timeInit;
    for (int j = 0; j < nbEulerSteps; ++j)
    {
        tSuiv += dtLoc;
        Eigen::Matrix< double, N, 1> gauss;
        for (int id = 0; id < N; ++id)
            gauss(id) = p_normal(p_gen);
        xProc +=  p_mu(tSuiv, xProc) * dtLoc + p_sigma(tSuiv, xProc) * sqrtdtLoc * gauss;
    }

    // for return
    double  ret ;

    // new date
    double newDate = p_timeInit + dt;

    // test
    if (newDate >=  p_T - 1e-5)
    {
        double unSurCDF = 1. / (1. - p_lawSwitch.cdf(dt));
        double finalG = p_g(xProc);
        ret = finalG * unSurCDF;
    }
    else
    {
        double lawSwitchPdf = p_lawSwitch.pdf(dt) ;

        if (p_nbSim.size() == p_depth)
        {
            // stop the recurson
            ret = p_f(newDate, xProc, p_g(xProc)) / lawSwitchPdf;
        }
        else
        {
            int nbsimul = p_nbSim[p_depth];
            double  yVal = 0.;
            for (int is = 0 ; is < nbsimul; ++is)
            {
                yVal += solveOneStepEuler<N, SwitchDistrib, TRNGGenerator>(xProc, p_mu, p_sigma,  p_f, newDate, p_T, p_lawSwitch, p_normal,
                        p_gen, p_g,  p_stepEuler, p_nbSim, p_depth + 1);
            }
            yVal /= nbsimul;
            double obstacle = p_f(newDate, xProc, yVal) ;
            ret = obstacle / lawSwitchPdf;
        }
    }
    return ret;
}


/// \brief PDE resolution for Euler Scheme
/// \param p_mu                drift vector
/// \param p_sigma             volatility matrix
/// \param p_f                 non linear function \f$ f(t,x,u) \f$
/// \param p_point             initial point  where to calculate the solution
/// \param p_timeInit          initial time
/// \param p_T                 maturity
/// \param p_lawSwitch           time switch  generator
/// \param p_normal            N(0,1) generator
/// \param p_gen               parallel RNG
/// \param p_g                 Terminal function
/// \param p_nbSim             for each nesting level,  give the number of simulations
/// \param p_stepEuler         step Euler
/// \param p_gen               TRNG random number generator
/// \return values std /sqrt(n) for val
template< int N, class SwitchDistrib, class TRNGGenerator >
std::tuple< double,  double  >   solvePDEMCEuler(const std::function< Eigen::Matrix<double, N, 1> (const double &, const Eigen::Matrix<double, N, 1> &) >   &p_mu,
        const std::function< Eigen::Matrix<double, N, N> (const double &, const Eigen::Matrix<double, N, 1> &) >   &p_sigma,
        const std::function< double (const double &, const Eigen::Matrix<double, N, 1>&, const double &) > &p_f,
        const Eigen::Array<double, N, 1>   &p_point,
        const double   &p_timeInit,
        const double   &p_T,
        SwitchDistrib &p_lawSwitch,
        std::function< double (const Eigen::Matrix<double, N, 1>&)>   &p_g,
        const std::vector<int> &p_nbSim,
        const double &p_stepEuler,
        TRNGGenerator &p_gen)
{

    // mpi
    boost::mpi::communicator world;
    //  random generator
    p_gen.split(world.size(), world.rank());
    trng::normal_dist<double> normal(0., 1.);

    // global simulation number
    int nbSimul = p_nbSim[0];
    int nbSimulProc = nbSimul / world.size();

    // store PDE value and std associated for first level
    double  val = 0.;
    double valSq =  0.;
    int depth = 1;

    //#pragma omp parallel for schedule(dynamic,10)
    for (int is = 0; is <   nbSimulProc ; ++is)
    {
        if (world.rank() == 0)
            if (is % 1000 == 0)
                std::cout << " is " << is <<   std::endl ;
        double solLoc = solveOneStepEuler<N, SwitchDistrib, TRNGGenerator>(p_point,  p_mu, p_sigma, p_f, p_timeInit, p_T, p_lawSwitch, normal, p_gen, p_g, p_stepEuler, p_nbSim, depth);
        val += solLoc;
        valSq += solLoc * solLoc;
    }
    double valRet = 0.;
    double valSqRet = 0.;
    boost::mpi::all_reduce(world, val, valRet, std::plus<double>());
    boost::mpi::all_reduce(world, valSq, valSqRet, std::plus<double>());
    int nbSim = nbSimulProc * world.size();
    valRet /= nbSim;
    valSqRet /= nbSim   ;
    valSqRet = sqrt((valSqRet - valRet * valRet) / nbSim);
    return std::make_tuple(valRet, valSqRet);
}




/// \brief PDE resolution for  one step and one simulation with an exact simulation of the process
///        Suppose that the drift and diffusion are given by functions
/// \param p_initVal           initial values for all points
/// \param p_A                 multiplicative factor for \f$ X_{t} \f$
/// \param P_B                 constant term in SDE
/// \param P_C                 volatility
/// \param p_f                 non linear function \f$ f(t,x,u) \f$
/// \param p_timeInit          initial time
/// \param p_T                 maturity
/// \param p_lawSwitch           time switch  generator
/// \param p_normal            N(0,1) generator
/// \param p_gen               parallel RNG
/// \param p_g                 Terminal function
/// \param p_nbSim             for each nesting level,  give the number of simulations
/// \param p_depth             current nesting depth.
/// \return value at current date, current position
template< int N, class SwitchDistrib, class TRNGGenerator>
inline double  solveOneStepExact(const Eigen::Matrix<double, N, 1>   &p_initVal,
                                 const std::function< Eigen::Matrix<double, N, N> (const double &, const  double &) >   &p_A,
                                 const std::function< Eigen::Matrix<double, N, 1> (const double &, const  double &) >   &p_B,
                                 const std::function< Eigen::Matrix<double, N, N> (const double &, const   double &) >     &p_C,
                                 const std::function< double (const double &, const Eigen::Matrix<double, N, 1>&, const double &) > &p_f,
                                 const double   &p_timeInit,
                                 const double &p_T,
                                 SwitchDistrib &p_lawSwitch,
                                 trng::normal_dist<double> &p_normal,
                                 TRNGGenerator &p_gen,
                                 const std::function< double (const Eigen::Matrix<double, N, 1>&)> &p_g,
                                 const std::vector<int> &p_nbSim,
                                 int p_depth)
{
    // one step for position
    double dt = std::max(std::min(p_lawSwitch(p_gen), p_T - p_timeInit), 1e-20);
    double sqrtdt = sqrt(dt);

    // store SDE solution
    Eigen::Matrix< double, N, 1> gauss;
    for (int id = 0; id < N; ++id)
        gauss(id) = p_normal(p_gen);
    Eigen::Matrix<double, N, 1>  xProc =  p_A(p_timeInit, dt) * p_initVal + p_B(p_timeInit, dt) + p_C(p_timeInit, dt) * gauss;

    // for return
    double  ret ;

    // new date
    double newDate = p_timeInit + dt;

    // test
    if (newDate >=  p_T - 1e-5)
    {
        double unSurCDF = 1. / (1. - p_lawSwitch.cdf(dt));
        double finalG = p_g(xProc);
        ret = finalG * unSurCDF;
    }
    else
    {
        double lawSwitchPdf = p_lawSwitch.pdf(dt) ;

        if (p_nbSim.size() == p_depth)
        {
            // stop the recurson
            ret = p_f(newDate, xProc, p_g(xProc)) / lawSwitchPdf;
        }
        else
        {
            int nbsimul = p_nbSim[p_depth];
            double  yVal = 0.;
            for (int is = 0 ; is < nbsimul; ++is)
            {
                yVal += solveOneStepExact<N, SwitchDistrib, TRNGGenerator >(xProc, p_A, p_B, p_C,   p_f, newDate, p_T, p_lawSwitch, p_normal,
                        p_gen, p_g, p_nbSim, p_depth + 1);
            }
            yVal /= nbsimul;
            double obstacle = p_f(newDate, xProc, yVal) ;
            ret = obstacle / lawSwitchPdf;
        }
    }
    return ret;
}


/// \brief PDE resolution with exact simulation
/// \param p_A                 multiplicative factor for \f$ X_{t} \f$
/// \param P_B                 constant term in SDE
/// \param P_C                 volatility
/// \param p_f                 non linear function \f$ f(t,x,u) \f$
/// \param p_point             initial point  where to calculate the solution
/// \param p_timeInit          initial time
/// \param p_T                 maturity
/// \param p_lawSwitch         time switch  generator
/// \param p_normal            N(0,1) generator
/// \param p_gen               parallel RNG
/// \param p_g                 Terminal function
/// \param p_nbSim             for each nesting level,  give the number of simulations
/// \param p_gen               TRNG random number generator
/// \return values std /sqrt(n) for val
template< int N, class SwitchDistrib, class TRNGGenerator  >
std::tuple< double,  double  >   solvePDEMCExact(const std::function< Eigen::Matrix<double, N, N> (const double &, const  double &) >   &p_A,
        const std::function< Eigen::Matrix<double, N, 1> (const double &, const  double &) >   &p_B,
        const std::function< Eigen::Matrix<double, N, N> (const double &, const   double &) >     &p_C,
        const std::function< double (const double &, const Eigen::Matrix<double, N, 1>&, const double &) > &p_f,
        const Eigen::Array<double, N, 1>   &p_point,
        const double   &p_timeInit,
        const double   &p_T,
        SwitchDistrib &p_lawSwitch,
        std::function< double (const Eigen::Matrix<double, N, 1>&)>   &p_g,
        const std::vector<int> &p_nbSim,
        TRNGGenerator &p_gen)
{

    // mpi
    boost::mpi::communicator world;
    //  random generator
    p_gen.split(world.size(), world.rank());
    trng::normal_dist<double> normal(0., 1.);

    // global simulation number
    int nbSimul = p_nbSim[0];
    int nbSimulProc = nbSimul / world.size();

    // store PDE value and std associated for first level
    double  val = 0.;
    double valSq =  0.;
    int depth = 1;

    //#pragma omp parallel for schedule(dynamic,10)
    for (int is = 0; is <   nbSimulProc ; ++is)
    {
        if (world.rank() == 0)
            if (is % 1000 == 0)
                std::cout << " is " << is <<   std::endl ;
        double solLoc = solveOneStepExact<N, SwitchDistrib, TRNGGenerator>(p_point,  p_A, p_B, p_C,  p_f, p_timeInit, p_T, p_lawSwitch, normal, p_gen, p_g, p_nbSim, depth);
        val += solLoc;
        valSq += solLoc * solLoc;
    }
    double valRet = 0.;
    double valSqRet = 0.;
    boost::mpi::all_reduce(world, val, valRet, std::plus<double>());
    boost::mpi::all_reduce(world, valSq, valSqRet, std::plus<double>());
    int nbSim = nbSimulProc * world.size();
    valRet /= nbSim;
    valSqRet /= nbSim   ;
    valSqRet = sqrt((valSqRet - valRet * valRet) / nbSim);
    return std::make_tuple(valRet, valSqRet);
}

}

#endif
