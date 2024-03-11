
#ifndef OPTIMIZERSLCASE1_H
#define OPTIMIZERSLCASE1_H
#include <vector>
#include <functional>
#include <Eigen/Dense>
#include "libflow/core/grids/SpaceGrid.h"
#include "libflow/semilagrangien/OptimizerSLBase.h"
#include "libflow/semilagrangien/SemiLagrangEspCond.h"

namespace libflow
{

/// \class OptimizerSLCase1 OptimizeSLCase1.h
///  class for optimizer for the  resolution of PDE by semi Lagrangian methods of HJB equations (test case 1)
class OptimizerSLCase1 : public OptimizerSLBase
{

private :

    double m_dt ; // time step

    /// \brief define source term class
    class SourceTerm
    {
    public :

        SourceTerm() {};

        double operator()(const double &t, const Eigen::ArrayXd &x)
        {
            if (x(0) < 0)
                return  sin(x(1) / 2.) * sin(x(0) / 2.) * (1. + ((1. + t) / 4.) * (pow(sin(x(0)), 2.) + pow(sin(x(1)), 2.)))  - sin(x(0)) * sin(x(1)) * cos(x(1) / 2.) * ((1. + t) / 2.) * cos(x(0) / 2.);
            else
                return  sin(x(1) / 2.) * sin(x(0) / 4.) * (1 + (1 + t) / 16.* (pow(sin(x(0)), 2.) + 4.*pow(sin(x(1)), 2.)))  - sin(x(0)) * sin(x(1)) * cos(x(1) / 2.) * (1. + t) / 4.* cos(x(0) / 4.);
        }
    };

    /// \brief Defines sigma term
    class SinSomme1
    {
    public :
        SinSomme1() {};

        double operator()(const   Eigen::ArrayXd &x)
        {
            return sqrt(2.) * sin(x(0));
        }
    };

    class SinSomme2
    {
    public :
        SinSomme2() {};

        double operator()(const   Eigen::ArrayXd &x)
        {
            return sqrt(2.) * sin(x(1));
        }
    };

public :

    /// \brief constructor
    /// \param p_dt time step resolution
    OptimizerSLCase1(const double &p_dt): m_dt(p_dt) {}


    /// \brief define the diffusion cone for parallelism
    /// \param  p_regionByProcessor         region (min max) treated by the processor for the different regimes treated
    /// \return returns in each dimension the min max values in the stock that can be reached from the grid p_gridByProcessor for each regime
    std::vector< std::array< double, 2> > getCone(const  std::vector<  std::array< double, 2>  > &p_regionByProcessor) const;


    /// \brief defines a step in optimization
    /// \param p_point     coordinate of the point to treat
    /// \param p_semiLag   semi Lagrangian operator for each regime for solution at the previous step
    /// \param p_time      current time
    /// \param p_phiInPt   value of the function at the previous time step at p_point for each regime
    /// \return a pair :
    ///          - first an array of the solution (for each regime)
    ///          - second an array of the optimal controls ( for each control)
    std::pair< Eigen::ArrayXd, Eigen::ArrayXd>  stepOptimize(const Eigen::ArrayXd   &p_point,
            const std::vector< std::shared_ptr<SemiLagrangEspCond> > &p_semiLag, const double &p_time,
            const Eigen::ArrayXd &p_phiInPt) const;



    /// \brief get back the dimension of the control
    virtual int getNbControl() const
    {
        return 1;
    }

    /// \brief get number of regimes
    inline  int getNbRegime() const
    {
        return 1 ;
    }
    /// \brief do we modify the volatility to stay in the domain
    inline  bool getBModifVol() const
    {
        return true;
    }

    /// \brief defines a step in simulation
    void stepSimulate(const libflow::SpaceGrid &,
                      const std::vector< std::shared_ptr< libflow::SemiLagrangEspCond> > &,
                      Eigen::Ref<Eigen::ArrayXd>,   int &,
                      const Eigen::ArrayXd &,
                      const Eigen::ArrayXd &, Eigen::Ref<Eigen::ArrayXd>) const  {}


    /// \brief defines a step in simulation using the control calculated in optimization
    virtual void stepSimulateControl(const SpaceGrid &,
                                     const  std::vector< std::shared_ptr< InterpolatorSpectral> > &,
                                     Eigen::Ref<Eigen::ArrayXd>,   int &,
                                     const Eigen::ArrayXd &,
                                     Eigen::Ref<Eigen::ArrayXd>) const {}


    /// \brief get the number of Brownians involved in semi Lagrangian for simulation
    inline int getBrownianNumber() const
    {
        return 0;
    }

    /// \brief get size of the  function to follow in simulation
    inline int getSimuFuncSize() const
    {
        return 0;
    }

    /// \brief Permit to deal with some boundary points that do not need boundary conditions
    ///        Return false if all points on the boundary need some boundary conditions
    /// \param  p_point  potentially on the  boundary
    inline  bool isNotNeedingBC(const Eigen::ArrayXd &p_point)  const
    {
        return false;
    }

    /// \brief defines the dimension to split for MPI parallelism
    ///        For each dimension return true is the direction can be split
    Eigen::Array< bool, Eigen::Dynamic, 1> getDimensionToSplit() const
    {
        Eigen::Array< bool, Eigen::Dynamic, 1> bDim = Eigen::Array< bool, Eigen::Dynamic, 1>::Constant(2, true);
        return  bDim ;
    }
};
}
#endif /* OPTIMIZERSLCASE1_H */
