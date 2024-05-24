
#ifndef OPTIMIZESLEMISSIVE_H
#define OPTIMIZESLEMISSIVE_H
#include <vector>
#include <functional>
#include <boost/random.hpp>
#include <Eigen/Dense>
#include "reflow/core/grids/SpaceGrid.h"
#include "reflow/core/grids/InterpolatorSpectral.h"
#include "reflow/core/utils/comparisonUtils.h"
#include "reflow/semilagrangien/OptimizerSLBase.h"
#include "reflow/semilagrangien/SemiLagrangEspCond.h"


namespace reflow
{


class OptimizeSLEmissive : public OptimizerSLBase
{

private :

    double m_alpha; // mean reverting coefficient for demand
    double  m_m  ; // mean reverting value for demand
    double m_sig ; // demand volatility
    std::function< double(double, double)>   m_PI ;    // PI function depending on D and L the cumulative investment (gain for spot)
    std::function< double(double, double) >   m_cBar ;   // cbAR function depending on l (control) and L investment already achieved
    double m_s ;  // linear coefficient for subvention
    double m_dt ; // time step
    double m_lMax ; // max of investment control
    double m_lStep ; // max for control
    std::vector <std::array< double, 2>  > m_extrem ;// extremal values of the grid

public :

    /// \brief Constructor
    /// \param p_alpha     mean reverting coefficient for demand
    /// \param p_m         demand mean reverting
    /// \param p_sig       demand volatility
    /// \param p_PI        gain from spot
    /// \param p_cBar      investment cost
    /// \param p_s         linear coefficient for subvention
    /// \param p_dt        resolution time step
    /// \param p_lMax      max of the investment control
    /// \param p_lStep     Step discretization for l
    /// \param p_extrem    extremal values of the grid

    OptimizeSLEmissive(const double &p_alpha,  const double &p_m, const double &p_sig, const std::function<double(double, double)> &p_PI,
                       const std::function< double(double, double) >     &p_cBar,  const double   &p_s, const double &p_dt,
                       const  double &p_lMax, const double &p_lStep, const  std::vector <std::array< double, 2>  >   &p_extrem);

    /// \brief defines the dimension to split for MPI parallelism
    ///        For each dimension return true is the direction can be split
    Eigen::Array< bool, Eigen::Dynamic, 1> getDimensionToSplit() const;


    /// \brief define the diffusion cone for parallelism
    /// \param  p_regionByProcessor         region (min max) treated by the processor for the different regimes treated
    /// \return returns in each dimension the min max values in the stock that can be reached from the grid p_gridByProcessor for each regime
    std::vector< std::array< double, 2> > getCone(const  std::vector<  std::array< double, 2>  > &p_regionByProcessor) const;


    /// \brief defines a step in optimization
    /// \param p_point     coordinate of the point to treat
    /// \param p_semiLag   semi Lagrangian operator for each regime for solution at the previous step
    /// \param p_time      current time
    /// \param p_phiIn     value of the function at the previous time step at p_point for each regime
    /// \return a pair :
    ///          - first an array of the solution (for each regime)
    ///          - second an array of the optimal controls ( for each control)
    std::pair< Eigen::ArrayXd, Eigen::ArrayXd> stepOptimize(const Eigen::ArrayXd   &p_point,
            const std::vector< std::shared_ptr<SemiLagrangEspCond> > &p_semiLag,
            const double &p_time, const Eigen::ArrayXd &p_phiIn) const;




    /// \brief get number of regimes
    inline  int getNbRegime() const
    {
        return 2 ;
    }

    /// \brief do we modify the volatility to stay in the domain
    inline  bool getBModifVol() const
    {
        return false;
    }

    /// \brief defines a step in simulation
    /// \param p_gridNext      grid at the next step
    /// \param p_semiLag       semi Lagrangian operator at the current step in each regime
    /// \param p_state         state array (can be modified)
    /// \param p_iReg          regime number
    /// \param p_gaussian      Brownian realization
    /// \param p_phiInPt       value of the function at the previous time step at p_point for each regime
    /// \param p_phiInOut      defines the value function (modified)
    void stepSimulate(const SpaceGrid   &p_gridNext,
                      const  std::vector< std::shared_ptr< reflow::SemiLagrangEspCond> > &p_semiLag,
                      Eigen::Ref<Eigen::ArrayXd>  p_state,   int &p_iReg,
                      const Eigen::ArrayXd &p_gaussian,
                      const Eigen::ArrayXd &p_phiInPt,
                      Eigen::Ref<Eigen::ArrayXd> p_phiInOut) const   ;


    /// \brief defines a step in simulation using the control calculated in optimization
    void stepSimulateControl(const SpaceGrid &,  const  std::vector< std::shared_ptr< InterpolatorSpectral> > &,
                             Eigen::Ref<Eigen::ArrayXd>,   int &, const Eigen::ArrayXd &,  Eigen::Ref<Eigen::ArrayXd>) const {}



    /// \brief get the number of Brownians involved in semi Lagrangian for simulation
    inline int getBrownianNumber() const
    {
        return 1;
    }
    // \brief get back the dimension of the control
    inline int getNbControl() const
    {
        return 1 ;
    }

    /// \brief get size of the  function to follow in simulation
    virtual int getSimuFuncSize() const
    {
        return 2;
    }

    /// \brief Permit to deal with some boundary points that do not need boundary conditions
    ///        Return false if the point on the boundary need some boundary conditions
    /// \param  p_point  potentially on the  boundary
    inline  bool isNotNeedingBC(const Eigen::ArrayXd &p_point)  const
    {
        if ((reflow::almostEqual(p_point(0), m_extrem[0][0], 10)) || (reflow::almostEqual(p_point(0), m_extrem[0][1], 10))
                || (reflow::almostEqual(p_point(1), m_extrem[1][1], 10)))
            return false;
        return true;
    }
};
}
#endif /* OPTIMIZESLEMISSIVE_H */
