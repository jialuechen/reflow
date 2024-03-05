
#ifndef OPTIMIZERSLBASE_H
#define OPTIMIZERSLBASE_H
#include <vector>
#include <Eigen/Dense>
#include "libflow/core/grids/SpaceGrid.h"
#include "libflow/core/grids/FullGrid.h"
#include "libflow/core/grids/InterpolatorSpectral.h"
#include "libflow/semilagrangien/SemiLagrangEspCond.h"

/** \file OptimizerSLBase.h
 *  \brief Define an abstract class for Dynamic Programming problems
 *     \author Xavier Warin
 */

namespace libflow
{

/// \class OptimizerSLBase OptimizerSLBase.h
///  Base class for optimizer for resolution by semi Lagrangian methods of HJB equations
class OptimizerSLBase
{


public :

    OptimizerSLBase() {}

    virtual ~OptimizerSLBase() {}


    /// \brief define the diffusion cone for parallelism
    /// \param  p_regionByProcessor         region (min max) treated by the processor for the different regimes treated
    /// \return returns in each dimension the min max values in the stock that can be reached from the grid p_gridByProcessor for each regime
    virtual std::vector< std::array< double, 2> > getCone(const  std::vector<  std::array< double, 2>  > &p_regionByProcessor) const = 0;

    /// \brief defines the dimension to split for MPI parallelism
    ///        For each dimension return true is the direction can be split
    virtual Eigen::Array< bool, Eigen::Dynamic, 1> getDimensionToSplit() const = 0 ;

    /// \brief defines a step in optimization
    /// \param p_point     coordinates of the point to treat
    /// \param p_semiLag   semi Lagrangian operator for each regime for solution at the previous step
    /// \param p_time      current date
    /// \param p_phiInPt   value of the function at the previous time step at p_point for each regime
    /// \return a pair :
    ///          - first an array of the solution (for each regime)
    ///          - second an array of the optimal controls ( for each control)
    virtual std::pair< Eigen::ArrayXd, Eigen::ArrayXd>  stepOptimize(const Eigen::ArrayXd   &p_point,
            const std::vector< std::shared_ptr<SemiLagrangEspCond> > &p_semiLag,
            const double &p_time,
            const Eigen::ArrayXd &p_phiInPt) const = 0;


    /// \brief defines a step in simulation
    /// \param p_gridNext      grid at the next step
    /// \param p_semiLag       semi Lagrangian operator at the current step in each regime
    /// \param p_state         state array (can be modified)
    /// \param p_iReg          regime number
    /// \param p_gaussian      unitary Gaussian realization
    /// \param p_phiInPt       value of the function at the next time step at p_point for each regime
    /// \param p_phiInOut      defines the value functions (modified) to follow
    virtual void stepSimulate(const SpaceGrid   &p_gridNext,
                              const  std::vector< std::shared_ptr< libflow::SemiLagrangEspCond> > &p_semiLag,
                              Eigen::Ref<Eigen::ArrayXd>  p_state,   int &p_iReg,
                              const Eigen::ArrayXd &p_gaussian,
                              const Eigen::ArrayXd &p_phiInPt,
                              Eigen::Ref<Eigen::ArrayXd> p_phiInOut) const  = 0 ;


    /// \brief defines a step in simulation using the control calculated in optimization
    /// \param p_gridNext      grid at the next step
    /// \param p_controlInterp the optimal controls interpolator
    /// \param p_state         state array (can be modified)
    /// \param p_iReg          regime number
    /// \param p_gaussian      unitary Gaussian realization
    /// \param p_phiInOut      defines the value functions (modified) to follow
    virtual void stepSimulateControl(const SpaceGrid    &p_gridNext,
                                     const  std::vector< std::shared_ptr< InterpolatorSpectral> >   &p_controlInterp,
                                     Eigen::Ref<Eigen::ArrayXd>  p_state,   int &p_iReg,
                                     const Eigen::ArrayXd &p_gaussian,
                                     Eigen::Ref<Eigen::ArrayXd> p_phiInOut) const  = 0 ;

    /// \brief get number of regimes
    virtual   int getNbRegime() const = 0 ;

    /// \brief get back the dimension of the control
    virtual int getNbControl() const = 0 ;

    /// \brief do we modify the volatility to stay in the domain
    virtual  bool getBModifVol() const = 0 ;

    /// \brief get the number of Brownians involved in semi Lagrangian for simulation
    virtual int getBrownianNumber() const = 0 ;

    /// \brief get size of the  function to follow in simulation
    virtual int getSimuFuncSize() const = 0;

    /// \brief Permit to deal with some boundary points that do not need boundary conditions
    ///        Return false if all points on the boundary need some boundary conditions
    /// \param  p_point  potentially on the  boundary
    virtual bool isNotNeedingBC(const Eigen::ArrayXd &p_point)  const = 0;
};
}
#endif /* OPTIMIZERSLBASE_H */
