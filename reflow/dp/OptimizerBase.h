
#ifndef OPTIMIZERBASE_H
#define OPTIMIZERBASE_H
#include <Eigen/Dense>

/** \file OptimizerBase.h
 *  \brief Define an abstract class for Dynamic Programming problems solved by Monte Carlo methods
 *     \author Xavier Warin
 */

namespace reflow
{

/// \class OptimizerBase OptimizerBase.h
///  Base class for optimizer for Dynamic Programming with and without regression methods
class OptimizerBase
{


public :

    OptimizerBase() {}

    virtual ~OptimizerBase() {}

    /// \brief defines the dimension to split for MPI parallelism
    ///        For each dimension return true is the direction can be split
    virtual Eigen::Array< bool, Eigen::Dynamic, 1> getDimensionToSplit() const = 0 ;

    /// \brief defines the diffusion cone for parallelism
    /// \param  p_regionByProcessor         region (min max) treated by the processor for the different regimes treated
    /// \return returns in each dimension the min max values in the stock that can be reached from the grid p_gridByProcessor for each regime
    virtual std::vector< std::array< double, 2> > getCone(const  std::vector<  std::array< double, 2>  > &p_regionByProcessor) const = 0;


    /// \brief Get the number of regimes allowed for the asset to be reached  at the current time step
    virtual   int getNbRegime() const = 0 ;


    /// \brief get back the dimension of the control
    virtual int getNbControl() const = 0 ;

    /// \brief get size of the  function to follow in simulation
    virtual int getSimuFuncSize() const = 0;

};
}
#endif /* OPTIMIZERBASE_H */
