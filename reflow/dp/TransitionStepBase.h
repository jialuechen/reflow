
#ifndef TRANSITIONSTEPBASE_H
#define TRANSITIONSTEPBASE_H
#include <vector>
#include <memory>
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "reflow/regression/GridAndRegressedValue.h"

/** \file TransitionStepBase.h
 * \brief Base class to optimize on a time step in  methods by Monte Carlo but without regression
 * \author Xavier Warin
 */

namespace reflow
{

/// \class TransitionStepBase TransitionStepBase.h
///        Defines one step of dynamic programming without regression
class TransitionStepBase
{
public :

    /// \brief One step for dynamic programming in optimization
    /// \param p_phiIn      for each regime the function value at the next time step
    /// \return     solution obtained after one step of dynamic programming and the optimal control
    virtual std::pair<  std::shared_ptr< std::vector< Eigen::ArrayXXd  > >,  std::shared_ptr<std::vector< Eigen::ArrayXXd   > > >   oneStep(const std::vector< Eigen::ArrayXXd  > &p_phiIn) const  = 0 ;

};
}

#endif
