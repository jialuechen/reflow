
#ifndef TRANSITIONSTEPREGRESSIONBASE_H
#define TRANSITIONSTEPREGRESSIONBASE_H
#include <vector>
#include <memory>
#ifdef USE_MPI
#include "boost/mpi.hpp"
#endif
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "libflow/regression/BaseRegression.h"

/** \file TransitionStepRegressionBase.h
 * \brief Base class to optimize on a time step in regression methods
 * \author Xavier Warin
 */

namespace libflow
{

/// \class TransitionStepRegressionBase TransitionStepRegressionBase.h
///        Defines one step of dynamic programming
class TransitionStepRegressionBase
{
public :

    /// \brief One step for dynamic programming in optimization
    /// \param p_phiIn      for each regime the function value ( nb simulation, nb stocks )
    /// \param p_condExp    Conditional expectation objet
    /// \return     solution obtained after one step of dynamic programming and the optimal control
    virtual std::pair< std::vector< std::shared_ptr< Eigen::ArrayXXd > >, std::vector<  std::shared_ptr<  Eigen::ArrayXXd > > >   oneStep(const std::vector< std::shared_ptr< Eigen::ArrayXXd > > &p_phiIn,
            const std::shared_ptr< BaseRegression>     &p_condExp) const  = 0 ;

};
}

#endif
