
#ifndef TRANSITIONSTEPTREEBASE_H
#define TRANSITIONSTEPTREEBASE_H
#include <vector>
#include <memory>
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "libflow/tree/Tree.h"

/** \file TransitionStepTreeBase.h
 * \brief Base class to optimize on a time step in tree methods
 * \author Xavier Warin
 */

namespace libflow
{

/// \class TransitionStepTreeBase TransitionStepTreeBase.h
///        Defines one step of dynamic programming
class TransitionStepTreeBase
{
public :

    /// \brief One step for dynamic programming in optimization
    /// \param p_phiIn      for each regime the function value ( nb node at next date, nb stocks )
    /// \param p_condExp    Tree object to calculate conditional expectatons
    /// \return     solution obtained after one step of dynamic programming and the optimal control
    virtual std::pair< std::vector< std::shared_ptr< Eigen::ArrayXXd > >, std::vector<  std::shared_ptr<  Eigen::ArrayXXd > > >   oneStep(const std::vector< std::shared_ptr< Eigen::ArrayXXd > > &p_phiIn,
            const std::shared_ptr< Tree>     &p_condExp) const  = 0 ;

};
}

#endif
