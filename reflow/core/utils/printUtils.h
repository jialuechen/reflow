
#ifndef PRINTUTILS_H
#define  PRINTUTILS_H
#include "reflow/core/utils/RegularSpacegrid.h"
#include "reflow/core/utils/GeneralSpacegrid.h"
#include "reflow/regression/LocalLinearRegression.h"
#include "reflow/regression/ContinuationValue.h"

namespace reflow
{


/// \brief Function to dump a RegularSpaceGrid
std::ostream &operator<<(std::ostream &os, const RegularSpaceGrid &grid)
{
    os << "Regular grid  " << std::endl ;
    os << " Low values " <<  grid.getLowValues() << std::endl ;
    os << " Step       "   <<  grid.getStep() << std::endl ;
    os << " nbStep     "   <<  grid.getNbStep() << std::endl ;
    return os;
}



/// \brief Function to dump a GeneralSpaceGrid
std::ostream &operator<<(std::ostream &os, const GeneralSpaceGrid   &grid)
{
    os  << " General Mesh " << grid.getMeshPerDimension().size() <<   std::endl ;
    for (int id = 0 ; id < grid.getMeshPerDimension().size(); ++id)
    {
        os << " Direction " << id << " meshes " <<  *(grid.getMeshPerDimension()[id]) << std::endl ;
    }
    return os;
}

/// \brief To dump a LocalLinearRegression object
std::ostream &operator<<(std::ostream &os, const LocalLinearRegression   &regressor)
{
    os << " Local regressor  : zero date ?" <<  regressor.getBZerodate() <<  "  m_mesh1D "  <<  regressor.m_mesh1D.size()  <<  std::endl ;
    for (int id = 0 ; id < regressor.getMesh1D().size() ; ++id)
    {
        os << " Direction " << id << " meshes " <<  *(regressor..getMesh1D()[id]) << std::endl ;
    }
    return os;
}

/// \brief Function to dump a ContinuationValue
template< class Grid, class  CondExpectation >
std::ostream &operator<<(std::ostream &os, const ContinuationValue< Grid, CondExpectation >    &cont)
{
    os  << " Continuation value " <<   std::endl ;
    if (cont.getGrid() != 0)
        os  << " Grid " << *cont.getGrid() << std::endl ;
    else
        os << " No grid " << std::endl ;
    if (cont.getCondExp() != 0)
        os  << " Conditional expectation object  " << *cont.getCondExp() << std::endl ;
    else
        os  << " No Cond exp" << std::endl ;
    if (cont.getValues().size() > 0)
        os  << " Regressed by stock " << cont.getValues() << std::endl ;
    else
        os << "No regressed values "  << std::endl ;
    return os ;
}
}
#endif /*  PRINTUTILS_H */
