// Copyright (C) 2016 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#ifdef USE_MPI
#include <memory>
#include <Eigen/Dense>
#include "libflow/core/utils/types.h"
#include "libflow/core/grids/RegularSpaceIntGrid.h"
#include "libflow/core/parallelism/ParallelComputeGridSplitting.h"

using namespace Eigen;
using namespace std;

namespace libflow
{
ArrayXd reconstructProc0ForIntMpi(const ArrayXi &p_point, const shared_ptr< RegularSpaceIntGrid> &p_grid,  const shared_ptr< ArrayXXd >    &p_values, const Eigen::Array< bool, Eigen::Dynamic, 1>   &p_bdimToSplit, const boost::mpi::communicator &p_world)
{
    int nDim = p_grid->getDimension();
    ArrayXi initialDimension   = p_grid->getDimensions();
    // organize the hypercube splitting for parallel
    ArrayXi splittingRatio = paraOptimalSplitting(initialDimension, p_bdimToSplit, p_world);
    // for parallelization
    ParallelComputeGridSplitting parall(initialDimension, splittingRatio, p_world);
    // create the subgrid
    SubMeshIntCoord retGrid(nDim);
    for (int id = 0; id <  nDim; ++id)
    {
        retGrid(id)[0] = p_point(id) - p_grid->getLowValueDim(id);
        retGrid(id)[1] = p_point(id) - p_grid->getLowValueDim(id) + 1;
    }
    // new grid
    shared_ptr<RegularSpaceIntGrid> grid = p_grid->getSubGrid(retGrid);
    // values on grid
    ArrayXXd valuesExtended;
    if (p_values)
    {
        // values on grid
        valuesExtended = parall.reconstruct<double>(*p_values, retGrid);
    }
    // now the extendend value has only one point
    if (p_world.rank() == 0)
        return valuesExtended.col(0);
    return ArrayXd::Zero(1);
}
}
#endif
