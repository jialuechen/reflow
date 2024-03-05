
#ifdef USE_MPI
#include <memory>
#include <Eigen/Dense>
#include "libflow/core/utils/types.h"
#include "libflow/core/grids/FullGrid.h"
#include "libflow/core/parallelism/ParallelComputeGridSplitting.h"

using namespace Eigen;
using namespace std;

namespace libflow
{
ArrayXd reconstructProc0Mpi(const ArrayXd &p_point, const shared_ptr< FullGrid> &p_grid,  const shared_ptr< ArrayXXd >    &p_values, const Eigen::Array< bool, Eigen::Dynamic, 1>   &p_bdimToSplit,  const boost::mpi::communicator   &p_world)
{
    int nDim = p_grid->getDimension();
    ArrayXi initialDimension   = p_grid->getDimensions();
    // organize the hypercube splitting for parallel
    ArrayXi splittingRatio = paraOptimalSplitting(initialDimension, p_bdimToSplit, p_world);
    // for parallelization
    ParallelComputeGridSplitting parall(initialDimension, splittingRatio, p_world);
    // create the subgrid
    SubMeshIntCoord retGrid(nDim);
    // define the grid for reconstruction
    ArrayXi pMin = p_grid->lowerPositionCoord(p_point);
    ArrayXi pMax = p_grid->upperPositionCoord(p_point);
    for (int id = 0; id <  nDim; ++id)
    {
        retGrid(id)[0] = pMin(id);
        retGrid(id)[1] = pMax(id) + 1;
    }
    // new grid
    shared_ptr<FullGrid> grid = p_grid->getSubGrid(retGrid);
    // values on grid
    ArrayXXd valuesExtended;
    if (p_values)
    {
        // values on grid
        valuesExtended = parall.reconstruct<double>(*p_values, retGrid);
    }
    // now interpolated
    if (p_world.rank() == 0)
        return (grid->createInterpolator(p_point)->applyVec(valuesExtended));
    return ArrayXd::Zero(1);
}
}
#endif
