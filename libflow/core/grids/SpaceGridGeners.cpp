#include "libflow/core/grids/SpaceGridGeners.h"
// add include for all derived classes
#include "libflow/core/grids/RegularLegendreGridGeners.h"
#include "libflow/core/grids/GeneralSpaceGridGeners.h"
#include "libflow/core/grids/RegularSpaceGridGeners.h"
#include "libflow/core/grids/SparseSpaceGridBoundGeners.h"
#include "libflow/core/grids/SparseSpaceGridNoBoundGeners.h"

// Register all wrappers
SerializationFactoryForSpaceGrid::SerializationFactoryForSpaceGrid()
{
    this->registerWrapper<RegularLegendreGridGeners>();
    this->registerWrapper<GeneralSpaceGridGeners>();
    this->registerWrapper<RegularSpaceGridGeners>();
    this->registerWrapper<SparseSpaceGridBoundGeners>();
    this->registerWrapper<SparseSpaceGridNoBoundGeners>();
}
