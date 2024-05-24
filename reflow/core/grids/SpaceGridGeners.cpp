#include "reflow/core/grids/SpaceGridGeners.h"
// add include for all derived classes
#include "reflow/core/grids/RegularLegendreGridGeners.h"
#include "reflow/core/grids/GeneralSpaceGridGeners.h"
#include "reflow/core/grids/RegularSpaceGridGeners.h"
#include "reflow/core/grids/SparseSpaceGridBoundGeners.h"
#include "reflow/core/grids/SparseSpaceGridNoBoundGeners.h"

// Register all wrappers
SerializationFactoryForSpaceGrid::SerializationFactoryForSpaceGrid()
{
    this->registerWrapper<RegularLegendreGridGeners>();
    this->registerWrapper<GeneralSpaceGridGeners>();
    this->registerWrapper<RegularSpaceGridGeners>();
    this->registerWrapper<SparseSpaceGridBoundGeners>();
    this->registerWrapper<SparseSpaceGridNoBoundGeners>();
}
