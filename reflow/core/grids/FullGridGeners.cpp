#include "reflow/core/grids/FullGridGeners.h"
// add include for all derived classes
#include "reflow/core/grids/RegularLegendreGridDerivedGeners.h"
#include "reflow/core/grids/GeneralSpaceGridDerivedGeners.h"
#include "reflow/core/grids/RegularSpaceGridDerivedGeners.h"

// Register all wrappers
SerializationFactoryForFullGrid::SerializationFactoryForFullGrid()
{
    this->registerWrapper<RegularLegendreGridDerivedGeners>();
    this->registerWrapper<GeneralSpaceGridDerivedGeners>();
    this->registerWrapper<RegularSpaceGridDerivedGeners>();
}
