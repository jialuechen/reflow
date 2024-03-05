#include "libflow/core/grids/FullGridGeners.h"
// add include for all derived classes
#include "libflow/core/grids/RegularLegendreGridDerivedGeners.h"
#include "libflow/core/grids/GeneralSpaceGridDerivedGeners.h"
#include "libflow/core/grids/RegularSpaceGridDerivedGeners.h"

// Register all wrappers
SerializationFactoryForFullGrid::SerializationFactoryForFullGrid()
{
    this->registerWrapper<RegularLegendreGridDerivedGeners>();
    this->registerWrapper<GeneralSpaceGridDerivedGeners>();
    this->registerWrapper<RegularSpaceGridDerivedGeners>();
}
