#include "reflow/core/grids/InterpolatorSpectralGeners.h"
// add include for all derived classes
#include "reflow/core/grids/LinearInterpolatorSpectralGeners.h"
#include "reflow/core/grids/LegendreInterpolatorSpectralGeners.h"
#include "reflow/core/grids/SparseInterpolatorSpectralGeners.h"

// Register all wrappers
SerializationFactoryForInterpolatorSpectral::SerializationFactoryForInterpolatorSpectral()
{
    this->registerWrapper<LinearInterpolatorSpectralGeners>();
    this->registerWrapper<LegendreInterpolatorSpectralGeners>();
    this->registerWrapper<SparseInterpolatorSpectralGeners>();
}
