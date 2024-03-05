#include "libflow/core/grids/InterpolatorSpectralGeners.h"
// add include for all derived classes
#include "libflow/core/grids/LinearInterpolatorSpectralGeners.h"
#include "libflow/core/grids/LegendreInterpolatorSpectralGeners.h"
#include "libflow/core/grids/SparseInterpolatorSpectralGeners.h"

// Register all wrappers
SerializationFactoryForInterpolatorSpectral::SerializationFactoryForInterpolatorSpectral()
{
    this->registerWrapper<LinearInterpolatorSpectralGeners>();
    this->registerWrapper<LegendreInterpolatorSpectralGeners>();
    this->registerWrapper<SparseInterpolatorSpectralGeners>();
}
