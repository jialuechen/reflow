// Copyright (C) 2020 EDF

#include <Eigen/Dense>
#include <libflow/cdf/nDDominanceAlone.h>

using namespace Eigen ;

namespace libflow
{

ArrayXd fastCDFOnSample(const Eigen::ArrayXXd &p_x, const Eigen::ArrayXd &p_y)
{

    ArrayXd cdfDivid(p_y.size());

    // dominance excluding current point
    nDDominanceAlone(p_x, p_y, cdfDivid);
    cdfDivid += p_y;
    return cdfDivid / p_y.size();
}
}

