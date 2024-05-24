// Copyright (C) 2020 EDF

#ifndef FASTCDFONSAMPLE_H
#define FASTCDFONSAMPLE_H


/** \file fastCDFOnSample.h
 *  \brief Calculate CDF with fast divide and conquer in  Langrené-Warin 2020 :
 *         "Fast multivariate empirical cumulative distribution function with connection to kernel density estimation"
 *          The CDF estimated at the sample points
 *  \author Xavier Warin
 */


namespace reflow
{
/// \brief Calculate fast CDF
/// \param p_x particules (sample)  size : (dimension, nbSim)
/// \param p_y  estimate for each p_x point
/// \return for each point of the grid return the CDF
Eigen::ArrayXd fastCDFOnSample(const Eigen::ArrayXXd &p_x, const Eigen::ArrayXd &p_y);

}
#endif /* FASTCDFONSAMPLE_H */


