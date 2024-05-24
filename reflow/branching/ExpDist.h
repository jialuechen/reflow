#ifndef EXPDIST_H
#define EXPDIST_H
#include <cmath>
#include <trng/yarn2.hpp>
#include <trng/uniform01_dist.hpp>

/** \file ExpDist.h
 * \brief Generates a random number with an expoential law with density
 *   \[
 *       \lambda e^{ - \lambda x}
 *   \]
 */

namespace reflow
{


class ExpDist
{
private:

    double m_lambda; ///< lambda parameter

    /// \brief  uniform law
    trng::uniform01_dist<double>  m_uniform;

public :

    /// \brief constructor
    /// \param p_lambda  lambda parameter
    ExpDist(const double &p_lambda) : m_lambda(p_lambda) {}

    /// \brief enerates a random number with exponential  distribution
    /// \param p_gen  TRNG generrator
    template< class TRNGGenerator >
    inline double operator()(TRNGGenerator  &p_gen)
    {
        return - std::log(1. - m_uniform(p_gen)) / m_lambda;
    }

    /// \brief gives the PDF
    double pdf(const double &p_x) const
    {
        return m_lambda * exp(-m_lambda * p_x);
    }

    /// \brief Gives the CDF
    double cdf(const double &p_x) const
    {
        return 1 - exp(-m_lambda * p_x);
    }
};
}

#endif
