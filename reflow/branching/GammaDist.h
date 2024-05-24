
#ifndef GAMMADIST_H
#define GAMMADIST_H
#include <boost/math/special_functions.hpp>
#include <trng/yarn2.hpp>
#include <trng/uniform01_dist.hpp>
#include <trng/normal_dist.hpp>

/** \file GammaDist.h
 *  \brief Generates random Number with Gamma distribution from TRNG library
 *  The pdf
 * \[
 *   \theta^{-k} x^{k-1} \frac{e^{-\frac{ x}{\theta}}}{\Gamma(k)}
 * \]
 */

namespace reflow
{
class GammaDist
{
private:
    double m_k ; ///< k value
    double m_theta ;///< theta value
    /// uniform law parallel generator
    trng::uniform01_dist<double>  m_uniform;
    /// \brief normal law gaussian generator
    trng::normal_dist<double> m_normal;

    /// \brief gamma with m_theta =1
    /// \param p_k  k  value
    /// \param p_gen random number generator
    template< class TRNGGenerator >
    inline double gamma(double p_k,  TRNGGenerator   &p_gen)
    {
        if (p_k < 1)
        {
            return gamma(p_k + 1, p_gen) * pow(m_uniform(p_gen), 1. / m_k);
        }
        else
        {
            double d = p_k - 1. / 3.;
            double c = 1. / sqrt(9.*d);
            bool bCont = true;
            double V = 0 ;
            while (bCont)
            {
                double Z = m_normal(p_gen);
                if (Z > -1. / c)
                {
                    V = pow(1 + c * Z, 3.);
                    double U = m_uniform(p_gen);
                    while (U < 1e-30)
                        U = m_uniform(p_gen);
                    bCont = (log(U) > 0.5 * Z * Z + d - d * V + d * log(V));
                }
            }
            return d * V;
        }
    }

public :

    /// \brief Constructor
    /// \param p_k  k parametr
    /// \param p_theta theta parameter
    GammaDist(const double &p_k, const double &p_theta) : m_k(p_k), m_theta(p_theta), m_normal(0., 1.) {}

    /// \brief generates a random number with gamma distribution
    /// \param p_gen  TRNG generrator
    inline double operator()(trng::yarn2   &p_gen)
    {
        return gamma(m_k, p_gen) * m_theta;
    }

    /// \brief gives the PDF
    inline double pdf(const double &p_x)
    {
        return pow(p_x, m_k - 1) * exp(-p_x / m_theta) / (boost::math::tgamma(m_k) * pow(m_theta, m_k));
    }

    /// \brief Gives the CDF
    inline double cdf(const double &p_x)
    {
        return boost::math::gamma_p(m_k, p_x / m_theta);
    }
};

}
#endif
