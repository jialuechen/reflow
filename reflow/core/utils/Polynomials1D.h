
#ifndef _POLYNOMIALS1D_H
#define _POLYNOMIALS1D_H



namespace reflow
{
/**
 * \defgroup Polynomial Single variate polynomials
 * \brief It Implements single variate polynomials. each class must provide
 * three methods f, Df, D2f to compute the function value, its first
 * derivatives and its second derivative.
 */
/**@{*/

/**
 * \class Canonical
 * \brief Implementation of the canonical polynomials with one variate
 * P(X, n) = X^n
 */
class Canonical
{
public:
    /**
     * \brief Evaluate the canonical Polynomial
     * \param p_x point at which to evaluate the polynomial
     * \param p_n degree of the polynomial
     * \return P(p_x, p_n)
     */
    double F(double p_x, int p_n) const;
    /**
     * \brief Evaluate the first derivative of the canonical Polynomial
     * \param p_x point at which to evaluate the polynomial
     * \param p_n degree of the polynomial
     * \return P'(p_x, p_n)
     */
    double DF(double p_x, int p_n) const;
    /**
     * \brief Evaluate the second derivative of the canonical Polynomial
     * \param p_x point at which to evaluate the polynomial
     * \param p_n degree of the polynomial
     * \return P''(p_x, p_n)
     */
    double D2F(double p_x, int p_n) const;
};

/**
 * \class Tchebychev
 * \brief Implementation of the Tchebychev polynomials with one variate
 * P(X, n + 2) = 2 X P(X, n + 1) - P(X, n)
 */
class Tchebychev
{
public:
    /**
     * \brief Evaluate the Tchebychev Polynomial
     *
     * \param p_x point at which to evaluate the polynomial
     * \param p_n degree of the polynomial
     * \return P(p_x, p_n)
     */
    double F(double p_x, int p_n) const;
    /**
     * \brief Evaluate the first derivative of the Tchebychev Polynomial
     *
     * \param p_x point at which to evaluate the polynomial
     * \param p_n degree of the polynomial
     * \return P'(p_x, p_n)
     */
    double DF(double p_x, int p_n) const;
    /**
     * \brief Evaluate the second derivative of the Tchebychev Polynomial
     *
     * \param p_x point at which to evaluate the polynomial
     * \param p_n degree of the polynomial
     * \return P''(p_x, p_n)
     */
    double D2F(double p_x, int p_n) const;
private:
    /**
     * \brief The terminal recursive function to compute Tchebychev polynomials of any
     * order. This function is only used for order > 7
     *  \param p_x the evaluation point
     *  \param p_n the order of the polynomial to be evaluated
     *  \param p_n0 initialization of the recurrence
     *  \param p_fn0 used to store the polynomial of order p_n0.
     *  \param p_fn0_1 used to store the polynomial of order p_n0 - 1.
     */
    double FRec(double p_x, int p_n, int p_n0, double &p_fn0, double &p_fn0_1) const;
    /**
     * \brief The terminal recursive function to compute the first derivative of the
     * Tchebychev polynomials of any order. This function is only used for order > 7
     *
     *  \param p_x the evaluation point
     *  \param p_n the order of the polynomial to be evaluated
     *  \param p_n0 initialization of the recurrence
     *  \param Dfn0 used to store the derivative of the polynomial of order p_n0.
     *  \param Dfn0_1 used to store the derivative of the polynomial of order p_n0 - 1.
     */
    double DFRec(double p_x, int p_n, int p_n0, double &p_dfn0, double &p_dfn0_1) const;
    /**
     * \brief The terminal recursive function to compute the second derivative of the
     * Tchebychev polynomials of any order. This function is only used for order >  7
     *
     *  \param p_x point to evaluate at
     *  \param p_n the order of the polynomial
     *  \param p_n0 initialization of the recurrence
     *  \param D2fn0 used to store the second derivative of the polynomial of order p_n0.
     *  \param D2fn0_1 used to store the second derivative of the polynomial of order p_n0 - 1.
     */
    double D2FRec(double p_x, int p_n, int p_n0, double &p_d2fn0, double &p_d2fn0_1) const;
};

/**
 * \class Hermite
 * \brief Implementation of the Hermite polynomials with one variate.
 *  P(X, n + 1) = X P(X, n) - 2 n P(X, n - 1)
 * These polynomials are orthogonal for the normal distribution and morover if
 * G is a standard normal random variable E[P(G, n)] = n!
 */
class Hermite
{
public:

    /**
     * \brief Evaluate the Hermite Polynomial
     *
     * \param p_x point at which to evaluate the polynomial
     * \param p_n degree of the polynomial
     * \return P(p_x, p_n)
     */
    double F(double p_x, int p_n) const;
    /**
     * \brief Evaluate the first derivative of the Hermite Polynomial
     *
     * \param p_x point at which to evaluate the polynomial
     * \param p_n degree of the polynomial
     * \return P'(p_x, p_n)
     */
    double DF(double p_x, int p_n) const;
    /**
     * \brief Evaluate the second derivative of the Hermite Polynomial
     *
     * \param p_x point at which to evaluate the polynomial
     * \param p_n degree of the polynomial
     * \return P''(p_x, p_n)
     */
    double D2F(double p_x, int p_n) const;
private:
    /**
     * \brief The terminal recursive function to compute Hermite polynomials of any
     * order. This function is only used for order > 7
     *
     *  \param p_x point at which to evaluate the polynomial
     *  \param p_n the order of the polynomial to be evaluated
     *  \param p_n0 rank of initialization
     *  \param p_fn0 used to store the polynomial of order p_n0.
     *  \param p_fn0_1 used to store the polynomial of order p_n0 - 1.
     */
    double FRec(double p_x, int p_n, int p_n0, double &p_fn0, double &p_fn0_1) const;
};
/**@}*/
}

#endif /* _POLYNOMIALS1D_H */
