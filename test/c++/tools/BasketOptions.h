
#ifndef BASKETOPTIONS_H
#define BASKETOPTIONS_H
#include <Eigen/Dense>

/** \file BasketOptions.h
 * \brief class for basket option payoff
 * \author Xavier Warin
 */

/// \class  BasketCall BasketOptions.h
/// call payoff for basket
class BasketCall
{
    double m_strike;
public:
    /// \brief Constructor
    /// \param p_strike   strike
    BasketCall(const double &p_strike): m_strike(p_strike) {}

    /// \brief for one simulation, give the basket value
    /// \param  p_assets   For the current simulation values of all assets
    inline double apply(const Eigen::VectorXd &p_assets) const
    {
        return std::max(p_assets.mean() - m_strike, 0.);
    }
    /// \brief  for all simulations, give the basket values
    /// \param  p_assets    values of all assets (one column is a simulation)
    Eigen::VectorXd applyVec(const Eigen::MatrixXd &p_assets) const
    {
        Eigen::VectorXd ret(p_assets.cols());
        for (int is = 0 ; is < p_assets.cols(); ++is)
        {
            ret(is) = std::max(p_assets.col(is).mean() - m_strike, 0.);
        }
        return ret;
    }
    /// \brief derivative (first dimension => asset number, second dimension => simulation number)
    Eigen::MatrixXd  getDerivativeVec(const Eigen::MatrixXd &p_assets) const
    {
        Eigen::MatrixXd ret(p_assets.rows(), p_assets.cols());
        for (int is = 0 ; is < p_assets.cols(); ++is)
        {
            if (p_assets.col(is).mean() > m_strike)
            {
                ret.col(is).setConstant(1. / p_assets.rows());
            }
            else
            {
                ret.col(is).setConstant(0.);
            }
        }
        return ret;
    }
};

/// \class  BasketPut BasketOptions.h
/// Put payoff for basket
class BasketPut
{
    double m_strike;
public:

    /// \brief Constructor
    /// \param p_strike   strike
    BasketPut(const double &p_strike): m_strike(p_strike) {}

    /// \brief  for all simulations, give the basket values
    /// \param  p_assets    values of all assets (one column is a simulation)
    Eigen::VectorXd operator()(const Eigen::MatrixXd &p_assets) const
    {
        Eigen::VectorXd ret(p_assets.cols());
        for (int is = 0 ; is < p_assets.cols(); ++is)
        {
            ret(is) = std::max(m_strike - p_assets.col(is).mean(), 0.);
        }
        return ret;
    }
};

#endif /* BASKETOPTIONS_H */
