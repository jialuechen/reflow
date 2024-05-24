
#ifndef STATEWITHSTOCKS_H
#define STATEWITHSTOCKS_H
#include<Eigen/Dense>

namespace reflow
{

/// \class StateWithStocks StateWithStocks.h
/// State class dealing with regime, stock and uncertainty
class StateWithStocks
{
private :

    Eigen::ArrayXd  m_ptStock ; ///< Stock point (on the grid)
    Eigen::ArrayXd  m_stochasticRealisation ; ///< uncertainty realization
    int m_regime ; ///<regime number
public :

/// \brief Default Constructor
    StateWithStocks() {}

/// \brief main constructor
    StateWithStocks(const int &p_regime, const Eigen::ArrayXd &p_ptStock, const Eigen::ArrayXd &p_stochasticRealisation):
        m_ptStock(p_ptStock), m_stochasticRealisation(p_stochasticRealisation), m_regime(p_regime) {}

/// \brief accessor
///@{
    inline const Eigen::ArrayXd &getPtStock() const
    {
        return m_ptStock;
    }
    inline double getPtOneStock(const int &p_iStock) const
    {
        return m_ptStock(p_iStock);
    }
    inline int  getStockSize() const
    {
        return  m_ptStock.size();
    }

    inline const Eigen::ArrayXd &getStochasticRealization() const
    {
        return m_stochasticRealisation;
    }
    inline int getStochasticRealizationSize() const
    {
        return m_stochasticRealisation.size();
    }
    inline int  getRegime() const
    {
        return m_regime;
    }
    inline void setPtStock(const Eigen::ArrayXd &p_ptStock)
    {
        m_ptStock = p_ptStock;
    }
    inline void setPtOneStock(const int &p_istock, const double &p_ptStockValue)
    {
        m_ptStock(p_istock) = p_ptStockValue;
    }
    inline void  addToOneStock(const int &p_istock, const double &p_ptStockValue)
    {
        m_ptStock(p_istock) += p_ptStockValue;
    }

    inline void setStochasticRealization(const Eigen::ArrayXd &p_stochasticRealisation)
    {
        m_stochasticRealisation = p_stochasticRealisation;
    }
    inline void setRegime(const int &p_regime)
    {
        m_regime = p_regime;
    }
///@}

};
}
#endif /* STATEWITHSTOCKS_H */
