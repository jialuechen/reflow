
#ifndef STATETREESTOCKS_H
#define STATETREESTOCKS_H
#include<Eigen/Dense>

/** \file StateWithStocks.h
 *  \brief Permits to define a state containing
 *   - a stock position
 *   - a realisation of the non controlled stochastic state defined on a tree and identified by the index of the node in tree
 */
namespace reflow
{

/// \class StateTreeStocks StateTreeStocks.h
/// State class dealing with regime, stock and uncertainty
class StateTreeStocks
{
private :

    Eigen::ArrayXd  m_ptStock ; ///< Stock point (on the grid)
    int  m_stochasticRealisation ; ///< uncertainty realization (index in tree)
    int m_regime ; ///<regime number
public :

/// \brief Default Constructor
    StateTreeStocks() {}

/// \brief main constructor
    StateTreeStocks(const int &p_regime, const Eigen::ArrayXd &p_ptStock, const int &p_stochasticRealisation):
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

    inline const int  &getStochasticRealization() const
    {
        return m_stochasticRealisation;
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

    inline void setStochasticRealization(const int  &p_stochasticRealisation)
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
#endif /* STATETREESTOCKS_H */
