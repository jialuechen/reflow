// Copyright (C) 2021 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#ifndef STATEWITHINTSTATE_H
#define STATEWITHINTSTATE_H
#include<Eigen/Dense>

/** \file StateWithIntState.h
 *  \brief Permits to define a state containing
 *   - an  integer deterministic state
 *   - a realisation of the non controlled stochastic state
 */
namespace libflow
{

/// \class StateWithIntState StateWithIntState.h
/// State class dealing with regime, stock and uncertainty
class StateWithIntState
{
private :

    Eigen::ArrayXi  m_ptIntState ; ///< State point (on the grid)
    Eigen::ArrayXd  m_stochasticRealisation ; ///< uncertainty realization
    int m_regime ; ///<regime number
public :

/// \brief Default Constructor
    StateWithIntState() {}

/// \brief main constructor
    StateWithIntState(const int &p_regime, const Eigen::ArrayXi &p_ptIntState, const Eigen::ArrayXd &p_stochasticRealisation):
        m_ptIntState(p_ptIntState), m_stochasticRealisation(p_stochasticRealisation), m_regime(p_regime) {}

/// \brief accessor
///@{
    inline const Eigen::ArrayXi &getPtState() const
    {
        return m_ptIntState;
    }
    inline int  getPtOneState(const int &p_iState) const
    {
        return m_ptIntState(p_iState);
    }
    inline int  getStateSize() const
    {
        return  m_ptIntState.size();
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
    inline void setPtState(const Eigen::ArrayXi &p_ptIntState)
    {
        m_ptIntState = p_ptIntState;
    }
    inline void setPtOneState(const int &p_istate, const double &p_ptIntStateValue)
    {
        m_ptIntState(p_istate) = p_ptIntStateValue;
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
#endif /* STATEWITHINTSTATES_H */
