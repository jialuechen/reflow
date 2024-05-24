
#ifndef ONEDIMDATA_H
#define ONEDIMDATA_H
#include <memory>
#include <Eigen/Dense>
#include <iostream>
#include <vector>

/** \file OneDimData.h
 * \brief Defines a Data specified on a One Dimensional Grid with a constant par step value
 * \author Xavier Warin
 */
namespace reflow
{
/// \class OneDimData OneDimData.h
/// Some data being constant per mesh in one dimensional setting
template< class OneDimGrid, typename T  >
class OneDimData
{
private :
    std::shared_ptr< OneDimGrid> m_grid ; ///< One dimensional grid
    std::shared_ptr<std::vector<T> > m_values ; ///< values associated to the grid

public :
    ///\brief Constructor
    /// \param p_grid  One dimensional grid
    /// \param p_values values on the grid
    OneDimData(const std::shared_ptr< OneDimGrid>   &p_grid, const std::shared_ptr<std::vector<T> > &p_values): m_grid(p_grid), m_values(p_values)
    {
        assert(p_values->size() ==  p_grid->getNbStep() + 1);
    }
    /// \brief get the value interpolated constant per mesh at a given point
    /// \param p_coord   the abscissa
    /// \return interpolated value
    inline T  get(const double &p_coord) const
    {
        return (*m_values)[m_grid->getMesh(p_coord)];
    }
    /// \brief get the average value
    inline T mean(const double &p_deb, const double &p_last) const
    {
        int ideb = m_grid->getMesh(p_deb);
        int ilast = m_grid->getMesh(p_last)   ;
        if (ideb == ilast)
            return (*m_values)[ideb];
        double sumOfSteps = 0;
        T ponderValues = 0.;
        double firstStep = m_grid->getValue(ideb + 1) - p_deb;
        sumOfSteps += firstStep;
        ponderValues += firstStep * (*m_values)[ideb];
        for (int i = ideb + 1; i < ilast; ++i)
        {
            double step =  m_grid->getValue(i + 1) - m_grid->getValue(i);
            sumOfSteps += step;
            ponderValues += step * (*m_values)[i];
        }
        double lastStep = p_last - m_grid->getValue(ilast);
        sumOfSteps += lastStep;
        ponderValues += lastStep * (*m_values)[ilast];
        return ponderValues / sumOfSteps;
    }
};
}
#endif
