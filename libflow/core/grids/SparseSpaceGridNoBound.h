
#ifndef SPARSESPACEGRIDNOBOUND_H
#define SPARSESPACEGRIDNOBOUND_H
#include <iosfwd>
#include <iostream>
#include <boost/lexical_cast.hpp>
#include <Eigen/Dense>
#include "libflow/core/grids/SpaceGrid.h"
#include "libflow/core/grids/SparseSpaceGrid.h"
#include "libflow/core/grids/SparseGridNoBoundIterator.h"
#include "libflow/core/grids/SparseNoBoundInterpolator.h"
#include "libflow/core/sparse/sparseGridTypes.h"
#include "libflow/core/sparse/sparseGridUtils.h"
#include "libflow/core/sparse/sparseGridCommon.h"
#include "libflow/core/sparse/sparseGridNoBound.h"
#include "libflow/core/sparse/SparseGridLinNoBound.h"
#include "libflow/core/sparse/SparseGridQuadNoBound.h"
#include "libflow/core/sparse/SparseGridCubicNoBound.h"
#include "libflow/core/sparse/SparseGridHierarOnePointLinNoBound.h"
#include "libflow/core/sparse/SparseGridHierarOnePointQuadNoBound.h"
#include "libflow/core/sparse/SparseGridHierarOnePointCubicNoBound.h"


/** \file SparseSpaceGridNoBound.h
 *  \brief Defines a \f$n\f$ dimensional grid with equal space step
 *  \author Xavier Warin
 */
namespace libflow
{

/// \class SparseSpaceGridNoBound SparseSpaceGridNoBound.h
/// Defines Sparse grids with boundary points
class SparseSpaceGridNoBound : public SparseSpaceGrid
{
public :

    /// \brief Default constructor
    SparseSpaceGridNoBound(): SparseSpaceGrid() {}


    /// \brief Constructor
    /// \param p_lowValues   coordinates of bottom, left etc.. point of the domain
    /// \param p_sizeDomain  domain size in each dimension  such that the points lie in \f$ [ lowValues[0], lowValues[0] + sizeDomain[0]] \times ... \times  [ lowValues[NDIM], lowValues[NDIM] + sizeDomain[0]]\f$
    /// \param p_levelMax    maximum level of the sparse grid
    /// \param p_weight      weight for the anisotropy : the level \f$ (l_i)_i\f$  satisfy  \sum_i weight[i] l_i \le NDIM + levelMax -1 \f$
    /// \param p_degree      degree of the interpolation for the grid
    SparseSpaceGridNoBound(const Eigen::ArrayXd   &p_lowValues, const Eigen::ArrayXd &p_sizeDomain, const int &p_levelMax,  const Eigen::ArrayXd &p_weight,
                           const size_t &p_degree) : SparseSpaceGrid(p_lowValues, p_sizeDomain, p_levelMax, p_weight, p_degree)

    {
        initialSparseConstructionNoBound(p_levelMax, p_weight, *m_dataSet, m_nbPoints);
        m_iBase = sonEvaluationNoBound(*m_dataSet,  p_weight.size(), m_nbPoints, *m_son);
    }

    /// \brief Second constructor on \f$ [0,1]^{NDIM} \f$
    /// \param p_levelMax    maximum level of the sparse grid
    /// \param p_weight      weight for the anisotropy : the level \f$ (l_i)_i\f$  satisfy  \f$ \sum_i weight[i] l_i \le NDIM + levelMax -1 \f$
    /// \param p_degree      degree of the interpolation for the grid
    SparseSpaceGridNoBound(const int &p_levelMax,  const Eigen::ArrayXd &p_weight,
                           const size_t &p_degree) : SparseSpaceGrid(p_levelMax, p_weight, p_degree)

    {
        initialSparseConstructionNoBound(m_levelMax, m_weight, *m_dataSet, m_nbPoints);
        m_iBase = sonEvaluationNoBound(*m_dataSet,  p_weight.size(), m_nbPoints, *m_son);
    }


    /// \brief Constructor  after deserializating
    /// \param p_lowValues   coordinates of bottom, left etc.. point of the domain
    /// \param p_sizeDomain  domain size in each dimension  such that the points lie in \f$ [ lowValues[0], lowValues[0] + sizeDomain[0]] \times ... \times  [ lowValues[NDIM], lowValues[NDIM] + sizeDomain[0]] \f$
    /// \param p_levelMax    maximum level of the sparse grid
    /// \param p_weight      weight for the anisotropy : the level \f$ (l_i)_i\f$  satisfy  \f$ \sum_i weight[i] l_i \le NDIM + levelMax -1 \f$
    /// \param p_dataSet     data structure
    /// \param p_nbPoints    number of points in data structure
    /// \param p_degree      degree of the interpolation for the grid
    /// \param p_son         Store sons in data structure
    /// \param p_iBase        number of the node associated to the base of the sparse grid
    SparseSpaceGridNoBound(const Eigen::ArrayXd   &p_lowValues, const Eigen::ArrayXd &p_sizeDomain,  const int &p_levelMax,   const Eigen::ArrayXd &p_weight,
                           const std::shared_ptr< SparseSet> &p_dataSet, const size_t &p_nbPoints, const size_t &p_degree, const std::shared_ptr< Eigen::Array< std::array<int, 2 >, Eigen::Dynamic, Eigen::Dynamic > >   &p_son,
                           const int &p_iBase):
        SparseSpaceGrid(p_lowValues, p_sizeDomain, p_levelMax, p_weight, p_dataSet, p_nbPoints, p_degree, p_son, p_iBase)
    {}

    /// \brief Recalculate son
    void recalculateSon()
    {
        m_iBase = sonEvaluationNoBound(*m_dataSet,  m_weight.size(), m_nbPoints, *m_son);
    }

    /// \brief get back iterator associated to the grid (multi thread)
    /// \param   p_iThread  Thread number  (for multi thread purpose)
    std::shared_ptr< GridIterator> getGridIteratorInc(const int &p_iThread) const
    {
        return std::make_shared<SparseGridNoBoundIterator>(m_dataSet, m_lowValues, m_sizeDomain, p_iThread);
    }
    /// \brief get back iterator associated to the grid
    std::shared_ptr< GridIterator> getGridIterator() const
    {
        return std::make_shared< SparseGridNoBoundIterator>(m_dataSet, m_lowValues, m_sizeDomain) ;
    }

    /// \brief Get back a grid iterator on a given level of the grid
    /// \param p_iterLevel  iterator on a multi level in the sparse grid
    std::shared_ptr< SparseGridIterator> getLevelGridIterator(const  SparseSet::const_iterator &p_iterLevel) const
    {
        return std::make_shared< SparseGridNoBoundIterator>(m_dataSet,  p_iterLevel, m_lowValues, m_sizeDomain) ;
    }
    /// \brief Get back a grid iterator on a given level of the grid
    /// \param p_iterLevel  iterator on a multi level in the sparse grid
    /// \param   p_iThread  Thread number  (for multi thread purpose)
    std::shared_ptr< SparseGridIterator> getLevelGridIteratorInc(const  SparseSet::const_iterator &p_iterLevel, const int &p_iThread) const
    {
        return std::make_shared< SparseGridNoBoundIterator>(m_dataSet,  p_iterLevel, m_lowValues, m_sizeDomain, p_iThread) ;
    }

    /// \brief Hierarchize a function defined on the grid
    /// \param p_toHierachize   function to hierarchize
    void toHierarchize(Eigen::ArrayXd &p_toHierachize) const
    {
        switch (m_degree)
        {
        case 1 :
            ExplorationNoBound<  Hierar1DLinNoBound, double, Eigen::ArrayXd  >(*m_dataSet, m_weight.size(), p_toHierachize);
            break;
        case 2 :
            ExplorationNoBound<  Hierar1DQuadNoBound, double, Eigen::ArrayXd  >(*m_dataSet, m_weight.size(), p_toHierachize);
            break;
        case 3 :
            ExplorationNoBound<  Hierar1DCubicNoBound, double, Eigen::ArrayXd  >(*m_dataSet, m_weight.size(), p_toHierachize);
            break;
        default :
            std::cout << "degree not provided ";
            abort();

        }
    }

    /// \brief Hierarchize a function defined on the grid : duplicate the data for python mapping for example
    /// \param p_toHierachize   function to hierarchize
    Eigen::ArrayXd  toHierarchizeD(const Eigen::ArrayXd &p_toHierachize) const
    {
        Eigen::ArrayXd toHierachize(p_toHierachize);
        toHierarchize(toHierachize);
        return toHierachize;
    }

    /// \brief Hierarchize some points defined on the sparse grids
    ///        Hierarchization is performed point by point
    /// \param p_nodalValues         function to hierarchize
    /// \param p_sparsePoints        vector of sparse points to hierarchize (all points should belong to the dataset structure)
    /// \param p_hierarchized        array of all hierarchized values (it is updated)
    void  toHierarchizePByP(const Eigen::ArrayXd &p_nodalValues, const  std::vector<SparsePoint>  &p_sparsePoints, Eigen::ArrayXd &p_hierarchized) const
    {

        int ip ;
        switch (m_degree)
        {
        case 1 :
#ifdef _OPENMP
            #pragma omp parallel for  private(ip)
#endif
            for (ip  = 0; ip < static_cast<int>(p_sparsePoints.size()); ++ip)
                p_hierarchized(std::get<2>(p_sparsePoints[ip])) = SparseGridHierarOnePointLinNoBound<double, Eigen::ArrayXd>()(std::get<0>(p_sparsePoints[ip]), std::get<1>(p_sparsePoints[ip]), *m_dataSet, p_nodalValues);
            break;
        case 2 :
#ifdef _OPENMP
            #pragma omp parallel for  private(ip)
#endif
            for (ip  = 0; ip < static_cast<int>(p_sparsePoints.size()); ++ip)
                p_hierarchized(std::get<2>(p_sparsePoints[ip])) = SparseGridHierarOnePointQuadNoBound<double, Eigen::ArrayXd>()(std::get<0>(p_sparsePoints[ip]), std::get<1>(p_sparsePoints[ip]), *m_dataSet, p_nodalValues);
            break;
        case 3 :
#ifdef _OPENMP
            #pragma omp parallel for  private(ip)
#endif
            for (ip  = 0; ip < static_cast<int>(p_sparsePoints.size()); ++ip)
                p_hierarchized(std::get<2>(p_sparsePoints[ip])) = SparseGridHierarOnePointCubicNoBound<double, Eigen::ArrayXd>()(std::get<0>(p_sparsePoints[ip]), std::get<1>(p_sparsePoints[ip]), *m_dataSet, p_nodalValues);
            break;
        default :
            std::cout << "degree not provided ";
            abort();
        }
    }

    /// \brief Hierarchize all points defined on a given level of the sparse grids
    ///        Hierarchization is performed point by point
    /// \param p_nodalValues         function to hierarchize
    /// \param p_iterlevel           iterator on the level of the point to hierarchize
    /// \param p_hierarchized        array of all hierarchized values (it is updated)
    void  toHierarchizePByPLevel(const Eigen::ArrayXd &p_nodalValues, const  SparseSet::const_iterator &p_iterLevel, Eigen::ArrayXd &p_hierarchized) const
    {
        switch (m_degree)
        {
        case 1 :
#ifdef _OPENMP
            #pragma omp parallel
        {
            size_t ithread = omp_get_thread_num();
            size_t nthreads = omp_get_num_threads();
            size_t cnt = 0;
#else
        {
#endif

            for (const auto &position : p_iterLevel->second)
            {
#ifdef _OPENMP
                cnt = +1;
                if ((cnt - 1) % nthreads != ithread) continue;
#endif
                p_hierarchized(position.second) = SparseGridHierarOnePointLinNoBound<double, Eigen::ArrayXd>()(p_iterLevel->first, position.first, *m_dataSet, p_nodalValues);
            }
        }
        break;
        case 2 :

#ifdef _OPENMP
            #pragma omp parallel
        {
            size_t ithread = omp_get_thread_num();
            size_t nthreads = omp_get_num_threads();
            size_t cnt = 0;
#else
        {
#endif
            for (const auto &position : p_iterLevel->second)
            {
#ifdef _OPENMP
                cnt = +1;
                if ((cnt - 1) % nthreads != ithread) continue;
#endif
                p_hierarchized(position.second) = SparseGridHierarOnePointQuadNoBound<double, Eigen::ArrayXd>()(p_iterLevel->first, position.first, *m_dataSet, p_nodalValues);
            }
        }

        break;
        case 3 :


#ifdef _OPENMP
            #pragma omp parallel
        {
            size_t ithread = omp_get_thread_num();
            size_t nthreads = omp_get_num_threads();
            size_t cnt = 0;
#else
        {
#endif
            for (const auto &position : p_iterLevel->second)
            {
#ifdef _OPENMP
                cnt = +1;
                if ((cnt - 1) % nthreads != ithread) continue;
#endif
                p_hierarchized(position.second) = SparseGridHierarOnePointCubicNoBound<double, Eigen::ArrayXd>()(p_iterLevel->first, position.first, *m_dataSet, p_nodalValues);
            }
        }

        break;
        default :
            std::cout << "degree not provided ";
            abort();
        }

    }


    /// \brief Hierarchize a set of functions defined on the grid
    /// \param p_toHierachize   function to hierarchize
    void toHierarchizeVec(Eigen::ArrayXXd &p_toHierachize) const
    {
        switch (m_degree)
        {
        case 1 :
            ExplorationNoBound<  Hierar1DLinNoBound, Eigen::ArrayXd, Eigen::ArrayXXd  >(*m_dataSet, m_weight.size(), p_toHierachize);
            break;
        case 2 :
            ExplorationNoBound<  Hierar1DQuadNoBound, Eigen::ArrayXd, Eigen::ArrayXXd  >(*m_dataSet, m_weight.size(), p_toHierachize);
            break;
        case 3 :
            ExplorationNoBound<  Hierar1DCubicNoBound, Eigen::ArrayXd, Eigen::ArrayXXd  >(*m_dataSet, m_weight.size(), p_toHierachize);
            break;
        default :
            std::cout << "degree not provided ";
            abort();

        }
    }

    /// \brief Hierarchize a set of functions defined on the grid : duplicate data for example for python mapping
    /// \param p_toHierachize   function to hierarchize
    Eigen::ArrayXXd toHierarchizeVecD(const Eigen::ArrayXXd &p_toHierachize)
    {
        Eigen::ArrayXXd toHierachize(p_toHierachize);
        toHierarchizeVec(toHierachize);
        return toHierachize;
    }

    /// \brief Hierarchize some points defined on the sparse grids for a set of functions
    ///        Hierarchization is performed point by point
    /// \param p_nodalValues         functions to hierarchize (the row corresponds to the function number)
    /// \param p_sparsePoints        vector of sparse points to hierarchize (all points should belong to the dataset structure)
    /// \param p_hierarchized        array of all hierarchized values (it is updated)
    void toHierarchizePByPVec(const Eigen::ArrayXXd &p_nodalValues, const  std::vector<SparsePoint>  &p_sparsePoints, Eigen::ArrayXXd &p_hierarchized) const
    {
        int ip ;
        switch (m_degree)
        {
        case 1 :
#ifdef _OPENMP
            #pragma omp parallel for  private(ip)
#endif
            for (ip  = 0; ip < static_cast<int>(p_sparsePoints.size()); ++ip)
                p_hierarchized.col(std::get<2>(p_sparsePoints[ip])) = SparseGridHierarOnePointLinNoBound<Eigen::ArrayXd, Eigen::ArrayXXd >()(std::get<0>(p_sparsePoints[ip]), std::get<1>(p_sparsePoints[ip]), *m_dataSet, p_nodalValues);
            break;
        case 2 :
#ifdef _OPENMP
            #pragma omp parallel for  private(ip)
#endif
            for (ip  = 0; ip < static_cast<int>(p_sparsePoints.size()); ++ip)
                p_hierarchized.col(std::get<2>(p_sparsePoints[ip])) = SparseGridHierarOnePointQuadNoBound<Eigen::ArrayXd, Eigen::ArrayXXd >()(std::get<0>(p_sparsePoints[ip]), std::get<1>(p_sparsePoints[ip]), *m_dataSet, p_nodalValues);
            break;
        case 3 :
#ifdef _OPENMP
            #pragma omp parallel for  private(ip)
#endif
            for (ip  = 0; ip < static_cast<int>(p_sparsePoints.size()); ++ip)
                p_hierarchized.col(std::get<2>(p_sparsePoints[ip])) = SparseGridHierarOnePointCubicNoBound<Eigen::ArrayXd, Eigen::ArrayXXd>()(std::get<0>(p_sparsePoints[ip]), std::get<1>(p_sparsePoints[ip]), *m_dataSet, p_nodalValues);
            break;
        default :
            std::cout << "degree not provided ";
            abort();
        }
    }


    /// \brief Hierarchize all points defined on a gives level of the sparse grids for a set of functions
    ///        Hierarchization is performed point by point
    /// \param p_nodalValues         function to hierarchize (the row corresponds to the function number)
    /// \param p_iterlevel           iterator on the level of the point to hierarchize
    /// \param p_hierarchized        array of all hierarchized values (it is updated)
    void  toHierarchizePByPLevelVec(const Eigen::ArrayXXd &p_nodalValues, const  SparseSet::const_iterator &p_iterLevel, Eigen::ArrayXXd &p_hierarchized) const
    {

        switch (m_degree)
        {
        case 1 :
#ifdef _OPENMP
            #pragma omp parallel
        {
            size_t ithread = omp_get_thread_num();
            size_t nthreads = omp_get_num_threads();
            size_t cnt = 0;
#else
        {
#endif
            for (const auto &position : p_iterLevel->second)
            {
#ifdef _OPENMP
                cnt = +1;
                if ((cnt - 1) % nthreads != ithread) continue;
#endif

                p_hierarchized.col(position.second) = SparseGridHierarOnePointLinNoBound<Eigen::ArrayXd, Eigen::ArrayXXd >()(p_iterLevel->first, position.first, *m_dataSet, p_nodalValues);
            }
        }
        break;
        case 2 :
#ifdef _OPENMP
            #pragma omp parallel
        {
            size_t ithread = omp_get_thread_num();
            size_t nthreads = omp_get_num_threads();
            size_t cnt = 0;
#else
        {
#endif
            for (const auto &position : p_iterLevel->second)
            {
#ifdef _OPENMP
                cnt = +1;
                if ((cnt - 1) % nthreads != ithread) continue;
#endif

                p_hierarchized.col(position.second) = SparseGridHierarOnePointQuadNoBound<Eigen::ArrayXd, Eigen::ArrayXXd >()(p_iterLevel->first, position.first, *m_dataSet, p_nodalValues);
            }
        }
        break;
        case 3 :

#ifdef _OPENMP
            #pragma omp parallel
        {
            size_t ithread = omp_get_thread_num();
            size_t nthreads = omp_get_num_threads();
            size_t cnt = 0;
#else
        {
#endif
            for (const auto &position : p_iterLevel->second)
            {
#ifdef _OPENMP
                cnt = +1;
                if ((cnt - 1) % nthreads != ithread) continue;
#endif
                p_hierarchized.col(position.second) = SparseGridHierarOnePointCubicNoBound< Eigen::ArrayXd, Eigen::ArrayXXd>()(p_iterLevel->first, position.first, *m_dataSet, p_nodalValues);
            }
        }
        break;
        default :
            std::cout << "degree not provided ";
            abort();
        }

    }

    /// \brief calculate and get back son
    std::shared_ptr< Eigen::Array< std::array<int, 2 >, Eigen::Dynamic, Eigen::Dynamic > > calculateSon() const
    {
        std::shared_ptr< Eigen::Array< std::array<int, 2 >, Eigen::Dynamic, Eigen::Dynamic > > son = std::make_shared< Eigen::Array< std::array<int, 2 >, Eigen::Dynamic, Eigen::Dynamic > >();
        sonEvaluationNoBound(*m_dataSet,  m_lowValues.size(), m_nbPoints, *son);
        return son ;
    }

    /// \brief  Get back interpolator at a point Interpolate on the grid : here it is a linear interpolator
    /// \param  p_coord   coordinate of the point for interpolation
    /// \return interpolator at the point coordinates on the grid
    std::shared_ptr<Interpolator> createInterpolator(const Eigen::ArrayXd &p_coord) const
    {

        // rescale
        Eigen::ArrayXd coordRescaled = (p_coord - m_lowValues) / m_sizeDomain ;

        switch (m_degree)
        {
        case 1 :
            return 	std::make_shared<SparseNoBoundInterpolator<LinearHatValue, LinearHatValue, LinearHatValue > >(m_son, m_iBase, coordRescaled) ;
        case 2 :
            return 	std::make_shared<SparseNoBoundInterpolator<QuadraticValue, QuadraticValue, QuadraticValue> >(m_son, m_iBase, coordRescaled) ;
        case 3 :
            return 	std::make_shared<SparseNoBoundInterpolator< QuadraticValue, CubicLeftValue, CubicRightValue > >(m_son, m_iBase, coordRescaled) ;
        default :
            std::cout << "degree not provided ";
            abort();
        }
    }

    /// \brief test if the point is strictly inside the domain
    /// \param true if the point is strictly inside the closed domain
    bool isStrictlyInside(const Eigen::ArrayXd &) const
    {
        return true;
    }
    /// \brief test if the point is inside the domain
    /// \param true if the point is inside the open domain
    bool isInside(const Eigen::ArrayXd &) const
    {
        return true;
    }

    /// \brief Add level to data structure
    /// \param p_fatherLevel  iterator on father level
    /// \param p_idim         refinement dimension
    /// \return a n iterator of the new add level
    SparseSet::const_iterator addLevelToDataSet(const SparseSet::const_iterator &p_fatherLevel, const int &p_idim)
    {
        SparseLevel  pointsToAdd;
        for (SparseLevel::const_iterator iterIndex =  p_fatherLevel->second.begin(); iterIndex != p_fatherLevel->second.end(); ++iterIndex)
        {
            Eigen::ArrayXui index = iterIndex->first;
            index(p_idim) *= 2 ;
            pointsToAdd[index] = m_nbPoints++;
            index(p_idim) += 1;
            pointsToAdd[index] = m_nbPoints++;
        }
        // current level to add
        Eigen::ArrayXc  levelToadd = p_fatherLevel->first;
        levelToadd(p_idim) += 1;
        // current level
        Eigen::ArrayXc  currentLevel = p_fatherLevel->first ;
        currentLevel(p_idim) += 1;
        std::pair< SparseSet::iterator, bool> ret  = m_dataSet->insert(std::pair< Eigen::ArrayXc, SparseLevel>(currentLevel, pointsToAdd));
        assert(ret.second == true);
        return ret.first;
    }
};
}

#endif /* SPARSESPACENOBOUNDGRID_H */
