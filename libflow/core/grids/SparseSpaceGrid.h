
#ifndef SPARSESPACEGRID_H
#define SPARSESPACEGRID_H
#include <iosfwd>
#include <iostream>
#include <set>
#include <map>
#include <functional>
#include <tuple>
#include <boost/lexical_cast.hpp>
#include <Eigen/Dense>
#include "libflow/core/grids/SpaceGrid.h"
#include "libflow/core/sparse/sparseGridTypes.h"
#include "libflow/core/grids/SparseGridIterator.h"

/** \file SparseSpaceGrid.h
 *  \brief Defines a \f$n\f$ dimensional grid with equal space step
 *  \author Xavier Warin
 */
namespace libflow
{

/// \class SparseSpaceGrid SparseSpaceGrid.h
/// Defines Sparse grids with boundary points
class SparseSpaceGrid : public SpaceGrid
{
private :

    /// \brief Dimension adaptation step (Gerstner Griebel)
    ///        The error is estimated as on each multi level by the \f$p\_phi\f$ function.
    ///        The multi level with the highest error between the active multi level is split according to Gerstner rule
    /// \param p_hierarValues   hierarchical values calculated on the current dataSet
    /// \param p_phi            function for the error on a given level in the m_dataSet structure
    /// \param p_phiMult         from an error defined on different levels, send back a global error on the different levels
    /// \param p_precision      precision target
    /// \param  p_error      permits to store the errors on each active level during refinement
    /// \return  The multi levels added to refine (as  iterators),  the error reached
    std::pair< std::vector<SparseSet::const_iterator>,  double>  dimensionRefineStep(const Eigen::ArrayXd &p_hierarValues,
            const std::function< double(const SparseSet::const_iterator &,
                                        const Eigen::ArrayXd &)> &p_phi,
            const std::function< double(const std::vector< double> &) > &p_phiMult,
            const double &p_precision,
            std::map<  SparseSet::const_iterator, double, OrderLevel > &p_error) ;


    /// \brief Realize one step of coarsening the grid (data structure modified, active levels , old levels modified)
    /// \param levelPotenRm a list of potential levels to erase in the data structure
    /// \return the new active level calculated after a step (if any)
    SparseSet::const_iterator  dimensionCoarsenStep(std::map< SparseSet::const_iterator, double, OrderLevel >   &levelPotenRm);


    /// \brief Dimension adaptation initialization
    /// This initialization is only achieved before adaptation
    /// Active index are set according to Gerstner rule
    void  dimensionAdaptiveInit();


protected :

    ///\brief  minimal value of the mesh in each direction
    Eigen::ArrayXd  m_lowValues ;

    /// \brief Size of the resolution domain in each dimension
    Eigen::ArrayXd  m_sizeDomain;

    /// \brief weights associated to anisotropic sparse grids
    Eigen::ArrayXd m_weight;

    /// \brief level of the sparse grid
    int  m_levelMax;

    /// \brief Point number associated to the sparse grid
    size_t m_nbPoints;

    /// \brief Sparse grid structure
    std::shared_ptr<SparseSet>  m_dataSet;

    /// \brief degree of the interpolation
    size_t m_degree; ///< 1 is linear, 2 quadratic, 3 cubic
    std::shared_ptr< Eigen::Array< std::array<int, 2 >, Eigen::Dynamic, Eigen::Dynamic > > m_son ; ///<  Store the sons of all points
    int m_iBase ; ///< number of the node associated to the base

    /// \brief Dimension adaptive members
    ///@{
    std::set< SparseSet::const_iterator, OrderLevel > m_activeLevel;  /// \brief store active level that can be refined
    std::set< SparseSet::const_iterator, OrderLevel > m_oldLevel;  /// \brief old levels not to be refined
    ///@}

    /// \brief Add level to data structure
    /// \param p_fatherLevel  iterator on father level
    /// \param p_idim         refinement dimension
    /// \return an iterator on the new added level
    virtual  SparseSet::const_iterator addLevelToDataSet(const SparseSet::const_iterator &p_fatherLevel, const int &p_idim) = 0 ;

    /// \brief After coarsening,  modify data set to remove "holes" and hierarchical values to fit new data structure
    /// \param p_hierarValues Hierachical values to modify
    /// \param p_valuesFunction  an array storing the nodal values (modified on the new struture)
    /// \return number of points in new data structure
    size_t modifyHierarAndDataSetAfterCoarsen(Eigen::ArrayXd   &p_hierarValues, Eigen::ArrayXd   &p_valuesFunction);

    /// \brief After coarsening,  modify data set to remove "holes" and hierarchical values to fit new data structure
    /// \param p_hierarValues Hierachical values to modify
    /// \param p_valuesFunction  an array storing the nodal values (modified on the new struture)
    /// \return number of points in new data structure
    size_t modifyHierarAndDataSetAfterCoarsenVec(Eigen::ArrayXXd   &p_hierarValues, Eigen::ArrayXXd &p_valuesFunction);

public :

    /// \brief Default constructor
    SparseSpaceGrid();


    /// \brief Constructor on \f$ [0,1]^{NDIM} \f$
    /// \param p_levelMax    maximum level of the sparse grid
    /// \param p_weight      weight for the anisotropy : the level \f$ (l_i)_i\f$  satisfy \f$ \sum_i weight[i] l_i \le NDIM + levelMax -1 \f$
    /// \param p_degree      degree of the interpolation for the grid
    SparseSpaceGrid(const int &p_levelMax,  const Eigen::ArrayXd &p_weight, const size_t &p_degree);

    /// \brief Constructor
    /// \param p_lowValues   coordinates of bottom, left etc.. point of the domain
    /// \param p_sizeDomain  domain size in each dimension  such that the points lie in \f$ [ lowValues[0], lowValues[0] + sizeDomain[0]] \times ... \times  [ lowValues[NDIM], lowValues[NDIM] + sizeDomain[0]] \f$
    /// \param p_levelMax    maximum level of the sparse grid
    /// \param p_weight      weight for the anisotropy : the level \f$ (l_i)_i\f$  satisfy   \f$ \sum_i weight[i] l_i \le NDIM + levelMax -1 \f$
    /// \param p_degree      degree of the interpolation for the grid
    SparseSpaceGrid(const Eigen::ArrayXd   &p_lowValues, const Eigen::ArrayXd &p_sizeDomain, const int &p_levelMax,  const Eigen::ArrayXd &p_weight,
                    const size_t &p_degree);

    /// \brief Constructor  after deserialization
    /// \param p_lowValues   coordinates of bottom, left etc.. point of the domain
    /// \param p_sizeDomain  domain size in each dimension  such that the points lie in \f$ [ lowValues[0], lowValues[0] + sizeDomain[0]] \times ... \times  [ lowValues[NDIM], lowValues[NDIM] + sizeDomain[0]] \f$
    /// \param p_levelMax    maximum level of the sparse grid
    /// \param p_weight      weight for the anisotropy : the level \f$ (l_i)_i\f$  satisfy  \f$ \sum_i weight[i] l_i \le NDIM + levelMax -1 \f$
    /// \param p_dataSet     data structure
    /// \param p_nbPoints    number of points in data structure
    /// \param p_degree      degree of the interpolation for the grid
    /// \param p_son         Store sons in data structure
    /// \param p_iBase        number of the node associated to the base of the sparse grid
    SparseSpaceGrid(const Eigen::ArrayXd   &p_lowValues, const Eigen::ArrayXd &p_sizeDomain,  const int &p_levelMax,   const Eigen::ArrayXd &p_weight,
                    const std::shared_ptr< SparseSet> &p_dataSet, const size_t &p_nbPoints, const size_t &p_degree, const std::shared_ptr< Eigen::Array< std::array<int, 2 >, Eigen::Dynamic, Eigen::Dynamic > >   &p_son, const int &p_iBase);


    ///  dimension of the grid
    inline int getDimension() const
    {
        return    m_lowValues.size();
    }

    /// \brief Dimension adaptation nest
    /// \param p_precision       precision required for adaptation
    /// \param p_fInterpol      function to interpolate
    /// \param p_phi            function for the error on a given level in the m_dataSet structure
    /// \param p_phiMult         from an error defined on different levels, send back a global error on the different levels
    /// \param p_valuesFunction  an array storing the nodal values
    /// \param  p_hierarValues  an array storing hierarchized values (updated)
    void refine(const double &p_precision, const std::function<double(const Eigen::ArrayXd &p_x)> &p_fInterpol,
                const std::function< double(const SparseSet::const_iterator &, const Eigen::ArrayXd &)> &p_phi,
                const std::function< double(const std::vector< double> &) > &p_phiMult,
                Eigen::ArrayXd &p_valuesFunction,
                Eigen::ArrayXd &p_hierarValues);


    /// \brief Dimension adaptation coarsening : modify data struture by trying to remove all levels with local error
    ///        below a local precision
    /// \param p_precision      Precision under which coarsening  will be realized
    /// \param p_phi            function for the error on a given level in the m_dataSet structure
    /// \param p_valuesFunction  an array storing the nodal values (modified on the new struture)
    /// \param p_hierarValues   Hierarchical values on a data structure (modified on the new structure)
    void coarsen(const double &p_precision,  const std::function< double(const SparseSet::const_iterator &, const Eigen::ArrayXd &)> &p_phi,
                 Eigen::ArrayXd &p_valuesFunction,
                 Eigen::ArrayXd   &p_hierarValues);

    /// \brief Accessor
    ///@{

    inline const Eigen::ArrayXd &getLowValues() const
    {
        return  m_lowValues ;
    }
    inline size_t getNbPoints() const
    {
        return m_nbPoints;
    }
    inline int getLevelMax() const
    {
        return m_levelMax;
    }

    inline  const Eigen::ArrayXd &getSizeDomain() const
    {
        return m_sizeDomain;
    }

    inline  const Eigen::ArrayXd &getWeight() const
    {
        return m_weight;
    }
    inline std::shared_ptr<SparseSet> getDataSet() const
    {
        return m_dataSet;
    }
    inline int getDataSetDepth() const
    {
        return m_dataSet->size();
    }

    inline std::shared_ptr< Eigen::Array< std::array<int, 2 >, Eigen::Dynamic, Eigen::Dynamic > >  getSon() const
    {
        return m_son ;
    }
    inline size_t  getDegree() const
    {
        return m_degree;
    }

    inline int getIBase() const
    {
        return m_iBase;
    }

    ///@}


    /// \brief  To print object for debug (don't use operator << due to geners)
    void print() const;

    /// \brief get back bounds  associated to the grid
    /// \return to the grid in each dimension give the extreme values (min, max)
    std::vector <std::array< double, 2>  > getExtremeValues() const;

    /// \brief Sparse grids should provide Hierarchization procedure
    /// \param p_toHierachize  array of values at nodal points : the values corresponding to a function values are hierarchized
    virtual void toHierarchize(Eigen::ArrayXd &p_toHierachize) const = 0 ;
    /// \param p_toHierachize  array of values at nodal points : the values (each row of _toHierachize corresponds to a function value) are hierarchized.
    virtual void toHierarchizeVec(Eigen::ArrayXXd &p_toHierachize) const = 0 ;
    /// \brief Hierarchize some points defined on the sparse grids
    ///        Hierarchization is performed point by point
    /// \param p_nodalValues         function to hierarchize
    /// \param p_sparsePoints        vector of sparse points to hierarchize (all points should belong to the dataset structure)
    /// \param p_hierarchized        array of all hierarchized values (it is updated)
    virtual  void toHierarchizePByP(const Eigen::ArrayXd &p_nodalValues, const  std::vector<SparsePoint>  &p_sparsePoints, Eigen::ArrayXd &p_hierarchized) const = 0;
    /// \brief Hierarchize some points defined on the sparse grids for a set of functions
    ///        Hierarchization is performed point by point
    /// \param p_nodalValues         functions to hierarchize (the row corresponds to the function number)
    /// \param p_sparsePoints        vector of sparse points to hierarchize (all points should belong to the dataset structure)
    /// \param p_hierarchized        array of all hierarchized values (it is updated)
    virtual void toHierarchizePByPVec(const Eigen::ArrayXXd &p_nodalValues, const  std::vector<SparsePoint>  &p_sparsePoints, Eigen::ArrayXXd &p_hierarchized) const = 0;
    /// \brief Hierarchize all points defined on a given level of the sparse grids
    ///        Hierarchization is performed point by point
    /// \param p_nodalValues         function to hierarchize
    /// \param p_iterLevel           iterator on the level of the point to hierarchize
    /// \param p_hierarchized        array of all hierarchized values (it is updated)
    virtual  void toHierarchizePByPLevel(const Eigen::ArrayXd &p_nodalValues, const  SparseSet::const_iterator &p_iterLevel, Eigen::ArrayXd &p_hierarchized) const = 0;
    /// \brief Hierarchize all points defined on a given level of the sparse grids for a set of functions
    ///        Hierarchization is performed point by point
    /// \param p_nodalValues         function to hierarchize (the row corresponds to the function number)
    /// \param p_iterLevel           iterator on the level of the point to hierarchize
    /// \param p_hierarchized        array of all hierarchized values (it is updated)
    virtual void toHierarchizePByPLevelVec(const Eigen::ArrayXXd &p_nodalValues, const  SparseSet::const_iterator &p_iterLevel, Eigen::ArrayXXd &p_hierarchized) const = 0;

    /// \brief  Get back interpolator for a whole function
    /// \param  p_coord  coordinate of the point
    /// \return interpolator at the point coordinates  on the grid
    std::shared_ptr<InterpolatorSpectral> createInterpolatorSpectral(const Eigen::ArrayXd &p_coord) const ;


    /// \brief Get back a grid iterator on a given level of the grid
    /// \param p_iterLevel  iterator on a multi level in the sparse grid
    virtual std::shared_ptr< SparseGridIterator> getLevelGridIterator(const  SparseSet::const_iterator &p_iterLevel) const = 0;

    /// \brief Get back a grid iterator on a given level of the grid
    /// \param p_iterLevel  iterator on a multi level in the sparse grid
    /// \param   p_iThread  Thread number  (for multi thread purpose)
    virtual std::shared_ptr< SparseGridIterator> getLevelGridIteratorInc(const  SparseSet::const_iterator &p_iterLevel, const int &p_iThread) const = 0;

    /// \brief truncate a point that it stays inside the domain
    /// \param p_point  point to truncate
    void truncatePoint(Eigen::ArrayXd &p_point) const;

    /// \brief Recalculate son
    virtual void recalculateSon() = 0;

};
}

#endif /* SPARSESPACEGRID_H */
