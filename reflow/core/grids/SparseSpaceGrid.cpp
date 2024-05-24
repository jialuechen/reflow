
#include <memory>
#include "reflow/core/utils/constant.h"
#include "reflow/core/grids/SparseSpaceGrid.h"
#include "reflow/core/grids/SparseInterpolatorSpectral.h"

using namespace reflow;
using namespace Eigen;
using namespace std;

SparseSpaceGrid::SparseSpaceGrid(): m_nbPoints(0) {}


SparseSpaceGrid::SparseSpaceGrid(const int &p_levelMax,  const ArrayXd &p_weight, const size_t &p_degree) :
    m_lowValues(ArrayXd::Zero(p_weight.size())), m_sizeDomain(ArrayXd::Constant(p_weight.size(), 1.)), m_weight(p_weight), m_levelMax(p_levelMax), m_nbPoints(0), m_dataSet(make_shared<SparseSet>()),
    m_degree(p_degree), m_son(make_shared< Array< array<int, 2 >, Dynamic, Dynamic > >())

{}

SparseSpaceGrid::SparseSpaceGrid(const ArrayXd   &p_lowValues, const ArrayXd &p_sizeDomain, const int &p_levelMax,  const ArrayXd &p_weight,
                                 const size_t &p_degree) : m_lowValues(p_lowValues), m_sizeDomain(p_sizeDomain), m_weight(p_weight), m_levelMax(p_levelMax), m_nbPoints(0), m_dataSet(make_shared< SparseSet>()),
    m_degree(p_degree), m_son(make_shared<Array< array<int, 2 >, Dynamic, Dynamic > >())

{}

SparseSpaceGrid::SparseSpaceGrid(const ArrayXd   &p_lowValues, const ArrayXd &p_sizeDomain,  const int &p_levelMax,   const ArrayXd &p_weight,
                                 const shared_ptr< SparseSet> &p_dataSet, const size_t &p_nbPoints, const size_t &p_degree, const shared_ptr< Array< array<int, 2 >, Dynamic, Dynamic > >   &p_son, const int &p_iBase) :  m_lowValues(p_lowValues),
    m_sizeDomain(p_sizeDomain), m_weight(p_weight), m_levelMax(p_levelMax), m_nbPoints(p_nbPoints), m_dataSet(p_dataSet), m_degree(p_degree), m_son(p_son), m_iBase(p_iBase)
{}


/// \brief  To print object for debug (don't use operator << due to geners)
void SparseSpaceGrid::print() const
{
    for (const auto &level : *m_dataSet)
    {
        cout << "LEVEL ";
        for (int i = 0; i < m_lowValues.size(); ++i)
            cout << static_cast<int>(level.first[i]) << " " ;
        cout << endl ;
        for (const auto &position : level.second)
        {
            cout << " Point position " ;
            for (int i = 0; i < m_lowValues.size(); ++i)
                cout << position.first[i] << " ";
            cout <<  "Number " << position.second << endl ;
        }
    }
}

/// \brief get back bounds  associated to the grid
/// \return to the grid in each dimension give the extreme values (min, max)
vector <array< double, 2>  > SparseSpaceGrid::getExtremeValues() const
{
    vector<  array< double, 2> > retGrid(m_lowValues.size());
    for (int i = 0; i <  m_lowValues.size(); ++i)
    {
        retGrid[i][0] = m_lowValues(i);
        retGrid[i][1] = m_lowValues(i) + m_sizeDomain(i);
    }
    return retGrid;
}

shared_ptr<InterpolatorSpectral> SparseSpaceGrid::createInterpolatorSpectral(const ArrayXd &p_values) const
{
    return make_shared<SparseInterpolatorSpectral>(this, p_values);
}

void SparseSpaceGrid::truncatePoint(ArrayXd &p_point) const
{
    for (int id = 0 ; id < p_point.size(); ++id)
        p_point(id) = max(m_lowValues(id), min(m_lowValues(id) + m_sizeDomain(id), p_point(id)));
}

void  SparseSpaceGrid::dimensionAdaptiveInit()
{
    // clear
    m_activeLevel.clear();
    m_oldLevel.clear();
    int ndim = m_dataSet->begin()->first.size();
    ArrayXc level(ndim);
    for (SparseSet::const_iterator iterLevel = m_dataSet->begin(); iterLevel != m_dataSet->end(); ++iterLevel)
    {
        bool bFind = false;
        for (int id = 0; id < ndim; ++id)
        {
            level = iterLevel->first;
            level(id) += 1;
            if (m_dataSet->find(level) != m_dataSet->end())
            {
                bFind = true;
                break;
            }
        }
        if (bFind)
        {
            m_oldLevel.insert(iterLevel);
        }
        else
        {
            m_activeLevel.insert(iterLevel);
        }
    }
}

pair< vector<SparseSet::const_iterator>, double>   SparseSpaceGrid::dimensionRefineStep(const ArrayXd &p_hierarValues,
        const function< double(const SparseSet::const_iterator &, const ArrayXd &)> &p_phi,
        const function< double(const vector< double> &)> &p_phiMult,
        const double &p_precision,
        std::map<  SparseSet::const_iterator, double, OrderLevel > &p_error)
{
    // find the active level with the highest error
    SparseSet::const_iterator iterErrorMax ;
    double errorLocMax = 0. ;
    vector<double> vecError;
    vecError.reserve(m_activeLevel.size());
    for (const auto &level : m_activeLevel)
    {
        auto  iterError = p_error.find(level);
        double errLoc = 0 ;
        if (iterError == p_error.end())
        {
            errLoc = p_phi(level, p_hierarValues);
            p_error[level] = errLoc;
        }
        else
        {
            errLoc = iterError->second;
        }
        vecError.push_back(errLoc);
        if (errLoc > errorLocMax)
        {
            errorLocMax = errLoc;
            iterErrorMax = level;
        }
    }
    // calculate error
    double error = p_phiMult(vecError);
    if (error < p_precision)
        return make_pair(vector<SparseSet::const_iterator>(), error);
    // go on if precision not reached
    // update Active ,old levels and error
    m_oldLevel.insert(iterErrorMax);
    m_activeLevel.erase(iterErrorMax);
    p_error.erase(iterErrorMax);
    // vector of iterator to return
    vector<SparseSet::const_iterator> vecRet;
    // add all other levels
    ArrayXc level = iterErrorMax->first;
    vecRet.reserve(level.size());
    for (int  id = 0; id <  level.size(); ++id)
    {
        level(id) += 1;
        bool bAdd = true;
        for (int idd = 0 ; idd < level.size(); ++idd)
        {
            level(idd) -= 1;
            if (level(idd) > 0)
            {
                SparseSet::const_iterator iterFather = m_dataSet->find(level);
                level(idd) += 1;
                if (iterFather == m_dataSet->end())
                {
                    bAdd = false;
                    break;
                }
                else  if (m_oldLevel.find(iterFather) == m_oldLevel.end())
                {
                    bAdd = false;
                    break;
                }
            }
            else
                level(idd) += 1;
        }
        if (bAdd)
        {
            // modify data structure
            SparseSet::const_iterator iterNewLevel = addLevelToDataSet(iterErrorMax, id);
            // add to active level
            m_activeLevel.insert(iterNewLevel);
            vecRet.push_back(iterNewLevel);
        }
        level(id) -= 1;
    }
    return make_pair(vecRet, error);
}

void SparseSpaceGrid::refine(const double &p_precision, const function<double(const ArrayXd &p_x)> &p_fInterpol,
                             const function< double(const SparseSet::const_iterator &, const ArrayXd &)> &p_phi,
                             const function< double(const vector< double> &) > &p_phiMult,
                             ArrayXd &p_valuesFunction,
                             ArrayXd &p_hierarValues)
{

    dimensionAdaptiveInit();
    // to store the local error on each level
    std::map<  SparseSet::const_iterator, double, OrderLevel >  errorLevel;
    double error = infty;
    while (error > p_precision)
    {
        auto  levelAndPrec = dimensionRefineStep(p_hierarValues, p_phi, p_phiMult, p_precision, errorLevel) ;
        // update erro
        error = get<1>(levelAndPrec);
        // resize
        int nbPt = getNbPoints();
        if (nbPt > p_hierarValues.size())
        {
            p_hierarValues.conservativeResize(nbPt);
            p_valuesFunction.conservativeResize(nbPt);
            // add nodal values
            for (size_t i = 0; i < levelAndPrec.first.size(); ++i)
            {
                shared_ptr<SparseGridIterator> iterGridLevel = getLevelGridIterator(levelAndPrec.first[i]);
                while (iterGridLevel->isValid())
                {
                    ArrayXd pointCoord = iterGridLevel->getCoordinate();
                    p_valuesFunction(iterGridLevel->getCount()) = p_fInterpol(pointCoord);
                    iterGridLevel->next();
                }
                // hierarchize the level
                toHierarchizePByPLevel(p_valuesFunction, levelAndPrec.first[i], p_hierarValues);
            }
        }
    }
    // now recalculate son to be able to use added points in interpolation
    recalculateSon();

}

SparseSet::const_iterator  SparseSpaceGrid::dimensionCoarsenStep(map< SparseSet::const_iterator, double, OrderLevel >   &levelPotenRm)
{
    double errMin = infty;
    SparseSet::const_iterator iterMin;
    for (const auto &level  : levelPotenRm)
        if (level.second < errMin)
        {
            errMin = level.second;
            iterMin = level.first;
        }
    // test all directions
    ArrayXc level = iterMin->first;
    for (int id = 0; id < m_weight.size(); ++id)
    {
        if (level(id) > 1)
        {
            level(id) -= 1;
            // should we erase
            bool bErase = true;
            vector<SparseSet::const_iterator> toErase;
            for (int idd = 0; idd != m_weight.size(); ++idd)
            {
                level(idd) += 1;
                SparseSet::const_iterator  iterTest = m_dataSet->find(level);
                // should we erase
                if (iterTest != m_dataSet->end())
                {
                    if (levelPotenRm.find(iterTest) != levelPotenRm.end())
                    {
                        toErase.push_back(iterTest);
                    }
                    else
                        bErase = false;
                    level(idd) -= 1;
                    if (!bErase)
                        break;
                }
                else
                    level(idd) -= 1;
            }
            if (bErase)
            {
                // remove all from active level and potentially to erase
                for (size_t i = 0; i < toErase.size(); ++i)
                {
                    levelPotenRm.erase(toErase[i]);
                    m_activeLevel.erase(toErase[i]);
                    m_dataSet->erase(toErase[i]);
                }
                SparseSet::const_iterator iterChange = m_dataSet->find(level);
                m_activeLevel.insert(iterChange);
                m_oldLevel.erase(iterChange);
                return iterChange;
            }
            level(id) += 1;
        }
    }
    // remove from potentialy interesting (if not already removed)
    levelPotenRm.erase(iterMin);
    // for return
    return m_dataSet->end();

}

void SparseSpaceGrid::coarsen(const double &p_precision,
                              const function< double(const SparseSet::const_iterator &, const ArrayXd &)> &p_phi,
                              ArrayXd &p_valuesFunction,
                              ArrayXd   &p_hierarValues)
{
    dimensionAdaptiveInit();
    // select index potentially to remove
    map< SparseSet::const_iterator, double, OrderLevel > levelPotenRm;
    for (const auto &level : m_activeLevel)
    {
        double errLoc  = p_phi(level, p_hierarValues);
        if (errLoc < p_precision)
        {
            levelPotenRm[level] = errLoc;
        }
    }

    bool bModified = false;
    while (levelPotenRm.size() > 0)
    {
        SparseSet::const_iterator iterNewActive = dimensionCoarsenStep(levelPotenRm);
        if (iterNewActive != m_dataSet->end())
        {
            bModified = true;
            double errLoc = p_phi(iterNewActive, p_hierarValues);
            if (errLoc < p_precision)
                levelPotenRm[iterNewActive] = errLoc;
        }
    }

    if (bModified)
    {
        m_nbPoints = modifyHierarAndDataSetAfterCoarsen(p_hierarValues, p_valuesFunction);

        // now recalculate son to be able to use added points in interpolation
        recalculateSon();
    }

}

size_t SparseSpaceGrid::modifyHierarAndDataSetAfterCoarsen(ArrayXd   &p_hierarValues, ArrayXd &p_valuesFunction)
{
    shared_ptr<SparseSet>  newDataSet = make_shared<SparseSet>();
    ArrayXd newHierar, newValue;
    modifyDataSetAndHierachized<ArrayXd>(*m_dataSet, p_hierarValues, p_valuesFunction, *newDataSet, newHierar, newValue);
    p_hierarValues = newHierar;
    p_valuesFunction = newValue;
    m_dataSet = newDataSet;
    return p_valuesFunction.size();
}

size_t SparseSpaceGrid::modifyHierarAndDataSetAfterCoarsenVec(ArrayXXd   &p_hierarValues, ArrayXXd &p_valuesFunction)
{
    shared_ptr<SparseSet>  newDataSet = make_shared<SparseSet>();
    ArrayXXd newHierar, newValue ;
    modifyDataSetAndHierachized<ArrayXXd>(*m_dataSet, p_hierarValues, p_valuesFunction, *newDataSet, newHierar, newValue);
    p_hierarValues = newHierar;
    p_valuesFunction = newValue;
    m_dataSet = newDataSet;
    return p_valuesFunction.cols();
}
