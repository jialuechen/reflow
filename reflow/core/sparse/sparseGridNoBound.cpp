
#include <iostream>
#include <vector>
#include <functional>
#include <Eigen/Dense>
#include "reflow/core/sparse/sparseGridTypes.h"
#include "reflow/core/sparse/sparseGridUtils.h"
#include "reflow/core/sparse/sparseGridCommon.h"
#include "reflow/core/sparse/GetCoordinateNoBound.h"

namespace reflow
{

void recursiveSparseConstructionNoBound(Eigen::ArrayXc &p_levelCurrent,
                                        Eigen::ArrayXui &p_positionCurrent,
                                        SparseSet::iterator &p_iterDataStructureCurrent,
                                        const unsigned short  int &p_idim,
                                        SparseSet &p_dataSet,
                                        size_t &p_ipoint)
{
    if (p_iterDataStructureCurrent != p_dataSet.end())
    {
        if (p_idim == 0)
        {
            p_iterDataStructureCurrent->second[p_positionCurrent] = p_ipoint++ ;
            sparse1DConstruction(p_levelCurrent, p_positionCurrent, p_dataSet, p_ipoint);
        }
        else
        {
            recursiveSparseConstructionNoBound(p_levelCurrent, p_positionCurrent, p_iterDataStructureCurrent, p_idim - 1, p_dataSet, p_ipoint);

            char oldLevel = p_levelCurrent(p_idim);
            unsigned int oldPosition = p_positionCurrent(p_idim);

            // child level
            p_levelCurrent(p_idim) += 1;
            SparseSet::iterator iterDataStructureChild = p_dataSet.find(p_levelCurrent);
            // LEFT
            p_positionCurrent(p_idim) = 2 * oldPosition;
            recursiveSparseConstructionNoBound(p_levelCurrent, p_positionCurrent, iterDataStructureChild, p_idim, p_dataSet, p_ipoint);
            // RIGHT
            p_positionCurrent(p_idim) = 2 * oldPosition + 1;
            recursiveSparseConstructionNoBound(p_levelCurrent, p_positionCurrent, iterDataStructureChild, p_idim, p_dataSet, p_ipoint);

            p_positionCurrent(p_idim) = oldPosition;
            p_levelCurrent(p_idim) = oldLevel;
        }
    }
}



void initialSparseConstructionNoBound(const unsigned int &p_levelMax,
                                      const Eigen::ArrayXd &p_alpha,
                                      SparseSet   &p_dataSet,
                                      size_t     &p_ipoint)
{
    if (p_levelMax < 1)
    {
        std::cout << "Level should be above 0 " << std::endl;
        abort();
    }
    // fist level
    Eigen::ArrayXc firstLevel = Eigen::ArrayXc::Constant(p_alpha.size(), static_cast<char>(1));
    double levelCalc = p_alpha.sum();
    // create map
    createLevelsSparse(firstLevel, 0, p_levelMax + p_alpha.size() - 1, p_alpha, p_dataSet, levelCalc);

    // initialize point
    Eigen::ArrayXui positionCurrent = Eigen::ArrayXui::Constant(p_alpha.size(), static_cast<unsigned int>(0));

    // first iterator
    SparseSet::iterator iterDataFirst = p_dataSet.find(firstLevel);
    recursiveSparseConstructionNoBound(firstLevel, positionCurrent, iterDataFirst, p_alpha.size() - 1, p_dataSet, p_ipoint);
}

void initialFullConstructionNoBound(const unsigned int &p_levelMax,
                                    const Eigen::ArrayXd &p_alpha,
                                    SparseSet   &p_dataSet,
                                    size_t     &p_ipoint)
{
    if (p_levelMax < 1)
    {
        std::cout << "Level should be above 0 " << std::endl;
        abort();
    }
    // fist level
    Eigen::ArrayXc firstLevel = Eigen::ArrayXc::Constant(p_alpha.size(), static_cast<char>(1));

    // create map
    createLevelsFull(firstLevel, 0, p_levelMax, p_alpha, p_dataSet);

    // initialize point
    Eigen::ArrayXui positionCurrent = Eigen::ArrayXui::Constant(p_alpha.size(), static_cast<unsigned int>(0));

    // first iterator
    SparseSet::iterator iterDataFirst = p_dataSet.find(firstLevel);
    recursiveSparseConstructionNoBound(firstLevel, positionCurrent, iterDataFirst, p_alpha.size() - 1, p_dataSet, p_ipoint);
}




void createBasisFunctionNoBound(const int &p_levelMax, const Eigen::ArrayXd &p_weight, const int &p_degree, std::vector< std::vector< std::function< double(const double &) > > > &p_functionScal)
{
    int initLevelMax = 0;
    for (int id = 0 ; id < p_weight.size(); ++id)
        initLevelMax = std::max(initLevelMax, static_cast<int>(p_levelMax / p_weight(id)));
    p_functionScal.resize(initLevelMax);

    // first level
    {
        p_functionScal[0].resize(1);
        p_functionScal[0][0] = OneFunction();
    }
    // level two
    {
        p_functionScal[1].resize(2);
        p_functionScal[1][0] =  LinearHatValue(0., 2, 2.);
        p_functionScal[1][1] =  LinearHatValue(1, 2., 2.);
    }
    // nest on level (
    for (int ilevel = 2 ; ilevel < initLevelMax; ++ilevel)
    {
        p_functionScal[ilevel].resize(lastNode[ilevel] + 1);
        {
            // left element
            p_functionScal[ilevel][0] =   LinearHatValue(0., 2. / deltaSparseMesh[ilevel - 1], 2.);
        }
        {
            // right element
            p_functionScal[ilevel][lastNode[ilevel]] =  LinearHatValue(1., 2. / deltaSparseMesh[ilevel - 1], 2.);
        }
        if (p_degree == 1)
        {
            for (size_t iindex = 1; iindex < lastNode[ilevel]; ++iindex)
            {
                double coordNod = GetCoordinateNoBound()(ilevel + 1, iindex);
                p_functionScal[ilevel][iindex] =  LinearHatValue(coordNod, 2. / deltaSparseMesh[ilevel]);
            }
        }
        else if (p_degree == 2)
        {
            for (size_t iindex = 1; iindex < lastNode[ilevel]; ++iindex)
            {
                double coordNod = GetCoordinateNoBound()(ilevel + 1, iindex);
                p_functionScal[ilevel][iindex] =   QuadraticValue(coordNod, 2. / deltaSparseMesh[ilevel]);
            }
        }
        else
        {
            // degree 3
            {
                {
                    double coordNod = GetCoordinateNoBound()(ilevel + 1, 1);
                    p_functionScal[ilevel][1] = QuadraticValue(coordNod, 2. / deltaSparseMesh[ilevel]);
                }
                for (size_t iindex = 2; iindex < lastNode[ilevel] - 1; ++iindex)
                {
                    double coordNod = GetCoordinateNoBound()(ilevel + 1, iindex);
                    int iBaseType = iindex % 2;
                    // cubic.
                    if (iBaseType == 0)
                    {
                        p_functionScal[ilevel][iindex] = CubicLeftValue(coordNod, 2. / deltaSparseMesh[ilevel]);
                    }
                    else
                    {
                        p_functionScal[ilevel][iindex] = CubicRightValue(coordNod, 2. / deltaSparseMesh[ilevel]);
                    }
                }
                {
                    double coordNod = GetCoordinateNoBound()(ilevel + 1, lastNode[ilevel] - 1);
                    p_functionScal[ilevel][lastNode[ilevel] - 1] =   QuadraticValue(coordNod, 2. / deltaSparseMesh[ilevel]);
                }
            }
        }
    }
}

int  sonEvaluationNoBound(const SparseSet   &p_dataSet, const int &p_idim,
                          const int   &p_nbPoint,
                          Eigen::Array< std::array<int, 2 >, Eigen::Dynamic, Eigen::Dynamic > &p_son)
{
    p_son.resize(p_nbPoint, p_idim);
#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
#ifdef _OPENMP
        size_t ithread = omp_get_thread_num();
        size_t nthreads = omp_get_num_threads();
        size_t cnt = 0;
#endif
        for (const auto &fatherLevel : p_dataSet)
        {
#ifdef _OPENMP
            cnt = +1;
            if ((cnt - 1) % nthreads != ithread) continue;
#endif
            Eigen::ArrayXc level = fatherLevel.first;
            Eigen::ArrayXui position(p_idim);
            for (int id = 0; id < p_idim; ++id)
            {
                level(id) += 1;
                SparseSet::const_iterator iterSon = p_dataSet.find(level);
                if (iterSon != p_dataSet.end())
                {
                    for (const auto &iPosition : fatherLevel.second)
                    {
                        position = iPosition.first;
                        int iposPoint = iPosition.second;
                        // left son
                        position(id) *= 2;
                        p_son(iposPoint, id)[0] = iterSon->second.find(position)->second;
                        // right son
                        position(id) += 1;
                        p_son(iposPoint, id)[1] = iterSon->second.find(position)->second;
                    }
                }
                else
                {
                    for (const auto &iPosition : fatherLevel.second)
                    {
                        int iposPoint = iPosition.second;
                        p_son(iposPoint, id)[0] = -1;
                        p_son(iposPoint, id)[1] = -1;
                    }
                }
                level(id) -= 1;
            }
        }
    }
    // root
    Eigen::ArrayXc levelRoot =  Eigen::ArrayXc::Constant(p_idim, 1);
    Eigen::ArrayXui  positionRoot = Eigen::ArrayXui::Constant(p_idim, 0);
    SparseSet::const_iterator iterLevel = p_dataSet.find(levelRoot);
    SparseLevel::const_iterator iterPosition = iterLevel->second.find(positionRoot);
    return  iterPosition->second ;
}
}
