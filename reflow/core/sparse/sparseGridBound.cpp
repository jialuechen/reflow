
#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include "reflow/core/utils/comparisonUtils.h"
#include "reflow/core/sparse/sparseGridTypes.h"
#include "reflow/core/sparse/sparseGridUtils.h"
#include "reflow/core/sparse/sparseGridCommon.h"


namespace reflow
{



void recursiveSparseConstructionBound(Eigen::ArrayXc &p_levelCurrent,
                                      Eigen::ArrayXui &p_positionCurrent,
                                      const unsigned short  int &p_idim,
                                      const bool &p_bInsideBound,
                                      SparseSet &p_dataSet,
                                      size_t &p_ipoint)
{
    SparseSet::iterator iterDataStructure = p_dataSet.find(p_levelCurrent);
    if (iterDataStructure != p_dataSet.end())
    {
        if (p_bInsideBound)
        {

            unsigned int oldPosition = p_positionCurrent(p_idim);
            // position in tree
            if (p_idim == 0)
            {
                // jump to  dim below
                p_positionCurrent(p_idim) = 0; // left
                iterDataStructure->second[p_positionCurrent] = p_ipoint++;
                p_positionCurrent(p_idim) = 2; // right
                iterDataStructure->second[p_positionCurrent] = p_ipoint++;
                p_positionCurrent(p_idim) = 1; // center
                recursiveSparseConstructionBound(p_levelCurrent, p_positionCurrent, p_idim, false, p_dataSet, p_ipoint);
            }
            else
            {
                // jump to dim below
                p_positionCurrent(p_idim) = 0; // left
                recursiveSparseConstructionBound(p_levelCurrent, p_positionCurrent, p_idim - 1, true, p_dataSet, p_ipoint);
                p_positionCurrent(p_idim) = 2; // right
                recursiveSparseConstructionBound(p_levelCurrent, p_positionCurrent, p_idim - 1, true, p_dataSet, p_ipoint);
                // same dimension
                p_positionCurrent(p_idim) = 1; // center
                recursiveSparseConstructionBound(p_levelCurrent, p_positionCurrent, p_idim, false, p_dataSet, p_ipoint);

            }
            p_positionCurrent(p_idim) = oldPosition;

        }
        else
        {
            if (p_idim == 0)
            {
                // create point and go inside tree
                iterDataStructure->second[p_positionCurrent] = p_ipoint++;
                if (p_levelCurrent(0) > 1)
                {
                    // new level
                    sparse1DConstruction(p_levelCurrent, p_positionCurrent, p_dataSet, p_ipoint);
                }
                else
                {
                    unsigned int  oldPosition = p_positionCurrent(0);
                    p_positionCurrent(0) = 0 ;
                    // new level
                    sparse1DConstruction(p_levelCurrent, p_positionCurrent, p_dataSet, p_ipoint);
                    p_positionCurrent(0) =  oldPosition;
                }
            }
            else
            {
                // direction below
                recursiveSparseConstructionBound(p_levelCurrent, p_positionCurrent, p_idim - 1, true, p_dataSet, p_ipoint);
                // child level
                char oldLevel = p_levelCurrent(p_idim);
                p_levelCurrent(p_idim) += 1;
                unsigned int oldPosition = p_positionCurrent(p_idim);
                // same direction but level above
                if (p_levelCurrent(p_idim) > 2)
                {
                    // LEFT
                    p_positionCurrent(p_idim) = 2 * oldPosition;
                    recursiveSparseConstructionBound(p_levelCurrent, p_positionCurrent, p_idim, false, p_dataSet, p_ipoint);
                    // RIGHT
                    p_positionCurrent(p_idim) += 1;
                    recursiveSparseConstructionBound(p_levelCurrent, p_positionCurrent, p_idim, false, p_dataSet, p_ipoint);
                }
                else
                {
                    // LEFT
                    p_positionCurrent(p_idim) = 0;
                    recursiveSparseConstructionBound(p_levelCurrent, p_positionCurrent, p_idim, false, p_dataSet, p_ipoint);
                    // RIGHT
                    p_positionCurrent(p_idim) = 1;
                    recursiveSparseConstructionBound(p_levelCurrent, p_positionCurrent, p_idim, false, p_dataSet, p_ipoint);
                }
                p_positionCurrent(p_idim) = oldPosition;
                p_levelCurrent(p_idim) = oldLevel;

            }
        }
    }
}


void initialSparseConstructionBound(const unsigned int &p_levelMax,
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
    double levelCalc =  p_alpha.sum();
    // create map
    createLevelsSparse(firstLevel, 0, p_levelMax + p_alpha.size() - 1, p_alpha, p_dataSet, levelCalc);

    // initialize point
    Eigen::ArrayXui positionCurrent = Eigen::ArrayXui::Constant(p_alpha.size(), static_cast<unsigned int>(1));

    recursiveSparseConstructionBound(firstLevel, positionCurrent, p_alpha.size() - 1, true, p_dataSet, p_ipoint);
}



void initialFullConstructionBound(const unsigned int &p_levelMax,
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
    Eigen::ArrayXui positionCurrent = Eigen::ArrayXui::Constant(p_alpha.size(), static_cast<unsigned int>(1));

    // first iterator
    recursiveSparseConstructionBound(firstLevel, positionCurrent, p_alpha.size() - 1, true, p_dataSet, p_ipoint);
}




int sonEvaluationBound(const SparseSet   &p_dataSet, const int &p_idim,
                       const int   &p_nbPoint,
                       Eigen::Array< std::array<int, 2 >, Eigen::Dynamic, Eigen::Dynamic > &p_son,
                       Eigen::Array< std::array<int, 2 >, Eigen::Dynamic, Eigen::Dynamic > &p_neighbourBound)
{
    p_son.resize(p_nbPoint, p_idim);
    p_neighbourBound.resize(p_nbPoint, p_idim);

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
                    if (level(id) > 2)
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
                            position = iPosition.first;
                            int iposPoint = iPosition.second;
                            if (position(id) == 1)
                            {
                                // left son
                                position(id) = 0;
                                p_son(iposPoint, id)[0] = iterSon->second.find(position)->second;
                                // right son
                                position(id) = 1;
                                p_son(iposPoint, id)[1] = iterSon->second.find(position)->second;
                            }
                            else
                            {
                                p_son(iposPoint, id)[0] = -1;
                                p_son(iposPoint, id)[1] = -1;
                            }
                        }
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

        // for neighbours

#ifdef _OPENMP
        cnt = 0;
#endif
        for (const auto &iLevel :  p_dataSet)
        {
#ifdef _OPENMP
            cnt += 1;
            if ((cnt - 1) % nthreads != ithread) continue;
#endif
            Eigen::ArrayXc level = iLevel.first;
            Eigen::ArrayXui position(p_idim);
            for (int id = 0; id < p_idim; ++id)
                if (level(id) == 1)
                {
                    for (const auto &iPosition : iLevel.second)
                        if (iPosition.first(id) == 1)
                        {
                            int iposPoint = iPosition.second;
                            // central point
                            position = iPosition.first;
                            // left
                            position(id) = 0;
                            p_neighbourBound(iposPoint, id)[0] = iLevel.second.find(position)->second;
                            // right
                            position(id) = 2;
                            p_neighbourBound(iposPoint, id)[1] = iLevel.second.find(position)->second;
                        }
                }
        }
    }
    // root
    Eigen::ArrayXc levelRoot =  Eigen::ArrayXc::Constant(p_idim, 1);
    Eigen::ArrayXui  positionRoot = Eigen::ArrayXui::Constant(p_idim, 1);
    SparseSet::const_iterator iterLevel = p_dataSet.find(levelRoot);
    SparseLevel::const_iterator iterPosition = iterLevel->second.find(positionRoot);
    return   iterPosition->second ;
}

}
