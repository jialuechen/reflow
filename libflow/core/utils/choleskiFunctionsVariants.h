// Copyright (C) 2016 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#ifndef CHOLESKIFUNCTIONSVARIANTS_H
#define CHOLESKIFUNCTIONSVARIANTS_H
#include "Eigen/Dense"
#include "libflow/core/utils/constant.h"


/** \file choleskiFunctionsVariants.h
 * \brief Implement different Choleski inversion
 *         - a first used a recursive Choleski
 *           it permits to store  a modified factorised version when  Choleski factor is null
 *         - a second version permits to calculate the inverse of a matrix with Strassen algorithm
 *         - a third uses Eigen Choleski implementation to invert the non singular matrix
 */
namespace libflow
{
/// \brief Inversion of a Matrix by  a Strassen algorithm (regular invertible matrix)
/// \param m_mToInvert     Invertible matrix to invert
/// \param m_inverse       inverse matrix
void recurInverseStrassen(const Eigen::MatrixXd &m_mToInvert, Eigen::MatrixXd &m_inverse)
{
    if (m_mToInvert.size() == 1)
    {
        m_inverse(0, 0) = 1. / m_mToInvert(0, 0);
        return;
    }
    // use Strassen formulae
    int k = m_mToInvert.rows() * 0.5;
    int nk =  m_mToInvert.rows() - k ;
    Eigen::MatrixXd A11 =  m_mToInvert.topLeftCorner(k, k);
    Eigen::MatrixXd A12 =  m_mToInvert.topRightCorner(k, nk);
    Eigen::MatrixXd A21 =  m_mToInvert.bottomLeftCorner(nk, k);
    Eigen::MatrixXd A22 =  m_mToInvert.bottomRightCorner(nk, nk);

    Eigen::MatrixXd R1(A11.rows(), A11.cols());
    recurInverseStrassen(A11, R1);

    Eigen::MatrixXd R2 = A21 * R1;
    Eigen::MatrixXd R3 = R1 * A12;
    Eigen::MatrixXd R4 = A21 * R3;
    Eigen::MatrixXd R5 = R4 - A22;

    Eigen::MatrixXd R6(R5.rows(), R5.cols());
    recurInverseStrassen(R5, R6);

    m_inverse.topRightCorner(k, nk) = R3 * R6;
    m_inverse.bottomLeftCorner(nk, k) = R6 * R2;

    Eigen::MatrixXd R7 = R3 * m_inverse.bottomLeftCorner(nk, k);
    m_inverse.topLeftCorner(k, k) = R1 - R7;
    m_inverse.bottomRightCorner(nk, nk) = -R6;
}

/// \brief Recursive Choleski : besides test if the matrix is singular
/// \param  p_A           Matrix
/// \param  p_U           Choleski Factor
/// \param  p_Y           p_U inverse
/// \param  p_bSingular     True if singular
void recurChol(const Eigen::MatrixXd &p_A,  Eigen::MatrixXd &p_U, Eigen::MatrixXd &p_Y, bool &p_bSingular)
{
    if (p_A.rows() == 1)
    {
        if (std::fabs(p_A(0, 0)) > tiny)
        {
            p_U(0, 0) = std::sqrt(p_A(0, 0));
            p_Y(0, 0) = 1. / p_U(0, 0);
        }
        else
        {
            p_bSingular = true ;
            p_U(0, 0) = 0. ;
            p_Y(0, 0) = 0. ;
        }
        return ;
    }
    int k = p_A.rows() * 0.5;
    int nk = p_A.rows() - k ;
    // calculate
    //              A =   ( A11, A12)
    //                      A21, A22)
    Eigen::MatrixXd A11 =  p_A.topLeftCorner(k, k);
    Eigen::MatrixXd A12 =  p_A.topRightCorner(k, nk);
    Eigen::MatrixXd A22 =  p_A.bottomRightCorner(nk, nk);


    Eigen::MatrixXd Y11(k, k);
    Eigen::MatrixXd U11(k, k);
    // recursive Choleski
    recurChol(A11, U11, Y11, p_bSingular);
    Eigen::MatrixXd U12 = Y11.transpose() * A12;
    Eigen::MatrixXd T2 = A22 - U12.transpose() * U12;

    Eigen::MatrixXd Y22(nk, nk);
    Eigen::MatrixXd U22(nk, nk);
    // recursive Choleski
    recurChol(T2, U22, Y22, p_bSingular);
    Eigen::MatrixXd Y12  = - Y11 * U12 * Y22;

    // reconstruct U
    p_U.topLeftCorner(k, k) = U11;
    p_U.topRightCorner(k, nk) = U12;
    p_U.bottomLeftCorner(nk, k) = Eigen::MatrixXd::Zero(nk, k);
    p_U.bottomRightCorner(nk, nk) = U22;

    // reconstruct Y
    p_Y.topLeftCorner(k, k) = Y11;
    p_Y.topRightCorner(k, nk) = Y12;
    p_Y.bottomLeftCorner(nk, k) = Eigen::MatrixXd::Zero(nk, k);
    p_Y.bottomRightCorner(nk, nk) = Y22;

}


/// \brief From a matrix U  (upper triangular) previously calculated
///        Form  L* the matrix suppressing the 0 rows of U
///        calculate L*L
///        calculate its inverse with strassen algorithm
///        calculate the More Penrose inverse of U :  L(LTL)^{-2}LT
/// \param  p_U matrix to "invert"
/// \param  p_Y pseudo inverse
void    pseudoInverse(const Eigen::MatrixXd &p_U, Eigen::MatrixXd   &p_Y)
{
    /// count number of non 0 rows
    int iNonZero = 0;
    Eigen::MatrixXd LT(p_U.rows(), p_U.cols());
    for (int i = 0 ; i <  p_U.rows(); ++i)
    {
        if (p_U(i, i) > 0.)
        {
            LT.row(iNonZero) = p_U.row(i);
            iNonZero += 1;
        }
    }
    LT.conservativeResize(iNonZero, p_U.rows());
    // calculate LTL
    Eigen::MatrixXd T = LT * LT.transpose();
    // Inverse by strassen
    Eigen::MatrixXd M(T.rows(), T.cols());
    recurInverseStrassen(T, M);
    // Calculate pseudo inverse
    p_Y = LT.transpose() * M * M * LT;
}

/// \brief  Avoid Strassen : From a matrix U  (upper triangular) previously calculated
///         Form  L* the matrix suppressing the 0 rows of U
///         calculate L*L
///         Factorize it with Eigen Choleski
///         Store : LT and the factorisation of L*L
/// \param  p_U           matrix to "invert"
/// \param  p_LT         Choleski factor to calculate
/// \param  p_factMatrix  factorized matrix
void    pseudoInverseFac(const Eigen::MatrixXd &p_U, Eigen::MatrixXd   &p_LT, Eigen::LLT<Eigen::MatrixXd> &p_factMatrix)
{
    /// count number of non 0 rows
    int iNonZero = 0;
    for (int i = 0 ; i <  p_U.rows(); ++i)
    {
        if (p_U(i, i) > 0.)
        {
            p_LT.row(iNonZero) = p_U.row(i);
            iNonZero += 1;
        }
    }
    p_LT.conservativeResize(iNonZero, p_U.rows());
    // calculate LTL
    Eigen::MatrixXd T = p_LT * p_LT.transpose();
    // Factorize
    p_factMatrix.compute(T);
}

}
#endif /* CHOLESKIFUNCTIONSVARIANTS_H */

