/*
 * Copyright 2021 The DAPHNE Consortium
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/IsSymmetric.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <cstddef>

// ****************************************************************************
// Struct for partial template specialization
// res1 matrix for eigenvalues, res2 matrix for eigenvectors
// Column k of the returned matrix res2 is an eigenvector corresponding to
// eigenvalue number k as returned by eigenvalues(). The eigenvectors are
// normalized to have (Euclidean) norm equal to one.
// ****************************************************************************

template <class DTRes1, class DTRes2, class VTArg> struct EigenCal {
    static void apply(DTRes1 *&res1, DTRes2 *&res2, const VTArg *inMat, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************
template <class DTRes1, class DTRes2, class VTArg>
void eigenCal(DTRes1 *&res1, DTRes2 *&res2, const VTArg *inMat, DCTX(ctx)) {
    EigenCal<DTRes1, DTRes2, VTArg>::apply(res1, res2, inMat, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// Double Value types as input
// ----------------------------------------------------------------------------
template <> struct EigenCal<DenseMatrix<double>, DenseMatrix<double>, DenseMatrix<double>> {
    static void apply(DenseMatrix<double> *&res1, DenseMatrix<double> *&res2, const DenseMatrix<double> *inMat,
                      DCTX(ctx)) {
        const auto nr = static_cast<size_t>(inMat->getNumRows());
        const auto nc = static_cast<size_t>(inMat->getNumCols());
        if (!isSymmetric<DenseMatrix<double>>(inMat, nullptr)) {
            throw std::runtime_error("EigenCal - Input matrix must be symmetric");
        }

        Eigen::MatrixXd inputMatrix = Eigen::Map<const Eigen::MatrixXd>(inMat->getValues(), nr, nc);

        // the instance s(A) includes the eigensystem
        Eigen::EigenSolver<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> s(inputMatrix);

        size_t eigenValuesrows = s.eigenvalues().rows();
        size_t eigenValuescols = s.eigenvalues().cols();
        size_t eigenVectorsrows = s.eigenvectors().rows();
        size_t eigenVectorscols = s.eigenvectors().cols();
        Eigen::MatrixXd eigenVectors = s.eigenvectors().real().cast<double>();
        Eigen::MatrixXd eigenValues = s.eigenvalues().real().cast<double>();

        // TODO wrap the eigenValues, eigenVectors pointer into a shared_ptr and
        // then use
        // that to create the DenseMatrix

        if (res1 == nullptr)
            res1 = DataObjectFactory::create<DenseMatrix<double>>(eigenValuesrows, eigenValuescols, false);

        if (res2 == nullptr)
            res2 = DataObjectFactory::create<DenseMatrix<double>>(eigenVectorsrows, eigenVectorscols, false);

        for (size_t r = 0; r < eigenValuesrows; r++) {
            for (size_t c = 0; c < eigenValuescols; c++) {
                res1->set(r, c, s.eigenvalues()[r].real());
            }
        }
        for (size_t r = 0; r < eigenVectorsrows; r++) {
            for (size_t c = 0; c < eigenVectorscols; c++) {
                res2->set(r, c, eigenVectors.coeff(r, c));
            }
        }
    }
};

// ----------------------------------------------------------------------------
// DenseMatrix
// Float Value types as input
// ----------------------------------------------------------------------------
template <> struct EigenCal<DenseMatrix<float>, DenseMatrix<float>, DenseMatrix<float>> {
    static void apply(DenseMatrix<float> *&res1, DenseMatrix<float> *&res2, const DenseMatrix<float> *inMat,
                      DCTX(ctx)) {
        const auto nr = static_cast<size_t>(inMat->getNumRows());
        const auto nc = static_cast<size_t>(inMat->getNumCols());

        if (!isSymmetric<DenseMatrix<float>>(inMat, nullptr)) {
            throw std::runtime_error("EigenCal - Input matrix must be symmetric");
        }

        Eigen::MatrixXf inputMatrix = Eigen::Map<const Eigen::MatrixXf>(inMat->getValues(), nr, nc);

        // the instance s(A) includes the eigensystem
        Eigen::EigenSolver<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> s(inputMatrix);

        size_t eigenValuesrows = s.eigenvalues().rows();
        size_t eigenValuescols = s.eigenvalues().cols();
        size_t eigenVectorsrows = s.eigenvectors().rows();
        size_t eigenVectorscols = s.eigenvectors().cols();
        Eigen::MatrixXf eigenVectors = s.eigenvectors().real().cast<float>();
        Eigen::MatrixXf eigenValues = s.eigenvalues().real().cast<float>();

        // When it comes to float number, it has been noticed than at the
        // rounding part, errors occured at the sign of the results. So one
        // needs this iteration to make sure that the the results are correct
        // manually.

        for (size_t i = 0; i < eigenVectorscols; i++) {
            Eigen::VectorXcf ev = eigenVectors.col(i);
            size_t max_index = 0;
            for (int32_t j = 1; j < ev.size(); j++) {
                if (std::abs(ev(j)) > std::abs(ev(max_index)))
                    max_index = j;
            }
            if (ev(max_index).real() < 0)
                eigenVectors.col(i) *= -1;
        }

        if (res1 == nullptr)
            res1 = DataObjectFactory::create<DenseMatrix<float>>(eigenValuesrows, eigenValuescols, false);

        if (res2 == nullptr)
            res2 = DataObjectFactory::create<DenseMatrix<float>>(eigenVectorsrows, eigenVectorscols, false);

        for (size_t r = 0; r < eigenValuesrows; r++)
            for (size_t c = 0; c < eigenValuescols; c++)
                res1->set(r, c, s.eigenvalues()[r].real());

        for (size_t r = 0; r < eigenVectorsrows; r++)
            for (size_t c = 0; c < eigenVectorscols; c++)
                res2->set(r, c, eigenVectors.coeff(r, c));
    }
};
