#pragma once

#include <memory>
#include <stdexcept>
#include <vector>

#include "common/exception.h"

namespace bustub {

template <typename T>
class Matrix {
protected:
    Matrix(int rows, int cols) : _rows(rows), _cols(cols) {
        assert(rows > 0 && cols > 0);
        _linear = new T[_rows * _cols];
    }
    int _rows;
    int _cols;
    T* _linear;

public:
    virtual int GetRowCount() const = 0;
    virtual int GetColumnCount() const = 0;
    virtual T GetElement(int i, int j) const = 0;
    virtual void SetElement(int i, int j, T val) = 0;
    virtual void FillFrom(const std::vector<T>& source) = 0;
    virtual ~Matrix() { delete _linear; }
};

template <typename T>
class RowMatrix : public Matrix<T> {
public:
    RowMatrix(int rows, int cols) : Matrix<T>(rows, cols) {
        _data = new T*[rows];
        int offset = 0;
        for (int i = 0; i < rows; i++) {
            _data[i] = Matrix<T>::_linear + offset;
            offset += cols;
        }
    }
    int GetRowCount() const override { return Matrix<T>::_rows; }

    int GetColumnCount() const override { return Matrix<T>::_cols; }

    T GetElement(int i, int j) const override {
        if (i < 0 || i >= Matrix<T>::_rows || j < 0 || j >= Matrix<T>::_cols) {
            throw Exception(ExceptionType::OUT_OF_RANGE, "GetElement out of range");
        }
        return _data[i][j];
    }

    void SetElement(int i, int j, T val) override {
        if (i < 0 || i >= Matrix<T>::_rows || j < 0 || j >= Matrix<T>::_cols) {
            throw Exception(ExceptionType::OUT_OF_RANGE, "SetElement out of range");
        }
        _data[i][j] = val;
    }
    void FillFrom(const std::vector<T>& source) override {
        if (source.size() != (size_t)Matrix<T>::_rows * Matrix<T>::_cols) {
            throw Exception(ExceptionType::OUT_OF_RANGE, "FillFrom out of range");
        }
        for (int i = 0; i < Matrix<T>::_rows; i++) {
            for (int j = 0; j < Matrix<T>::_cols; j++) {
                _data[i][j] = source[i * Matrix<T>::_cols + j];
            }
        }
    }

    ~RowMatrix() override { delete[] _data; }

private:
    T** _data;
};

template <typename T>
class RowMatrixOperations {
public:
    static std::unique_ptr<RowMatrix<T>> Add(const RowMatrix<T>* matrixA, const RowMatrix<T>* matrixB) {
        if (!matrixA || !matrixB) {
            return nullptr;
        }
        int rowsA = matrixA->GetRowCount(), colsA = matrixA->GetColumnCount();
        int rowsB = matrixB->GetRowCount(), colsB = matrixB->GetColumnCount();

        if (rowsA != rowsB || colsA != colsB) {
            return nullptr;
        }
        auto result = std::make_unique<RowMatrix<T>>(rowsA, colsB);
        for (int i = 0; i < rowsA; i++) {
            for (int j = 0; j < colsA; j++) {
                T valueA = matrixA->GetElement(i, j);
                T valueB = matrixB->GetElement(i, j);
                result->SetElement(i, j, valueA + valueB);
            }
        }
        return result;
    }

    static std::unique_ptr<RowMatrix<T>> Multiply(const RowMatrix<T>* matrixA, const RowMatrix<T>* matrixB) {
        if (!matrixA || !matrixB) {
            return nullptr;
        }
        int rowsA = matrixA->GetRowCount(), colsA = matrixA->GetColumnCount();
        int rowsB = matrixB->GetRowCount(), colsB = matrixB->GetColumnCount();
        if (colsA != rowsB) {
            return nullptr;
        }
        auto result = std::make_unique<RowMatrix<T>>(rowsA, colsB);
        for (int i = 0; i < rowsA; i++) {
            for (int j = 0; j < colsB; j++) {
                // result->SetElement(i, j, (T)0);
                T tmp = 0;
                for (int k = 0; k < colsA; k++) {
                    tmp += matrixA->GetElement(i, k) * matrixB->GetElement(k, j);
                }
                result->SetElement(i, j, tmp);
            }
        }
        return std::unique_ptr<RowMatrix<T>>(nullptr);
    }

    static std::unique_ptr<RowMatrix<T>> GEMM(const RowMatrix<T>* matrixA, const RowMatrix<T>* matrixB,
                                              const RowMatrix<T>* matrixC) {
        return Add(Multiply(matrixA, matrixB), matrixC);
    }
};
}  // namespace bustub