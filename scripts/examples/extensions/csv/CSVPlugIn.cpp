#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cctype>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Structure.h>
#include "runtime/local/io/FileIORegistry.h"

// Helpers to detect types
bool is_integer(const std::string &s) {
    if (s.empty()) return false;
    size_t i = (s[0] == '+' || s[0] == '-') ? 1 : 0;
    for (; i < s.size(); ++i) if (!std::isdigit(s[i])) return false;
    return true;
}

bool is_double(const std::string &s) {
    std::istringstream ss(s);
    double d;
    char c;
    return !!(ss >> d) && !(ss >> c);
}

extern "C" {

/**
 * Reads a CSV file and outputs a DenseMatrix<T>* via the 'res' parameter.
 * Infers a single homogeneous type for the entire matrix: int32_t, double, or std::string.
 * Signature: void csv_read(Structure*& res, const FileMetaData&, const char* filename, DaphneContext*)
 */
void csv_read(Structure*& res,
              const FileMetaData& fmd,
              const char* filename,
              IOOptions& opts,
              DaphneContext* ctx) {
        std::ifstream file(filename);
    if(!file.is_open())
        throw std::runtime_error(std::string("Failed to open file: ") + filename);

    std::string line;
    size_t cols = 0;
    std::vector<std::vector<std::string>> raw;


    // 1) Respect opts.hasHeader when deciding whether to skip the first line
    if(std::getline(file, line)) {
        if(opts.hasHeader) {
            // This was the header row—use it to count columns, but don't store it
            std::istringstream ss(line);
            std::string col;
            while(std::getline(ss, col, opts.delimiter))
                ++cols;
        }
        else {
            // No header: treat first line as data
            std::istringstream ss(line);
            std::string cell;
            std::vector<std::string> row;
            while(std::getline(ss, cell, opts.delimiter))
                row.push_back(cell);
            cols = row.size();
            if(!row.empty()) raw.emplace_back(std::move(row));
        }
    }

    // 2) Read the rest of the file using opts.delimiter
    while(std::getline(file, line)) {
        std::istringstream ss(line);
        std::string cell;
        std::vector<std::string> row;
        while(std::getline(ss, cell, opts.delimiter))
            row.push_back(cell);
        if(!row.empty()) raw.emplace_back(std::move(row));
    }
    file.close();

    size_t rows = raw.size();
    if (rows == 0 || cols == 0)
        throw std::runtime_error("Empty CSV or missing header");

    // Validate consistent column counts
    for (auto &row : raw) {
        if (row.size() != cols)
            throw std::runtime_error("Inconsistent column count in CSV");
    }

    // Infer a single global type
    bool allInt = true, allNum = true;
    for (auto &row : raw) {
        for (auto &cell : row) {
            if (allInt && !is_integer(cell)) allInt = false;
            if (allNum && !(is_integer(cell) || is_double(cell))) allNum = false;
            if (!allInt && !allNum) break;
        }
        if (!allInt && !allNum) break;
    }

    // Allocate and populate DenseMatrix based on inferred type
    if (allInt) {
        auto *mat = DataObjectFactory::create<DenseMatrix<int32_t>>(rows, cols, false);
        int32_t *data = mat->getValues();
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                data[i * cols + j] = std::stoi(raw[i][j]);
        res = mat;
        return;
    }

    if (allNum) {
        auto *mat = DataObjectFactory::create<DenseMatrix<double>>(rows, cols, false);
        double *data = mat->getValues();
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                data[i * cols + j] = std::stod(raw[i][j]);
        res = mat;
        return;
    }

    // Fallback to string matrix
    auto *mat_str = DataObjectFactory::create<DenseMatrix<std::string>>(rows, cols, false);
    std::string *data_str = mat_str->getValues();
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            data_str[i * cols + j] = raw[i][j];
    res = mat_str;
}

/**
 * Writes a DenseMatrix<T> to CSV. Supports int32_t, double, and std::string.
 * Signature: void csv_write(const Structure* matrix, const FileMetaData&, const char* filename, DaphneContext*)
 */
void csv_write(const Structure* matrix,
               const FileMetaData& fmd,
               const char* filename,
               IOOptions& opts,
               DaphneContext* ctx) {
    std::ofstream file(filename);
    if (!file.is_open())
        throw std::runtime_error(std::string("Failed to open for writing: ") + filename);

    // Cast and write int32 matrix
    if (auto *im = dynamic_cast<const DenseMatrix<int32_t>*>(matrix)) {
        size_t r = im->getNumRows(), c = im->getNumCols();
        const int32_t *data = im->getValues();
        for (size_t i = 0; i < r; ++i) {
            for (size_t j = 0; j < c; ++j)
                file << data[i * c + j] << (j + 1 < c ? ',' : '\n');
        }
    }
    // Cast and write double matrix
    else if (auto *dm = dynamic_cast<const DenseMatrix<double>*>(matrix)) {
        size_t r = dm->getNumRows(), c = dm->getNumCols();
        const double *data = dm->getValues();
        for (size_t i = 0; i < r; ++i) {
            for (size_t j = 0; j < c; ++j)
                file << data[i * c + j] << (j + 1 < c ? ',' : '\n');
        }
    }
    // Cast and write string matrix
    else if (auto *sm = dynamic_cast<const DenseMatrix<std::string>*>(matrix)) {
        size_t r = sm->getNumRows(), c = sm->getNumCols();
        const std::string *data = sm->getValues();
        for (size_t i = 0; i < r; ++i) {
            for (size_t j = 0; j < c; ++j)
                file << data[i * c + j] << (j + 1 < c ? ',' : '\n');
        }
    }
    else {
        throw std::runtime_error("csv_write: unsupported matrix type");
    }
}

} // extern "C"