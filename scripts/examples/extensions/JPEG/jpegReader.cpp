// jpegPlugIn.cpp
// Compile with:
//   g++ -std=c++17 -fPIC -shared \
//     jpegPlugIn.cpp -o libjpegPlugIn.so \
//     -I/home/yazan/daphneFork/daphne/src \
//     -I/home/yazan/daphneFork/daphne/thirdparty/installed/include \
//     -L/home/yazan/daphneFork/daphne/thirdparty/installed/lib \
//     -ljpeg -lDataStructures -lspdlog -lfmt

/*#include <iostream>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/datastructures/Structure.h>
#include <runtime/local/io/FileMetaData.h>
#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/kernels/CreateFrame.h>

#include <jpeglib.h>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include <string>

extern "C" {

void jpeg_read_matrix(
    Structure *&res,
    const FileMetaData &fmd,
    const char *filename,
    IOOptions &opts,
    DaphneContext *ctx
) {
    FILE *infile = fopen(filename, "rb");
    if (!infile)
        throw std::runtime_error("Cannot open JPEG file: " + std::string(filename));

    jpeg_decompress_struct cinfo;
    jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);

    size_t width = cinfo.output_width;
    size_t height = cinfo.output_height;
    size_t channels = cinfo.output_components;

    if (channels != 3) {
        jpeg_destroy_decompress(&cinfo);
        fclose(infile);
        throw std::runtime_error("Only RGB JPEG images are supported.");
    }

    auto *mat = DataObjectFactory::create<DenseMatrix<uint32_t>>(height, width, false);
    uint32_t *data = mat->getValues();

    JSAMPARRAY buffer = (*cinfo.mem->alloc_sarray)
        ((j_common_ptr)&cinfo, JPOOL_IMAGE, width * channels, 1);

    while (cinfo.output_scanline < height) {
        jpeg_read_scanlines(&cinfo, buffer, 1);
        uint8_t *row_ptr = buffer[0];

        for (size_t x = 0; x < width; ++x) {
            uint8_t R = row_ptr[x * 3 + 0];
            uint8_t G = row_ptr[x * 3 + 1];
            uint8_t B = row_ptr[x * 3 + 2];
            data[(cinfo.output_scanline - 1) * width + x] =
                (R << 16) | (G << 8) | B;
        }
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);

    res = mat;
}

// ============================================================================
// JPEG → Frame (cols = image width, rows = image height)
// ============================================================================
void jpeg_read_frame(
    Frame *&res,
    const FileMetaData &fmd,
    const char *filename,
    IOOptions &opts,
    DaphneContext *ctx
) {
    // First decode as matrix
    Structure *matStruct = nullptr;
    jpeg_read(matStruct, fmd, filename, opts, ctx);
    auto *mat = static_cast<DenseMatrix<uint32_t>*>(matStruct);

    size_t rows = mat->getNumRows();
    size_t cols = mat->getNumCols();
    uint32_t *src = mat->getValues();

    // Prepare one DenseMatrix<uint32_t>(rows, 1) for each image column
    std::vector<Structure*> columns(cols);
    std::vector<std::string> labels(cols);
    for (size_t c = 0; c < cols; ++c) {
        auto *colMat = DataObjectFactory::create<DenseMatrix<uint32_t>>(rows, 1, false);
        uint32_t *dst = colMat->getValues();
        for (size_t r = 0; r < rows; ++r)
            dst[r] = src[r * cols + c];
        columns[c] = colMat;
        labels[c] = "col_" + std::to_string(c);
    }

    // Convert labels to const char* for createFrame
    std::vector<const char*> colLabels(cols);
    for (size_t c = 0; c < cols; ++c)
        colLabels[c] = labels[c].c_str();

    createFrame(res, columns.data(), cols, colLabels.data(), cols, ctx);
}

} // extern "C"
*/