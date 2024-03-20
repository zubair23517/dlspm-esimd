#include <cusparse.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#ifdef OUTPUT_CSV_FILE
#include <fstream>
#endif

#define MIN(a, b) ((a) < (b) ? (a) : (b))

/*
C =  A B
A is a sparse matrix of size MSZ X KSZ
B is a dense matrix of size  KSZ x NSZ
C is a dense matrix of size  MSZ x NSZ

*/

void Spmm(const float *a_values,
          const int *a_row_offsets, const int *a_column_indices,
          const float *b, float *c, const int MSZ, const int NSZ)
{
    for (int i = 0; i < MSZ; ++i)
    {
        for (int j = 0; j < NSZ; ++j)
        {
            float accum = 0.0;
            for (int l = a_row_offsets[i]; l < a_row_offsets[i + 1]; ++l)
            {
                float a_val = a_values[l];
                // print a_val
                // std::cout << "a_val = " << a_val << std::endl;
                int a_col = a_column_indices[l];
                accum += a_val * b[a_col * NSZ + j];
            }
            c[i * NSZ + j] = accum;
        }
    }
}

double checksum(float *x, int n)
{
    double sum = 0.0;
    for (int i = 0; i < n; i++)
    {
        sum += x[i] * x[i];
    }
    sum = sqrt(sum / n);
    return sum;
}

double validate(float *x_gold, float *x, int n)
{
    double max_diff = 0.0;
    int maxi = 0;
    for (int i = 0; i < n; i++)
    {
        double diff = fabs(x_gold[i] - x[i]);
        if (diff > max_diff)
        {
            max_diff = diff;
            maxi = i;
        }
    }
    // print maxi and the maximum difference
    std::cout << "maxi = " << maxi << " max_diff = " << max_diff << std::endl;
    // print the values of the two arrays at maxi with 7 digits of precision
    std::cout.precision(7);
    std::cout << "gold output[" << maxi << "] = " << x_gold[maxi] << std::endl;
    std::cout << "computed[" << maxi << "] = " << x[maxi] << std::endl;

    return max_diff;
}

int main(int argc, char **argv)
{
    if (argc != 3)
        throw std::invalid_argument("Usage: ./app <filename>.smtx <batchsize>");

    std::string filename(argv[1]);
    const int NSZ_rt = std::stoi(argv[2]);
    std::cout << "Filename = " << filename << ", batchsize (=NSZ) = " << NSZ_rt << std::endl;

    int nDevices;
    float peak_bw;
    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",
               prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
               prop.memoryBusWidth);

        peak_bw = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6;
        printf("  Peak Memory Bandwidth (GB/s): %6.2f\n\n", peak_bw);
    }

    // Load lhs matrix from file
    FILE *fp;
    fp = fopen(filename.c_str(), "r");
    if (fp == NULL)
    {
        throw std::logic_error("Error opening file!");
    }

    int MSZ_rt, KSZ_rt, nnz_rt;
    if (fscanf(fp, "%d,%d,%d", &MSZ_rt, &KSZ_rt, &nnz_rt) != 3)
        throw std::logic_error("Something went wrong readin M, K, nnz");

    printf("MSZ: %d, KSZ: %d, NSZ: %d, nnz: %d\n", MSZ_rt, KSZ_rt, NSZ_rt, nnz_rt);

    int *row_offsets = (int *)malloc((MSZ_rt + 1) * sizeof(int));
    int *column_indices = (int *)malloc(nnz_rt * sizeof(int));
    float *a_values = (float *)malloc(nnz_rt * sizeof(float));

    for (int i = 0; i < MSZ_rt + 1; i++)
    {
        if (fscanf(fp, "%d", &row_offsets[i]) != 1)
            throw std::logic_error("Something went wrong reading row_offsets");
    }

    for (int i = 0; i < nnz_rt; i++)
    {
        if (fscanf(fp, "%d", &column_indices[i]) != 1)
            throw std::logic_error("Something went wrong reading column_indices");

        a_values[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX); // fill lhs with random values
    }

    fclose(fp);

    float *b = (float *)malloc(KSZ_rt * NSZ_rt * sizeof(float));
    std::cout << "rhs_matrix allocated" << std::endl;

    for (int i = 0; i < NSZ_rt * KSZ_rt; i++)
    {
        b[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    // allocate output array of size MSZ X NSZ
    float *c = (float *)malloc(MSZ_rt * NSZ_rt * sizeof(float));
    // initialize output array with zeros using memset function
    memset(c, 0, MSZ_rt * NSZ_rt * sizeof(float));

    Spmm(a_values, row_offsets, column_indices, b, c, MSZ_rt, NSZ_rt);
    // print checksum for a_values
    // printf("checksum_a_values = %f\n", checksum(c, MSZ * NSZ));

#if 1
    // device arrays and copy data from host to device
    int *a_row_offsets_dev;
    cudaMalloc((void **)&a_row_offsets_dev, (MSZ_rt + 1) * sizeof(int));
    // copy data from host to device
    cudaMemcpy(a_row_offsets_dev, row_offsets, (MSZ_rt + 1) * sizeof(int), cudaMemcpyHostToDevice);
    int *a_column_indices_dev;
    cudaMalloc((void **)&a_column_indices_dev, nnz_rt * sizeof(int));
    cudaMemcpy(a_column_indices_dev, column_indices, nnz_rt * sizeof(int), cudaMemcpyHostToDevice);
    float *a_values_dev;
    cudaMalloc((void **)&a_values_dev, nnz_rt * sizeof(float));
    cudaMemcpy(a_values_dev, a_values, nnz_rt * sizeof(float), cudaMemcpyHostToDevice);

    float *b_dev;
    cudaMalloc((void **)&b_dev, KSZ_rt * NSZ_rt * sizeof(float));
    cudaMemcpy(b_dev, b, KSZ_rt * NSZ_rt * sizeof(float), cudaMemcpyHostToDevice);

    float *c_dev;
    cudaMalloc((void **)&c_dev, MSZ_rt * NSZ_rt * sizeof(float));
    cudaMemset(c_dev, 0, MSZ_rt * NSZ_rt * sizeof(float));

    cusparseHandle_t handle = NULL;
    cusparseDnMatDescr_t matC, matB;
    cusparseSpMatDescr_t matA;
    void *dBuffer = NULL;
    size_t bufferSize = 0;
    cusparseCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    // Create dense matrix C
    cusparseCreateDnMat(&matC, MSZ_rt, NSZ_rt, NSZ_rt, c_dev,
                        CUDA_R_32F, CUSPARSE_ORDER_ROW);

    // Create dense matrix B
    cusparseCreateDnMat(&matB, KSZ_rt, NSZ_rt, NSZ_rt, b_dev,
                        CUDA_R_32F, CUSPARSE_ORDER_ROW);

    // Create sparse matrix C in CSR format
    cusparseCreateCsr(&matA, MSZ_rt, KSZ_rt, nnz_rt,
                      a_row_offsets_dev, a_column_indices_dev, a_values_dev,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    // allocate an external buffer if needed
    cusparseSpMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize);

    // print buffer size
    // printf("sddmm_csr_example buffer size: %lld bytes\n", (long long)bufferSize);
    cudaMalloc(&dBuffer, bufferSize);

    cudaEvent_t tstart, tstop;
    cudaEventCreate(&tstart);
    cudaEventCreate(&tstop);

    float timeKernel;
    float exectime = 123456789999.0;

    for (int loop = 0; loop < 20; ++loop)
    {
        cudaEventRecord(tstart);

        cusparseSpMM(handle,
                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                     CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);

        cudaEventRecord(tstop);
        cudaEventSynchronize(tstop);
        cudaEventElapsedTime(&timeKernel, tstart, tstop);

        if (exectime > timeKernel)
            exectime = timeKernel;
    }

    printf("Kernel time in ms:  %.12f\n", exectime);
    float opcount = (2 * (float)nnz_rt - MSZ_rt) * NSZ_rt;
    float gflops1 = (opcount) / (exectime * 1.0E+6);
    printf("GFLOPS:  %8.2f\n", gflops1);

    float *c_host;
    c_host = (float *)malloc(MSZ_rt * NSZ_rt * sizeof(float));
    cudaMemcpy(c_host, c_dev, MSZ_rt * NSZ_rt * sizeof(float), cudaMemcpyDeviceToHost);
    printf("checksum c values from CPU = %f\n", checksum(c, MSZ_rt * NSZ_rt));
    printf("checksum C values from GPU = %f\n", checksum(c_host, MSZ_rt * NSZ_rt));
    const double max_diff = validate(c, c_host, MSZ_rt * NSZ_rt);

#ifdef OUTPUT_CSV_FILE
    std::ofstream csv_file;
    csv_file.open(OUTPUT_CSV_FILE, std::ios::app | std::ios::out);
    csv_file << MSZ_rt << "," << KSZ_rt << "," << NSZ_rt << "," << (double)nnz_rt / (MSZ_rt * KSZ_rt) << "," << gflops1 << "," << max_diff << std::endl;
    csv_file.close();
#endif

    cusparseDestroyDnMat(matC);
    cusparseDestroyDnMat(matB);
    cusparseDestroySpMat(matA);
    cusparseDestroy(handle);

#endif
    // free cuda memory
    cudaFree(a_row_offsets_dev);
    cudaFree(a_column_indices_dev);
    cudaFree(a_values_dev);
    cudaFree(b_dev);
    cudaFree(c_dev);
    cudaFree(dBuffer);

    // free host and device memory
    free(row_offsets);
    free(column_indices);
    free(a_values);
    free(c);
    free(b);

    return 0;
}
