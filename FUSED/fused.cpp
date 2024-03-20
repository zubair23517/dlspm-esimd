// basic version of fused operation where the two operations are
// performed sequentially
// sequential codes maded consistnet with the parallel code
// included last chunk processing
// separate D matrix is used in this version of the code in spmm
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cstring>
#include <cassert>
// C++ Code,
// Source for sequential: https://github.com/google-research/sputnik/blob/master/sputnik/sddmm/sddmm_test.cu.cc#L62

#define MIN(a, b) ((a) < (b) ? (a) : (b))

#ifndef MSZ
#define MSZ (2 * 1024)
#endif
#ifndef KSZ
#define KSZ (1 * 1024)
#endif
#ifndef NSZ
#define NSZ 32
#endif
#ifndef SP
#define SP 0.3
#endif

/*
A = C B^T
C is a dense matrix of size MSZ X NSZ
B is a dense matrix of size KSZ X NSZ
D is a dense matrix of size KSZ X NSZ 

A is a sparse matrix of size MSZ X KSZ
E is a dense matrix of size MSZ X NSZ

E = A D

E = FUSEDMM(Sparsity(A), C, B, D)


*/

// A = C B^T
void Sddmm(const int *a_row_offsets,
           const int *a_column_indices, const float *c,
           const float *b, float *a_values)
{
    for (int i = 0; i < MSZ; ++i)
    {
        for (int j = a_row_offsets[i]; j < a_row_offsets[i + 1]; ++j)
        {
            int idx_n = a_column_indices[j];
            double accumulator = 0.0;
            for (int l = 0; l < NSZ; ++l)
            {
                accumulator += static_cast<double>(c[i * NSZ + l]) *
                               static_cast<double>(b[idx_n * NSZ + l]);
            }
            a_values[j] = static_cast<float>(accumulator);
        }
    }
}
// E = A D
void Spmm(const float *a_values,
          const int *a_row_offsets, const int *a_column_indices,
          const float *d, float *e)
{
    for (int i = 0; i < MSZ; ++i)
    {
        for (int j = 0; j < NSZ; ++j)
        {
            double accum = 0.0;
            for (int l = a_row_offsets[i]; l < a_row_offsets[i + 1]; ++l)
            {
                float a_val = a_values[l];
                // print a_val
                // std::cout << "a_val = " << a_val << std::endl;
                int a_col = a_column_indices[l];
                accum += static_cast<double>(a_val) *
                         static_cast<double>(d[a_col * NSZ + j]);
            }
            e[i * NSZ + j] = static_cast<float>(accum);
            // if (i==0) printf("accum at %d : %f\n", i, accum);
        }
    }
}

#define VLC 32

void FusedEsimdCPU(const int *a_row_offsets,
                   const int *a_column_indices, 
                   const float *c, const float *b, 
                   const float *d, float *e)
{

    for (int i = 0; i < MSZ; i++)
    {

        float reg_left[NSZ];
        for (int m = 0; m < NSZ; m++)
        {
            reg_left[m] = c[i * NSZ + m];
        }
        int lrowptr2[2];
        lrowptr2[0] = a_row_offsets[i];
        lrowptr2[1] = a_row_offsets[i + 1];
        int nnzr = lrowptr2[1] - lrowptr2[0];
        float a_row[VLC];
        float e_row[NSZ] = {0.0};

        int nchunks = (nnzr + VLC - 1) / VLC;
        int col_ida[VLC];
        int idxb;
        for (int l = 0; l < nchunks - 1; l++)
        {
            idxb = lrowptr2[0] + l * VLC;

            for (int m = 0; m < VLC; m++)
            {
                col_ida[m] = a_column_indices[idxb + m];
            }

            for (int l0 = 0; l0 < VLC; l0++)
            {
                float reg_right[NSZ];
                float acc[NSZ];
                for (int m = 0; m < NSZ; m++)
                {
                    reg_right[m] = b[col_ida[l0] * NSZ + m];
                }
                for (int m = 0; m < NSZ; m++)
                {
                    acc[m] = reg_left[m] * reg_right[m];
                }
                for (int m = 1; m < NSZ; m++)
                {
                    acc[0] += acc[m];
                }
                a_row[l0] = acc[0];
            }
            float d_row[NSZ];
            // loop over VLC
            for (int j0 = 0; j0 < VLC; j0++)
            {
                float av = a_row[j0];
                int colid = col_ida[j0];

                // load d_row of size  NSZ identified by colid
                for (int l0 = 0; l0 < NSZ; l0++)
                {
                    d_row[l0] = d[colid * NSZ + l0];
                }
                // perform vector operations to compute e_row
                for (int l0 = 0; l0 < NSZ; l0++)
                {
                    e_row[l0] = e_row[l0] + av * d_row[l0];
                }
            }

        } // end of loop over chunks
          // last chunk processing
        int lcsz = nnzr - (nchunks - 1) * VLC;
        idxb = lrowptr2[0] + (nchunks - 1) * VLC;
        for (int m = 0; m < lcsz; m++)
        {
            col_ida[m] = a_column_indices[idxb + m];
        }
        for (int l0 = 0; l0 < lcsz; l0++)
        {
            float reg_right[NSZ];
            float acc[NSZ];
            for (int m = 0; m < NSZ; m++)
            {
                reg_right[m] = b[col_ida[l0] * NSZ + m];
            }
            for (int m = 0; m < NSZ; m++)
            {
                acc[m] = reg_left[m] * reg_right[m];
            }
            for (int m = 1; m < NSZ; m++)
            {
                acc[0] += acc[m];
            }
            a_row[l0] = acc[0];
        }

        float d_row[NSZ];
        // loop over VLC
        for (int j0 = 0; j0 < lcsz; j0++)
        {
            float av = a_row[j0];
            int colid = col_ida[j0];

            // load d_row of size  NSZ identified by colid
            for (int l0 = 0; l0 < NSZ; l0++)
            {
                d_row[l0] = d[colid * NSZ + l0];
            }
            // perform vector operations to compute e_row
            for (int l0 = 0; l0 < NSZ; l0++)
            {
                e_row[l0] = e_row[l0] + av * d_row[l0];
            }
        }

        for (int j0 = 0; j0 < NSZ; j0++)
        {

            e[i * NSZ + j0] = e_row[j0];
        }

    }
}

void SpmmEsimdCPU(const float *a_values,
                  const int *a_row_offsets, const int *a_column_indices,
                  const float *d, float *e)
{

    for (int i = 0; i < MSZ; i++)
    {

        // number of non zeroes in the ith row of lhs matrix
        int lrowptr2[2];
        lrowptr2[0] = a_row_offsets[i];
        lrowptr2[1] = a_row_offsets[i + 1];
        int nnzr = lrowptr2[1] - lrowptr2[0];

        int nchunks = (nnzr + VLC - 1) / VLC;
        int col_ida[VLC];
        float a_row[VLC];
        float e_row[NSZ] = {0.0};
        for (int l = 0; l < nchunks - 1; l++)
        {
            // load  VLC column indices and non-zero values in  temporary arrays
            int idxb = lrowptr2[0] + l * VLC;
            for (int l0 = 0; l0 < VLC; l0++)
            {
                int idx = idxb + l0;
                col_ida[l0] = a_column_indices[idx];
                a_row[l0] = a_values[idx];
            }
            float d_row[NSZ];
            // loop over VLC
            for (int j0 = 0; j0 < VLC; j0++)
            {
                float av = a_row[j0];
                int colid = col_ida[j0];

                // load d_row of size  NSZ identified by colid
                for (int l0 = 0; l0 < NSZ; l0++)
                {
                    d_row[l0] = d[colid * NSZ + l0];
                }
                // perform vector operations to compute e_row
                for (int l0 = 0; l0 < NSZ; l0++)
                {
                    e_row[l0] = e_row[l0] + av * d_row[l0];
                }
            }
        }

        // last chunk size
        int lcsz = nnzr - (nchunks - 1) * VLC;
        // load  VLC column indices and non-zero values in  vector registers
        int idxb = lrowptr2[0] + (nchunks - 1) * VLC;
        for (int l0 = 0; l0 < lcsz; l0++)
        {
            int idx = idxb + l0;
            col_ida[l0] = a_column_indices[idx];
            a_row[l0] = a_values[idx];
        }
        float d_row[NSZ];
        // loop over VLC
        for (int j0 = 0; j0 < lcsz; j0++)
        {
            float av = a_row[j0];
            int colid = col_ida[j0];

            // load d_row of size  NSZ identified by colid
            for (int l0 = 0; l0 < NSZ; l0++)
            {
                d_row[l0] = d[colid * NSZ + l0];
            }
            // perform vector operations to compute e_row
            for (int l0 = 0; l0 < NSZ; l0++)
            {
                e_row[l0] = e_row[l0] + av * d_row[l0];
            }
        }

        // store in c
        for (int j0 = 0; j0 < NSZ; j0++)
        {

            e[i * NSZ + j0] = e_row[j0];
        }

    } // end of msz loop over i
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

void validate(float *x_gold, float *x, int n)
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
}



int main(int argc, char const *argv[])
{

    // create a random sparse matrix with size m*n and initialize it with random values using rand() function
    int mnnz = (1.001 * MSZ * KSZ * SP) + 100;

    int *a_row_offsets = (int *)malloc((MSZ + 1) * sizeof(int));
    int *a_column_indices = (int *)malloc(mnnz * sizeof(int));
    // float *values = (float *)malloc(mnnz * sizeof(float));
    // initialize row_offsets and column_indices for a uniform random SP pattern
    a_row_offsets[0] = 0;
    for (int i = 0; i < MSZ; i++)
    {
        a_row_offsets[i + 1] = a_row_offsets[i];
        for (int j = 0; j < KSZ; j++)
        {
            if (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) < SP)
            {
                if (a_row_offsets[i + 1] >= mnnz)
                {
                    printf("nnz exceeding mnnz %d mnnz = %d\n", a_row_offsets[i + 1], mnnz);
                    exit(0);
                }
                a_column_indices[a_row_offsets[i + 1]] = j;
                // values not necessary
                // values[row_offsets[i + 1]] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
                a_row_offsets[i + 1]++;
            }
        }
    }

    int nnz = a_row_offsets[MSZ];
     
    printf("MSZ = %d KSZ = %d NSZ = %d SP = %f\n", MSZ, KSZ, NSZ, SP);
    printf("nnz = %d mnnz = %d\n", nnz, mnnz);
    assert(nnz < mnnz);
    // allocate lhs_matrix array of size m*k
    float *c = (float *)malloc(MSZ * NSZ * sizeof(float));
    float *e_gold = (float *)malloc(MSZ * NSZ * sizeof(float));
    float *e = (float *)malloc(MSZ * NSZ * sizeof(float));
    // create a random c array with size m*k and initialize it with random values using rand() function
    for (int i = 0; i < MSZ * NSZ; i++)
    {
        c[i] = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX))*0.01;
    }
    // allocate rhs_matrix array of size n*k
    // we are interested in multiplication with rhs_matrix transpose,
    // so we will read columns of rhs_matrix stored as rows (continuous memory access)
    float *b = (float *)malloc(KSZ * NSZ * sizeof(float));
    float *d = (float *)malloc(KSZ * NSZ * sizeof(float));
    // create a random b array with size n*k and initialize it with random values using rand() function

    for (int i = 0; i < KSZ * NSZ; i++)
    {
        b[i] = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX))*0.01;
        d[i] = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX))*0.01;
    }
    // allocate output array of size nnz
    float *a_values = (float *)malloc(nnz * sizeof(float));
    //  initialize a_values with zeros using memset
    memset(a_values, 0, nnz * sizeof(float));
    Sddmm(a_row_offsets, a_column_indices, c, b, a_values);
    // print checksum for a_values
    printf("checksum_a_values = %f\n", checksum(a_values, nnz));
     memset(e_gold, 0, MSZ * NSZ * sizeof(float));
    Spmm(a_values, a_row_offsets, a_column_indices, d, e_gold);
    printf("checksum_e = %f\n", checksum(e_gold, MSZ * NSZ));

    // esimd cpu

    memset(e, 0, MSZ * NSZ * sizeof(float));
    FusedEsimdCPU(a_row_offsets, a_column_indices, c, b, d, e);
    printf("checksum_e using esimd cpu = %f\n", checksum(e, MSZ * NSZ));
//
    validate(e_gold, e, MSZ * NSZ);
    return 0;
}
