#ifndef ONEMKL_TEST_HPP
#define ONEMKL_TEST_HPP

#include <iostream>
#include <vector>
#include <random>
#include <sycl/sycl.hpp>
#include "oneapi/mkl/blas.hpp"
#include "oneapi/mkl/spblas.hpp"
#include "mkl.h"
#include "oneapi/mkl/rng.hpp"

/// Structure representing M,N,K,SP parameters as in paper
struct Params
{
    Params() : M(1), K(1), N(1), SP(0.0)
    {
    }

    Params(size_t m, size_t k, size_t n, double sp) : M(m), K(k), N(n), SP(sp)
    {
    }

    size_t M;
    size_t K;
    size_t N;
    double SP;

    friend std::ostream &operator<<(std::ostream &os, const Params &p);
};

/// Overload operator for above structure
std::ostream &operator<<(std::ostream &os, const Params &p)
{
    os << p.M << "," << p.K << "," << p.N << "," << p.SP;
    return os;
}

/// Structure representing all test parameters
struct AllTestParams
{
    void Add(const Params &p) { params_.push_back(p); }
    const Params &Get(const int i) const
    {
        if (i < params_.size())
            return params_[i];
        else
            throw std::invalid_argument("Out of range");
    }
    size_t Size() const { return params_.size(); }

private:
    std::vector<Params> params_;
};

// Helper function to zero a matrix
void ZeroMatrix(float *M, const size_t size, sycl::queue &Q)
{
    Q.memset(M, 0.0f, size * sizeof(float)).wait();
}

// Helper function to fill a matrix with random values
void RandomizeMatrix(float *M, const size_t size, sycl::queue &Q)
{
    // create basic random number generator object
    oneapi::mkl::rng::philox4x32x10 engine(Q, 0);
    // create distribution object
    oneapi::mkl::rng::gaussian<float, oneapi::mkl::rng::gaussian_method::icdf> distr(5.0, 2.0);

    // perform generation
    oneapi::mkl::rng::generate(distr, engine, size, M).wait();
}

// Generate ia and ja array for a sparse matrix. 'nnz' gives the number of non-zero entries
void GenerateSparseMatrix(const int nrows, const int ncols, const int nnz,
                          int *ia, int *ja, sycl::queue &Q)
{
    std::vector<int> row_offsets(nrows + 1, 0);
    std::vector<int> column_indices(nnz, 0);

    std::vector<int> all_indices(nrows * ncols, 0);
    std::iota(all_indices.begin(), all_indices.end(), 0);

    std::random_device rd;
    std::mt19937 g(rd());

    std::vector<int> my_indices;
    std::sample(all_indices.begin(), all_indices.end(), std::back_inserter(my_indices), nnz, g);
    std::vector<int> rows(my_indices.size());
    std::transform(my_indices.begin(), my_indices.end(), rows.begin(), [=](int x)
                   { return x / ncols; });
    std::transform(my_indices.begin(), my_indices.end(), column_indices.begin(), [=](int x)
                   { return x % ncols; });

    for (int iter = 0; iter < nnz; iter++)
    {
        row_offsets[rows[iter]]++;
    }
    for (int iiter = 0; iiter < nrows; iiter++)
    {
        row_offsets[iiter + 1] += row_offsets[iiter];
    }

    Q.memcpy(ja, column_indices.data(), nnz * sizeof(int)).wait();
    Q.memcpy(ia, row_offsets.data(), (nrows + 1) * sizeof(int)).wait();
}

/// Dense Gemm. Ignoring SP value
/// Class which provides functions for a dense matrix-matrix multiplication
// Note that for the performance of SDDMM with this, the perf has to be multiplied with
// a factor of (1-sp)
class BlasGemm
{
public:
    void Apply(sycl::queue &Q)
    {
        oneapi::mkl::blas::row_major::gemm(Q,
                                           oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
                                           p_.M, p_.N, p_.K, 1.0f, A, p_.K, B, p_.N, 0, C, p_.N)
            .wait();
    }

    static void ShowName()
    {
        std::cout << "Dense-dense matrix matrix product";
    }

    /// Generates the matrices in the required format
    void GenerateMatrices(sycl::queue &Q, const Params &p)
    {
        // std::cout << "Generating matrices for " << p << std::endl;
        A = sycl::malloc_device<float>(p.M * p.K, Q);
        B = sycl::malloc_device<float>(p.K * p.N, Q);
        C = sycl::malloc_device<float>(p.M * p.N, Q);
        RandomizeMatrix(A, p.M * p.K, Q);
        RandomizeMatrix(B, p.K * p.N, Q);
        ZeroMatrix(C, p.M * p.N, Q);
        p_ = p;
        // std::cout << "Done generating" << std::endl;
    }

    void FreeMatrices(sycl::queue &Q)
    {
        sycl::free(A, Q);
        sycl::free(B, Q);
        sycl::free(C, Q);
    }

    static double GetOpCount(const Params &p)
    {
        return 2 * p.M * p.N * p.K; // for each value in C (mxn values), we perform an inner product consisting of k multiplications and k-1 additions, which we round to k additions.
    }

    double GetMaxError(sycl::queue &Q) const
    {
        std::vector<float> host_C(p_.M * p_.N);
        std::vector<float> host_B(p_.K * p_.N);
        std::vector<float> host_A(p_.M * p_.K);
        Q.memcpy(host_C.data(), C, p_.M * p_.N * sizeof(float)).wait();
        Q.memcpy(host_B.data(), B, p_.K * p_.N * sizeof(float)).wait();
        Q.memcpy(host_A.data(), A, p_.M * p_.K * sizeof(float)).wait();

        double max_rel_error = 0.0;
        for (int i = 0; i < p_.M; i++)
        {
            for (int j = 0; j < p_.N; j++)
            {
                float tmp_sum = 0.0f;
                for (int k = 0; k < p_.K; k++)
                {
                    tmp_sum += host_A[i * p_.K + k] * host_B[k * p_.N + j];
                }
                double tmp_error = std::abs<float>(tmp_sum - host_C[i * p_.N + j]);
                double tmp_rel_error = tmp_error == 0.0 ? 0.0 : tmp_error / std::max<float>(std::abs<float>(tmp_sum), std::abs<float>(host_C[i * p_.N + j]));
                max_rel_error = std::max<double>(max_rel_error, tmp_rel_error);
            }
        }

        return max_rel_error;
    }

private:
    float *A; // MxK
    float *B; // KxN
    float *C; // MxN
    Params p_;
};

/// This class emulates a SDDMM with a dense MKL gemm computing A=CB^T
/// and then performs a SPMM using E=DB
/// where A is MxK, C is MxN, B is KxN, B^T is NxK, D is MxK and E is MxN.
class FusedMM
{
public:
    void Apply(sycl::queue &Q)
    {
        oneapi::mkl::blas::row_major::gemm(Q,
                                           oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::trans,
                                           p_.M, p_.K, p_.N, 1.0f, C, p_.N, B, p_.N, 0, A, p_.K)
            .wait();

        oneapi::mkl::sparse::gemm(Q, oneapi::mkl::layout::row_major,
                                  oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
                                  1.0f, handle_, B, p_.N, p_.N, 0.0f, E, p_.N)
            .wait();
    }

    static void ShowName()
    {
        std::cout << "Fused matrix matrix product";
    }

    /// Generates the matrices in the required format
    void GenerateMatrices(sycl::queue &Q, const Params &p)
    {
        A = sycl::malloc_device<float>(p.M * p.K, Q);
        B = sycl::malloc_device<float>(p.K * p.N, Q);
        C = sycl::malloc_device<float>(p.M * p.N, Q);
        D = sycl::malloc_device<float>(p.M * p.K, Q);
        E = sycl::malloc_device<float>(p.M * p.N, Q);
        RandomizeMatrix(B, p.K * p.N, Q);
        RandomizeMatrix(C, p.M * p.N, Q);
        RandomizeMatrix(D, p.M * p.K, Q);
        nnz_ = p.M * p.K * (1.0 - p.SP);
        ia_ = sycl::malloc_device<int>(p.M + 1, Q);
        ja_ = sycl::malloc_device<int>(nnz_, Q);
        GenerateSparseMatrix(p.M, p.K, nnz_, ia_, ja_, Q);
        oneapi::mkl::sparse::init_matrix_handle(&handle_);
        oneapi::mkl::sparse::set_csr_data(Q, handle_, p.M, p.K,
                                          oneapi::mkl::index_base::zero, ia_, ja_, D)
            .wait();

        p_ = p;
    }

    void FreeMatrices(sycl::queue &Q)
    {
        sycl::free(A, Q);
        sycl::free(B, Q);
        sycl::free(C, Q);
        oneapi::mkl::sparse::release_matrix_handle(Q, &handle_);
        sycl::free(D, Q);
        sycl::free(E, Q);
    }

    static double GetOpCount(const Params &p)
    {
        return (1.0 - p.SP) * 2.0 * p.M * p.N * p.K +                  // sddmm part
               (2.0 * ((double)p.M * p.K * (1.0 - p.SP)) - p.M) * p.N; // spmm
    }

    double GetMaxError(sycl::queue &Q) const
    {
        std::vector<float> host_C(p_.M * p_.N);
        std::vector<float> host_B(p_.K * p_.N);
        std::vector<float> host_A(p_.M * p_.K);
        Q.memcpy(host_C.data(), C, p_.M * p_.N * sizeof(float)).wait();
        Q.memcpy(host_B.data(), B, p_.K * p_.N * sizeof(float)).wait();
        Q.memcpy(host_A.data(), A, p_.M * p_.K * sizeof(float)).wait();

        double max_rel_error = 0.0;
        for (int i = 0; i < p_.M; i++)
        {
            for (int j = 0; j < p_.K; j++)
            {
                float tmp_sum = 0.0f;
                for (int inner = 0; inner < p_.N; inner++)
                {
                    tmp_sum += host_C[i * p_.N + inner] * host_B[inner + j * p_.N];
                }
                double tmp_error = std::abs<float>(tmp_sum - host_A[i * p_.K + j]);
                double tmp_rel_error = tmp_error == 0.0 ? 0.0 : tmp_error / std::max<float>(std::abs<float>(tmp_sum), std::abs<float>(host_A[i * p_.K + j]));
                max_rel_error = std::max<double>(max_rel_error, tmp_rel_error);
            }
        }

        std::vector<float> host_E(p_.M * p_.N); // result
        std::vector<float> host_D(p_.M * p_.K); // sparse
        std::vector<int> host_ia(p_.M + 1);
        std::vector<int> host_ja(nnz_);
        Q.memcpy(host_E.data(), E, p_.M * p_.N * sizeof(float)).wait();
        Q.memcpy(host_D.data(), D, p_.M * p_.K * sizeof(float)).wait();
        Q.memcpy(host_ia.data(), ia_, (p_.M + 1) * sizeof(int)).wait();
        Q.memcpy(host_ja.data(), ja_, nnz_ * sizeof(int)).wait();

        for (int i = 0; i < p_.M; i++)
        {
            for (int j = 0; j < p_.N; j++)
            {
                float tmp_sum = 0.0f;
                for (int iterk = host_ia[i]; iterk < host_ia[i + 1]; iterk++)
                {
                    int k = host_ja[iterk];
                    tmp_sum += host_D[iterk] * host_B[k * p_.N + j];
                }
                double tmp_error = std::abs<float>(tmp_sum - host_E[i * p_.N + j]);
                double tmp_rel_error = tmp_error == 0.0 ? 0.0 : tmp_error / std::max<float>(std::abs<float>(tmp_sum), std::abs<float>(host_E[i * p_.N + j]));
                max_rel_error = std::max<double>(max_rel_error, tmp_rel_error);
            }
        }

        return max_rel_error;
    }

private:
    float *A; // MxK
    float *B; // KxN
    float *C; // MxN
    float *D; // MxK
    float *E; // MxN
    // Data for the sparse matrix D
    oneapi::mkl::sparse::matrix_handle_t handle_ = nullptr;
    int nnz_;
    int *ia_; // index array for sparse A
    int *ja_; // index array for sparse A
    Params p_;
};

/// Class which provides functions for sparse matrix times dense matrix
class SparseBlasGemm
{
public:
    void Apply(sycl::queue &Q)
    {
        oneapi::mkl::sparse::gemm(Q, oneapi::mkl::layout::row_major,
                                  oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
                                  1.0f, handle, B, p_.N, p_.N, 0.0f, C, p_.N)
            .wait();
    }

    static void ShowName()
    {
        std::cout << "Sparse-dense matrix matrix product";
    }

    void GenerateMatrices(sycl::queue &Q, const Params &p)
    {
        A = sycl::malloc_device<float>(p.M * p.K, Q); // not all are used
        B = sycl::malloc_device<float>(p.K * p.N, Q);
        C = sycl::malloc_device<float>(p.M * p.N, Q);
        nnz = p.M * p.K * (1.0 - p.SP);
        ia = sycl::malloc_device<int>(p.M + 1, Q);
        ja = sycl::malloc_device<int>(nnz, Q);

        RandomizeMatrix(A, p.M * p.K, Q); // values of sparse matrix. We only take the first nonzero number
        RandomizeMatrix(B, p.K * p.N, Q);
        ZeroMatrix(C, p.M * p.N, Q);
        GenerateSparseMatrix(p.M, p.K, nnz, ia, ja, Q);

        oneapi::mkl::sparse::init_matrix_handle(&handle);
        oneapi::mkl::sparse::set_csr_data(Q, handle, p.M, p.K,
                                          oneapi::mkl::index_base::zero, ia, ja, A)
            .wait();

        p_ = p;
    }

    Params GenerateMatrices(sycl::queue &Q, const std::string &filename, const size_t batch_size)
    {
        FILE *fp;
        fp = fopen(filename.c_str(), "r");
        if (fp == NULL)
            throw std::logic_error("Error opening file!");

        size_t M, K;
        if (fscanf(fp, "%zd,%zd,%d", &M, &K, &nnz) != 3)
            throw std::logic_error("Something went wrong readin M, K, nnz");

        std::vector<int> row_offsets(M + 1);
        std::vector<int> column_indices(nnz);

        for (size_t i = 0; i < M + 1; i++)
        {
            if (fscanf(fp, "%d", &row_offsets[i]) != 1)
                throw std::logic_error("Something went wrong reading row_offsets");
        }

        for (size_t i = 0; i < nnz; i++)
        {
            if (fscanf(fp, "%d", &column_indices[i]) != 1)
                throw std::logic_error("Something went wrong reading column_indices");
        }

        fclose(fp);

        Params p{M, K, batch_size, 1.0 - static_cast<double>(nnz) / (static_cast<double>(M) * K)};

        A = sycl::malloc_device<float>(p.M * p.K, Q); // not all are used
        B = sycl::malloc_device<float>(p.K * p.N, Q);
        C = sycl::malloc_device<float>(p.M * p.N, Q);
        ia = sycl::malloc_device<int>(p.M + 1, Q);
        ja = sycl::malloc_device<int>(nnz + 100, Q); // there is a bug in MKL
        Q.memcpy(ia, row_offsets.data(), (p.M + 1) * sizeof(int)).wait();
        Q.memcpy(ja, column_indices.data(), nnz * sizeof(int)).wait();

        RandomizeMatrix(A, p.M * p.K, Q); // values of sparse matrix. We only take the first nonzero number
        RandomizeMatrix(B, p.K * p.N, Q);
        ZeroMatrix(C, p.M * p.N, Q);

        oneapi::mkl::sparse::init_matrix_handle(&handle);
        oneapi::mkl::sparse::set_csr_data(Q, handle, p.M, p.K,
                                          oneapi::mkl::index_base::zero, ia, ja, A)
            .wait();

        p_ = p;

        return p;
    }

    void FreeMatrices(sycl::queue &Q)
    {
        oneapi::mkl::sparse::release_matrix_handle(Q, &handle);
        sycl::free(A, Q);
        sycl::free(B, Q);
        sycl::free(C, Q);
        sycl::free(ia, Q);
        sycl::free(ja, Q);
    }

    static double GetOpCount(const Params &p)
    {
        // for each value in C (mxn values),
        // we perform an inner product consisting of an expected k*(1-sp) multiplications
        // and k*(1-sp)-1 additions, which we round to k*(1-sp) additions.
        // return 2 * p.M * p.N * p.K * (1.0-p.SP);
        return (2.0 * ((double)p.M * p.K * (1.0 - p.SP)) - p.M) * p.N;
    }

    double GetMaxError(sycl::queue &Q) const
    {
        std::vector<float> host_C(p_.M * p_.N);
        std::vector<float> host_B(p_.K * p_.N);
        std::vector<float> host_A(p_.M * p_.K);
        std::vector<int> host_ia(p_.M + 1);
        std::vector<int> host_ja(nnz);
        Q.memcpy(host_C.data(), C, p_.M * p_.N * sizeof(float)).wait();
        Q.memcpy(host_B.data(), B, p_.K * p_.N * sizeof(float)).wait();
        Q.memcpy(host_A.data(), A, p_.M * p_.K * sizeof(float)).wait();
        Q.memcpy(host_ia.data(), ia, (p_.M + 1) * sizeof(int)).wait();
        Q.memcpy(host_ja.data(), ja, nnz * sizeof(int)).wait();

        double max_rel_error = 0.0;
        for (int i = 0; i < p_.M; i++)
        {
            for (int j = 0; j < p_.N; j++)
            {
                float tmp_sum = 0.0f;
                for (int iterk = host_ia[i]; iterk < host_ia[i + 1]; iterk++)
                {
                    const int k = host_ja[iterk];
                    tmp_sum += host_A[iterk] * host_B[k * p_.N + j];
                }
                const double tmp_error = std::abs<float>(tmp_sum - host_C[i * p_.N + j]);
                const double tmp_rel_error = tmp_error == 0.0 ? 0.0 : tmp_error / std::max<float>(std::abs<float>(tmp_sum), std::abs<float>(host_C[i * p_.N + j]));
                max_rel_error = std::max<double>(max_rel_error, tmp_rel_error);
            }
        }

        return max_rel_error;
    }

private:
    oneapi::mkl::sparse::matrix_handle_t handle = nullptr;
    int nnz;
    float *A; // MxK, sparse CSR format
    int *ia;  // index array for sparse A
    int *ja;  // index array for sparse A
    float *B; // KxN
    float *C; // MxN
    Params p_;
};

class SparseBlasMatMat
{
public:
    void Apply()
    {
    }

    static void ShowName()
    {
        std::cout << "Sparse-sparse matrix matrix product";
    }

    void GenerateMatrices(const Params &p)
    {
    }

private:
};

template <class Method>
class Tester
{
public:
    void Run(const AllTestParams &params, const bool ComputeError)
    {
        /// Generate all the matrices and plug them in 'Method' and measure perf
        sycl::queue Q(sycl::gpu_selector_v, sycl::property::queue::enable_profiling{});
        PrintHeader(Q, ComputeError);
        Method m;

        for (int param_iter = 0; param_iter < params.Size(); param_iter++)
        {
            auto p = params.Get(param_iter);
            m.GenerateMatrices(Q, p);
            m.Apply(Q); // apply before measurement to ensure no Jitting is included

            Q.wait();
            double min_duration = 1234567899999;
            StartTimer();
            for (int reps = 0; reps < NREPS; reps++)
            {
                // std::cout << reps << std::endl;
                auto start_time_loc = std::chrono::high_resolution_clock::now();
                m.Apply(Q);
                auto stop_time_loc = std::chrono::high_resolution_clock::now();
                const double duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop_time_loc - start_time_loc).count();
                min_duration = std::min(duration, min_duration);
            }
            StopTimer();

            ComputePerformance(p, m, min_duration, ComputeError, Q);

            m.FreeMatrices(Q);
        }
    }

    void Run(const std::string &filename, const int batch_size, const bool ComputeError)
    {
        /// Generate all the matrices and plug them in 'Method' and measure perf
        sycl::queue Q(sycl::gpu_selector_v, sycl::property::queue::enable_profiling{});
        // PrintHeader(Q, ComputeError);
        Method m;

        auto p = m.GenerateMatrices(Q, filename, batch_size);
        Q.wait();

        m.Apply(Q); // apply before measurement to ensure no Jitting is included

        Q.wait();
        double min_duration = 1234567899999;
        StartTimer();
        for (int reps = 0; reps < NREPS; reps++)
        {
            // std::cout << reps << std::endl;
            auto start_time_loc = std::chrono::high_resolution_clock::now();
            m.Apply(Q);
            auto stop_time_loc = std::chrono::high_resolution_clock::now();
            const double duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop_time_loc - start_time_loc).count();
            min_duration = std::min(duration, min_duration);
        }
        StopTimer();

        ComputePerformance(p, m, min_duration, ComputeError, Q);

        m.FreeMatrices(Q);
    }

private:
    void StartTimer()
    {
        start_time = std::chrono::high_resolution_clock::now();
    }

    void StopTimer()
    {
        stop_time = std::chrono::high_resolution_clock::now();
    }

    void PrintHeader(sycl::queue &Q, const bool ComputeError)
    {
        Method::ShowName();
        std::cout << std::endl;
        std::cout << "Running on " << Q.get_device().get_info<sycl::info::device::name>() << std::endl;
        std::cout << "M,K,N,SP,GFLOPS/s average, GFLOPS/s max,#Flops,avg duration (ns)";
        if (ComputeError)
            std::cout << ",max rel error";

        std::cout << std::endl;
    }

    void ComputePerformance(const Params &p, Method &m, const double min_duration, const bool ComputeError,
                            sycl::queue &Q)
    {
        const double avg_duration = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(stop_time - start_time).count()) / NREPS;
        const double NFLOPS = Method::GetOpCount(p);

        std::cout << p << "," << NFLOPS / avg_duration << "," << NFLOPS / min_duration << "," << NFLOPS << "," << avg_duration;

        if (ComputeError)
            std::cout << "," << m.GetMaxError(Q);

        std::cout << std::endl;
    }

    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    std::chrono::time_point<std::chrono::high_resolution_clock> stop_time;
    static constexpr int NREPS = 20;
};

#endif
