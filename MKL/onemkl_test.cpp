/// Testmain for onemkl matrix multiplication functions.
/// We test the performance of following three functions on 1T PVC :
/// https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2023-2/gemm.html#ONEMKL-BLAS-GEMM
/// https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2023-2/oneapi-mkl-sparse-gemm.html
/// https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2023-2/oneapi-mkl-sparse-matmat.html
///
/// We test it on the matrix sizes of the Sputnik publication, Figure 10:
/// https://github.com/google-research/sputnik/tree/master/sputnik/sddmm

#include "onemkl_test.hpp"
#include <vector>
#include <exception>
#include <chrono>

AllTestParams SetupPaperTestsSpMM()
{
    AllTestParams paper_params;

    for (size_t niter : {32, 128})
    {
        paper_params.Add({1024, 1024, niter, 0.7});
        paper_params.Add({1024, 1024, niter, 0.8});
        paper_params.Add({1024, 1024, niter, 0.9});
        paper_params.Add({3072, 1024, niter, 0.7});
        paper_params.Add({3072, 1024, niter, 0.8});
        paper_params.Add({3072, 1024, niter, 0.9});
        paper_params.Add({4096, 1024, niter, 0.7});
        paper_params.Add({4096, 1024, niter, 0.8});
        paper_params.Add({4096, 1024, niter, 0.9});
        paper_params.Add({2048, 2048, niter, 0.7});
        paper_params.Add({2048, 2048, niter, 0.8});
        paper_params.Add({2048, 2048, niter, 0.9});
        paper_params.Add({6144, 2048, niter, 0.7});
        paper_params.Add({6144, 2048, niter, 0.8});
        paper_params.Add({6144, 2048, niter, 0.9});
        paper_params.Add({8192, 2048, niter, 0.7});
        paper_params.Add({8192, 2048, niter, 0.8});
        paper_params.Add({8192, 2048, niter, 0.9});
        paper_params.Add({4096, 4096, niter, 0.7});
        paper_params.Add({4096, 4096, niter, 0.8});
        paper_params.Add({4096, 4096, niter, 0.9});
        paper_params.Add({12288, 4096, niter, 0.7});
        paper_params.Add({12288, 4096, niter, 0.8});
        paper_params.Add({12288, 4096, niter, 0.9});
        paper_params.Add({16384, 4096, niter, 0.7});
        paper_params.Add({16384, 4096, niter, 0.8});
        paper_params.Add({16384, 4096, niter, 0.9});
        paper_params.Add({8192, 8192, niter, 0.7});
        paper_params.Add({8192, 8192, niter, 0.8});
        paper_params.Add({8192, 8192, niter, 0.9});
        paper_params.Add({24576, 8192, niter, 0.7});
        paper_params.Add({24576, 8192, niter, 0.8});
        paper_params.Add({24576, 8192, niter, 0.9});
        paper_params.Add({32768, 8192, niter, 0.7});
        paper_params.Add({32768, 8192, niter, 0.8});
        paper_params.Add({32768, 8192, niter, 0.9});
    }

    return paper_params;
}

AllTestParams SetupPaperTestsSDDMM()
{
    AllTestParams paper_params;

    for (size_t kiter : {32, 128})
    {
        paper_params.Add({1024, kiter, 1024, 0.7});
        paper_params.Add({1024, kiter, 1024, 0.8});
        paper_params.Add({1024, kiter, 1024, 0.9});
        paper_params.Add({3072, kiter, 1024, 0.7});
        paper_params.Add({3072, kiter, 1024, 0.8});
        paper_params.Add({3072, kiter, 1024, 0.9});
        paper_params.Add({4096, kiter, 1024, 0.7});
        paper_params.Add({4096, kiter, 1024, 0.8});
        paper_params.Add({4096, kiter, 1024, 0.9});
        paper_params.Add({2048, kiter, 2048, 0.7});
        paper_params.Add({2048, kiter, 2048, 0.8});
        paper_params.Add({2048, kiter, 2048, 0.9});
        paper_params.Add({6144, kiter, 2048, 0.7});
        paper_params.Add({6144, kiter, 2048, 0.8});
        paper_params.Add({6144, kiter, 2048, 0.9});
        paper_params.Add({8192, kiter, 2048, 0.7});
        paper_params.Add({8192, kiter, 2048, 0.8});
        paper_params.Add({8192, kiter, 2048, 0.9});
        paper_params.Add({4096, kiter, 4096, 0.7});
        paper_params.Add({4096, kiter, 4096, 0.8});
        paper_params.Add({4096, kiter, 4096, 0.9});
        paper_params.Add({12288, kiter, 4096, 0.7});
        paper_params.Add({12288, kiter, 4096, 0.8});
        paper_params.Add({12288, kiter, 4096, 0.9});
        paper_params.Add({16384, kiter, 4096, 0.7});
        paper_params.Add({16384, kiter, 4096, 0.8});
        paper_params.Add({16384, kiter, 4096, 0.9});
        paper_params.Add({8192, kiter, 8192, 0.7});
        paper_params.Add({8192, kiter, 8192, 0.8});
        paper_params.Add({8192, kiter, 8192, 0.9});
        paper_params.Add({24576, kiter, 8192, 0.7});
        paper_params.Add({24576, kiter, 8192, 0.8});
        paper_params.Add({24576, kiter, 8192, 0.9});
        paper_params.Add({32768, kiter, 8192, 0.7});
        paper_params.Add({32768, kiter, 8192, 0.8});
        paper_params.Add({32768, kiter, 8192, 0.9});
    }

    return paper_params;
}

int main(int argc, char *argv[])
{
    constexpr bool ComputeError = true;

    Tester<BlasGemm> test_dense_gemm;
    Tester<SparseBlasGemm> test_sparse_gemm; // corresponds to spmm
    Tester<FusedMM> test_fused_mm;           // corresponds to spmm

    if (argc == 1)
    {
        test_dense_gemm.Run(SetupPaperTestsSDDMM(), ComputeError);
        test_sparse_gemm.Run(SetupPaperTestsSpMM(), ComputeError);
        test_fused_mm.Run(SetupPaperTestsSpMM(), ComputeError);
    }
    else if (argc == 3)
    {
        std::string filename(argv[1]);
        const int batch_size = std::stoi(argv[2]);
        // test_dense_gemm.Run(filename, batch_size, ComputeError);
        test_sparse_gemm.Run(filename, batch_size, ComputeError);
        // test_fused_mm.Run(filename, batch_size, ComputeError);
    }
    else
        throw std::invalid_argument("Usage: ./onemkl_test.out for artificial benchmark OR ./onemkl_test.out <file> <batch_size>");

    return 0;
}
