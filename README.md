# Educational-BLAS
This is an educational implementation of BLAS which utilizes HPC techniques to (ideally) speed up computations.

List of implementations (as of now):
1. Basic BLAS core: `axpy`, `scal`, `gemv`, etc.
2. Basic parallel BLAS core: utilized OpenMP to parallelize for loops

Planned implementations:
1. Cache optimized BLAS
2. SIMD vectorization + multithreaded BLAS
3. Mini-HPC kernel BLAS
