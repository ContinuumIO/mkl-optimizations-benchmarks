# References:
#
# http://software.intel.com/en-us/intel-mkl
# https://code.google.com/p/numexpr/wiki/NumexprVML

from __future__ import print_function
import sys
import numpy as np
import numexpr as ne
import time
import gc
import os.path

data_dir = './'


def time_dgemm(N=100, trials=3, dtype=np.double):
    A = np.asarray(np.random.rand(N, N), dtype=dtype)
    B = np.asarray(np.random.rand(N, N), dtype=dtype)
    gcold = gc.isenabled()
    gc.disable()
    tic = time.time()
    for i in range(trials):
        A.dot(B)
    toc = time.time()-tic
    if gcold:
        gc.enable()
    return toc/trials, 2*N*N*N*1e-9


def time_cholesky(N=100, trials=3, dtype=np.double):
    A = np.asarray(np.random.rand(N, N), dtype=dtype)
    A = A*A.transpose() + N*np.eye(N)
    gcold = gc.isenabled()
    gc.disable()
    tic = time.time()
    for i in range(trials):
        np.linalg.cholesky(A)
    toc = time.time()-tic
    if gcold:
        gc.enable()
    return toc/trials, N*N*N/3.0*1e-9

def time_svd(N=100, trials=3, dtype=np.double, **kwargs):
    np.random.seed(0)

    scale = kwargs.get('scale', 1)

    # try a few M, that depend on N
    M = int(N * scale)
    A = np.asarray(np.random.rand(M, N), dtype=dtype)
    gcold = gc.isenabled()
    gc.disable()
    tic = time.time()
    for i in range(trials):
        u, s, vt = np.linalg.svd(A)
    toc = time.time()-tic
    if gcold:
        gc.enable()
    # complexity estimate is v. crude and assumes all sort of things!
    return toc/trials, min(N*M*M, M*N*N)*1e-9

def time_numexpr(N=100, trials=3, dtype=np.double):
    # these are used, despite what a linter may say.  Numexpr uses them
    x = np.asarray(np.linspace(-1, 1, N), dtype=dtype)
    y = np.asarray(np.linspace(-1, 1, N), dtype=dtype)   # NOQA
    z = np.empty_like(x)
    gcold = gc.isenabled()
    gc.disable()
    tic = time.time()
    for i in range(trials):
        ne.evaluate('2*y+4*x', out=z)
    toc = time.time()-tic
    if gcold:
        gc.enable()
    return (toc/trials, dtype().itemsize*3*N*1e-9)


def time_fft(size, repeat, dtype=np.double):
    size = int(size)
    # naive radix approach: https://stackoverflow.com/a/42261160/1170370
    # 3 n log(n) additions, 2 n log(n) multiplies
    gflop = 5.0 * (size ** 2) * np.log2((size**2)) / 1.0E9
    array_size = size, size

    a = np.random.randn(*array_size) + 1j * np.random.randn(*array_size)
    a = a.astype(np.complex64)

    start_time = time.time()
    for i in range(repeat):
        np.fft.fftn(a)
    toc = time.time() - start_time

    return toc/repeat, gflop


def test_timers():
    N = 512
    trials = 3
    dtype = np.float64
    s, gflop = time_dgemm(N, trials, dtype)
    print("DGEMM   : N: %d s: %e GFLOP/s: %e" % (N, s, gflop/s))
    s, gflop = time_cholesky(N, trials, dtype)
    print("Cholesky: N: %d s: %e GFLOP/s: %e" % (N, s, gflop/s))
    s, gflop = time_svd(N, trials, dtype, scale=0.5)
    print("SVD (under)     : N: %d s: %e GFLOP/s: %e" % (N, s, gflop/s))
    s, gflop = time_svd(N, trials, dtype, scale=2)
    print("SVD (over)     : N: %d s: %e GFLOP/s: %e" % (N, s, gflop/s))
    s, gbyte = time_numexpr(50000, trials, dtype)
    print("NumExpr : N: %d s: %e GBytes/s: %e" % (N, s, gbyte/s))
    s, gflop = time_fft(512, trials, dtype)
    print("FFT     : N: %d s: %e GFLOP/s: %e" % (N, s, gflop/s))


def bench(test_fun, Ns, trials, dtype=None, **kwargs):
    data = np.empty((len(Ns), 2))
    print("%d tests" % len(Ns))
    tic = time.time()
    for i, N in enumerate(Ns):
        N = int(N)
        sys.stdout.write('.')
        sys.stdout.flush()
        if dtype is not None:
            out_tuple = test_fun(N, trials, dtype, **kwargs)
        else:
            out_tuple = test_fun(N, trials, **kwargs)

        if len(out_tuple) > 1:
            data[i, :] = (N, out_tuple[1]/out_tuple[0])
        else:
            data[i, :] = (N, out_tuple[0])
    print('done')
    toc = time.time() - tic
    print('tests took: %e seconds' % toc)
    return data


def dump_data(data, data_dir, backend, algo):
    filename = backend + '-' + algo + '.csv'
    out_file = os.path.join(data_dir, filename)
    np.savetxt(out_file, data, delimiter=', ')


if __name__ == '__main__':
    backend = sys.argv[1]
    print("checking timers...")
    test_timers()
    logNs = np.arange(6, 13.5, 0.5)  # uncomment to run the big stuff
    #    logNs = np.arange(3,7,0.5) # uncomment to run quick tests
    Ns = np.asarray(np.exp2(logNs), dtype=np.int64)
    trials = 5
    dtype = np.float64

    print('benchmarking DGEMM')
    dgemm_data = bench(time_dgemm, Ns, trials, dtype)
    dump_data(dgemm_data, data_dir, backend, 'DGEMM')

    print('benchmarking Cholesky')
    cholesky_data = bench(time_cholesky, Ns, trials, dtype)
    dump_data(cholesky_data, data_dir, backend, 'Cholesky')

    print('benchmarking SVD')
    Ns = 2 ** np.arange(4, 12)
    svd_data = bench(time_svd, Ns, trials, dtype, scale=0.5)
    dump_data(svd_data, data_dir, backend, 'SVD_under')
    svd_data = bench(time_svd, Ns, trials, dtype, scale=2)
    dump_data(svd_data, data_dir, backend, 'SVD_over')

    print('benchmarking NumExpr')
    logNs = np.arange(12, 18.5, 0.5)  # uncomment to run big tests
    #    logNs = np.arange(6,13.5,0.5) # uncomment to run quick tests
    Ns = np.exp2(logNs)
    numexpr_data = bench(time_numexpr, Ns, trials, dtype)
    dump_data(numexpr_data, data_dir, backend, 'NumExpr')

    print('benchmarking fft')
    Ns = 2 ** np.arange(4, 14)
    fft_data = bench(time_fft, Ns, trials, dtype)
    dump_data(fft_data, data_dir, backend, 'fft')
