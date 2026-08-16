# Test_SVD.py -- Python equivalent of tests/tamm/Test_SVD.cpp
#
# Reconstruction test for tamm.svd. Builds a random distributed A (M x N), computes
# A = U diag(S) Vh, and checks ||A - U diag(S) Vh||_F / ||A||_F is small along with
# orthonormality of U and Vh.

import sys
import time

import numpy as np
import pytamm as tamm


def expects(condition, message="EXPECTS failed"):
    if not condition:
        raise AssertionError(message)


def test_svd(ec, M, N, tilesize, complex_type=False):
    sch = tamm.Scheduler(ec)
    # exhw=GPU when built with USE_CUDA/HIP/DPCPP, else CPU
    exhw = ec.exhw()

    # expects(M >= N)

    MR = tamm.TiledIndexSpace(tamm.IndexSpace(tamm.range(M)), tilesize)
    NC = tamm.TiledIndexSpace(tamm.IndexSpace(tamm.range(N)), tilesize)

    label = "complex<double>" if complex_type else "double"
    A = tamm.TensorComplexDouble([MR, NC]) if complex_type else tamm.TensorDouble([MR, NC])
    sch.allocate(A).execute()
    tamm.random_ip(A)

    opts = tamm.SVDOptions()
    opts.full_matrices = True
    t1 = time.perf_counter()
    U, S, Vh = tamm.svd(ec, A, opts, exhw)
    t2 = time.perf_counter()
    if ec.print():
        print(f"SVD elapsed time: {t2 - t1:.2f} seconds")
    # S is always real-valued, even for complex_type, matching tamm::svd/LAPACK's gesvd.
    S = np.asarray(S, dtype=np.float64)

    # Verify from the ACTUAL returned factor shapes, so this works for both full_matrices
    # (U: M x M, Vh: N x N) and reduced (U: M x K, Vh: K x N), and for M<N. Sigma is uc x vr with
    # the min(uc, vr) singular values on the diagonal; the reconstruction Ue @ Sigma @ Ve is then
    # M x N in every case. U's columns and Vh's rows are orthonormal in every case.
    Ae = A.to_numpy()   # M x N
    Ue = U.to_numpy()   # M x uc
    Ve = Vh.to_numpy()  # vr x N
    uc = Ue.shape[1]
    vr = Ve.shape[0]
    Sd = np.zeros((uc, vr), dtype=Ae.dtype)
    Kd = min(uc, vr)
    for k in range(Kd):
        Sd[k, k] = S[k]

    # conj().T (adjoint) rather than .T: U/Vh are unitary for complex_type, orthogonal for
    # real, and adjoint reduces to plain transpose in the real case.
    anorm = np.linalg.norm(Ae)
    rel = np.linalg.norm(Ue @ Sd @ Ve - Ae) / (anorm if anorm > 0 else 1.0)
    utu = np.linalg.norm(Ue.conj().T @ Ue - np.eye(uc))
    vvt = np.linalg.norm(Ve @ Ve.conj().T - np.eye(vr))

    A_reconstructed = Ue @ Sd @ Ve
    err = np.linalg.norm(Ae - A_reconstructed)

    if ec.print():
        print(f"SVD [{label}] M={M} N={N} tile={tilesize} : ||A-USVh||/||A||={rel} "
              f"||U^HU-I||={utu} ||VhVh^H-I||={vvt}")
        print(f"U Dims={Ue.shape[0]}x{Ue.shape[1]}\nVt Dims={Ve.shape[0]}x{Ve.shape[1]}")
        print(f"Reconstruction error: {err:.3e}")

    expects(rel < 1e-8)
    expects(utu < 1e-8)
    expects(vvt < 1e-8)

    sch.deallocate(A, U, Vh).execute()


def main(argv=None):
    if argv is None:
        argv = sys.argv

    tamm.initialize(argv)
    pg = tamm.ProcGroup.create_world_coll()
    ec = tamm.ExecutionContext(pg, tamm.DistributionKind.nw, tamm.MemoryManagerKind.ga)

    M, N = 100, 20
    tile = 50
    if len(argv) >= 3:
        M = int(argv[1])
        N = int(argv[2])
    if len(argv) >= 4:
        tile = int(argv[3])

    test_svd(ec, M, N, tile, complex_type=False)
    test_svd(ec, M, N, tile, complex_type=True)

    tamm.finalize()


if __name__ == "__main__":
    main()
