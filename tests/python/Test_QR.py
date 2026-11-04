# Test_QR.py -- Python equivalent of tests/tamm/Test_QR.cpp
#
# Reconstruction test for tamm.qr. Builds a random distributed A (M x N), computes
# A = Q R, and checks ||A - Q R||_F / ||A||_F is small along with orthonormality of Q and the
# upper-triangularity of R.

import sys
import time

import numpy as np
import pytamm as tamm


def expects(condition, message="EXPECTS failed"):
    if not condition:
        raise AssertionError(message)


def test_qr(ec, M, N, tilesize, complex_type=False):
    sch = tamm.Scheduler(ec)
    # exhw=GPU when built with USE_CUDA/HIP/DPCPP, else CPU
    exhw = ec.exhw()

    MR = tamm.TiledIndexSpace(tamm.IndexSpace(tamm.range(M)), tilesize)
    NC = tamm.TiledIndexSpace(tamm.IndexSpace(tamm.range(N)), tilesize)

    label = "complex<double>" if complex_type else "double"
    A = tamm.TensorComplexDouble([MR, NC]) if complex_type else tamm.TensorDouble([MR, NC])
    sch.allocate(A).execute()
    tamm.random_ip(A)

    opts = tamm.QROptions()
    opts.full_matrices = True
    t1 = time.perf_counter()
    Q, R = tamm.qr(ec, A, opts, exhw)
    t2 = time.perf_counter()
    if ec.print():
        print(f"QR elapsed time: {t2 - t1:.2f} seconds")

    # Verify from the ACTUAL returned factor shapes, so this works for both full_matrices
    # (Q: M x M, R: M x N) and reduced (Q: M x K, R: K x N), and for M<N. The reconstruction
    # Qe @ Re is M x N in every case. Q's columns are orthonormal in every case.
    Ae = A.to_numpy()  # M x N
    Qe = Q.to_numpy()  # M x qc
    Re = R.to_numpy()  # rr x N
    qc = Qe.shape[1]

    # conj().T (adjoint) rather than .T: Q is unitary for complex_type, orthogonal for real,
    # and adjoint reduces to plain transpose in the real case.
    anorm = np.linalg.norm(Ae)
    rel = np.linalg.norm(Qe @ Re - Ae) / (anorm if anorm > 0 else 1.0)
    qtq = np.linalg.norm(Qe.conj().T @ Qe - np.eye(qc))
    # R's strictly-lower part should be exactly zero (upper triangular / upper trapezoidal).
    lower = np.linalg.norm(np.tril(Re, -1))

    A_reconstructed = Qe @ Re
    err = np.linalg.norm(Ae - A_reconstructed)

    if ec.print():
        print(f"QR [{label}] M={M} N={N} tile={tilesize} : ||A-QR||/||A||={rel} "
              f"||Q^HQ-I||={qtq} ||strictly-lower(R)||={lower}")
        print(f"Q Dims={Qe.shape[0]}x{Qe.shape[1]}\nR Dims={Re.shape[0]}x{Re.shape[1]}")
        print(f"Reconstruction error: {err:.3e}")

    expects(rel < 1e-8)
    expects(qtq < 1e-8)
    expects(lower < 1e-8)

    sch.deallocate(A, Q, R).execute()


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

    test_qr(ec, M, N, tile, complex_type=False)
    test_qr(ec, M, N, tile, complex_type=True)

    tamm.finalize()


if __name__ == "__main__":
    main()
