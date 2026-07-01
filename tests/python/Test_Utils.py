#!/usr/bin/env python3

import math
import faulthandler
import pytamm as pt

faulthandler.enable()


def rank0():
    try:
        return int(pt.ProcGroup.world_rank()) == 0
    except Exception:
        return True


def rprint(*args, **kwargs):
    if rank0():
        print(*args, **kwargs)


def test_utils(sch, ex_hw):
    N = 10
    tilesize = 5

    ec = sch.ec()

    # -------------------------------------------------------------------------
    # get_scalar
    # -------------------------------------------------------------------------
    scalar_t = pt.TensorDouble()
    scalar_ct = pt.TensorComplexDouble()

    sch.allocate(scalar_t, scalar_ct).execute()

    val_t = pt.get_scalar(scalar_t)
    val_ct = pt.get_scalar(scalar_ct)

    sch.deallocate(scalar_t, scalar_ct).execute()

    # -------------------------------------------------------------------------
    # TiledIndexSpace and labels
    # -------------------------------------------------------------------------
    tis1 = pt.TiledIndexSpace(pt.IndexSpace(pt.range(N)), tilesize)
    i, j, k, l, m, o = tis1.labels("all", count=6)

    # -------------------------------------------------------------------------
    # identity_matrix
    #
    # Match C++ exactly: schedule deallocation, but do not execute it until the
    # next scheduler execute.
    # -------------------------------------------------------------------------
    imat = pt.identity_matrix(ec, tis1)
    sch.deallocate(imat)

    # -------------------------------------------------------------------------
    # A complex, B real, C complex
    # -------------------------------------------------------------------------
    A = pt.TensorComplexDouble([i, j, m, o])
    B = pt.TensorDouble([m, o, k, l])
    C = pt.TensorComplexDouble([i, j, k, l])

    sch.allocate(A, B, C)
    sch(A(), "=", 21.0)
    sch(B(), "=", 2.0)
    sch(C(), "=", 0.0)
    sch.execute()

    # C = C + A x B
    sch(C(j, i, k, l), "+=", A(i, j, m, o) * B(m, o, k, l))
    sch.execute(ex_hw)

    # Keep the ProcGroup object alive for the dense ExecutionContext.
    dense_pg = ec.pg()
    ec_dense = pt.ExecutionContext(
        dense_pg,
        pt.DistributionKind.dense,
        pt.MemoryManagerKind.ga,
    )

    # -------------------------------------------------------------------------
    # Operations only on 2D tensors
    # -------------------------------------------------------------------------
    ctens = pt.TensorComplexDouble([i, j])
    rtens = pt.TensorDouble([i, j])

    sch.allocate(ctens, rtens).execute()

    pt.random_ip(rtens)

    # trace
    rtrace = pt.trace(rtens)
    ctrace = pt.trace(ctens)

    # trace_sqr
    rtrace_sqr = pt.trace_sqr(rtens)
    ctrace_sqr = pt.trace_sqr(ctens)

    # diagonal
    rtens_diag = pt.diagonal(rtens)
    ctens_diag = pt.diagonal(ctens)

    # update_diagonal
    pt.update_diagonal(rtens, rtens_diag)
    pt.update_diagonal(ctens, ctens_diag)

    sch.deallocate(ctens, rtens).execute()

    # -------------------------------------------------------------------------
    # sum
    # -------------------------------------------------------------------------
    b_sum = pt.sum(B)
    c_sum = pt.sum(C)

    # -------------------------------------------------------------------------
    # norm
    # -------------------------------------------------------------------------
    b_norm = pt.norm(B)
    c_norm = pt.norm(C)

    # -------------------------------------------------------------------------
    # linf_norm
    # -------------------------------------------------------------------------
    b_linf_norm = pt.linf_norm(B)
    c_linf_norm = pt.linf_norm(C)

    # -------------------------------------------------------------------------
    # sqrt
    # -------------------------------------------------------------------------
    b_sqrt = pt.sqrt(B)
    c_sqrt = pt.sqrt(C)

    sch.deallocate(b_sqrt, c_sqrt).execute()

    # -------------------------------------------------------------------------
    # square
    # -------------------------------------------------------------------------
    b_square = pt.square(B)
    c_square = pt.square(C)

    sch.deallocate(b_square, c_square).execute()

    # -------------------------------------------------------------------------
    # conj, conj_ip
    # -------------------------------------------------------------------------
    c_conj = pt.conj(C)
    pt.conj_ip(c_conj)

    sch.deallocate(c_conj).execute()

    # -------------------------------------------------------------------------
    # pow
    # -------------------------------------------------------------------------
    b_pow = pt.pow(B, 2.0)
    c_pow = pt.pow(C, 2.0 + 0.0j)

    sch.deallocate(b_pow, c_pow).execute()

    # -------------------------------------------------------------------------
    # log
    # -------------------------------------------------------------------------
    b_log = pt.log(B)
    c_log = pt.log(C)

    sch.deallocate(b_log, c_log).execute()

    # -------------------------------------------------------------------------
    # log10
    # -------------------------------------------------------------------------
    b_log10 = pt.log10(B)
    c_log10 = pt.log10(C)

    sch.deallocate(b_log10, c_log10).execute()

    # -------------------------------------------------------------------------
    # einverse
    # -------------------------------------------------------------------------
    b_einverse = pt.einverse(B)
    c_einverse = pt.einverse(C)

    sch.deallocate(b_einverse, c_einverse).execute()

    # -------------------------------------------------------------------------
    # scale, scale_ip
    # -------------------------------------------------------------------------
    b_scale = pt.scale(B, 2.0)
    c_scale = pt.scale(C, 2.0 + 0.0j)

    pt.scale_ip(b_scale, 2.0)
    pt.scale_ip(c_scale, 2.0 + 0.0j)

    sch.deallocate(b_scale, c_scale).execute()

    # -------------------------------------------------------------------------
    # random_ip
    # -------------------------------------------------------------------------
    pt.random_ip(A)
    pt.random_ip(B)

    # -------------------------------------------------------------------------
    # max_element, min_element
    # -------------------------------------------------------------------------
    bmaxval, bmaxblockid, bmaxcoord = pt.max_element(B)
    bminval, bminblockid, bmincoord = pt.min_element(B)

    # -------------------------------------------------------------------------
    # update_tensor_val
    # -------------------------------------------------------------------------
    val_t = 42.0
    val_ct = 2.0 + 3.0j
    t_coord = [1, 2, 4, 1]

    pt.update_tensor_val(B, t_coord, val_t)
    pt.update_tensor_val(C, t_coord, val_ct)

    # -------------------------------------------------------------------------
    # hash_tensor
    # -------------------------------------------------------------------------
    b_hash = pt.hash_tensor(B)
    c_hash = pt.hash_tensor(C)

    # -------------------------------------------------------------------------
    # permute_tensor
    # -------------------------------------------------------------------------
    b_perm = pt.permute_tensor(B, [1, 3, 2, 0])
    c_perm = pt.permute_tensor(C, [1, 3, 2, 0])

    sch.deallocate(b_perm, c_perm).execute()

    # -------------------------------------------------------------------------
    # to_dense_tensor
    # -------------------------------------------------------------------------
    b_dens = pt.to_dense_tensor(ec_dense, B)
    c_dens = pt.to_dense_tensor(ec_dense, C)

    # -------------------------------------------------------------------------
    # get_tensor_element
    # -------------------------------------------------------------------------
    b_dens_val = pt.get_tensor_element(b_dens, [0, 0, 0, 0])
    c_dens_val = pt.get_tensor_element(c_dens, [0, 0, 0, 0])

    # -------------------------------------------------------------------------
    # tensor_block
    # -------------------------------------------------------------------------
    b_block = pt.tensor_block(
        b_dens,
        [0, 0, 0, 0],
        [N // 2, N // 2, N // 2, N // 2],
    )

    c_block = pt.tensor_block(
        c_dens,
        [0, 0, 0, 0],
        [N // 2, N // 2, N // 2, N // 2],
    )

    # -------------------------------------------------------------------------
    # local_buf_size and access_local_buf checks
    # -------------------------------------------------------------------------
    dense_pg_for_check = ec_dense.pg()
    pg_size = int(dense_pg_for_check.size())
    pg_rank = int(dense_pg_for_check.rank())

    expected_tile_blocks = int(math.pow(N // tilesize, b_dens.num_modes()))

    if expected_tile_blocks % pg_size == 0:
        expected_local_size = int(math.pow(N, b_dens.num_modes()) / pg_size)
        actual_local_size = int(b_dens.local_buf_size())

        assert expected_local_size == actual_local_size, (
            "local_buf_size mismatch",
            expected_local_size,
            actual_local_size,
        )

    if pg_rank == 0:
        local_buf = b_dens.access_local_buf()

        assert local_buf is not None, "rank 0 dense tensor local buffer is None"
        assert len(local_buf) > 0, "rank 0 dense tensor local buffer is empty"

        ref = pt.get_tensor_element(b_dens, [0, 0, 0, 0])

        assert float(local_buf[0]) == float(ref), (
            "access_local_buf first element mismatch",
            float(local_buf[0]),
            float(ref),
        )

    sch.deallocate(b_dens, c_dens).execute()
    sch.deallocate(b_block, c_block).execute()

    # -------------------------------------------------------------------------
    # redistribute_tensor
    # -------------------------------------------------------------------------
    tis_red = pt.TiledIndexSpace(pt.IndexSpace(pt.range(N)), N // 2)
    tis_red_vec = [tis_red, tis_red, tis_red, tis_red]

    b_red = pt.redistribute_tensor(B, tis_red_vec)
    c_red = pt.redistribute_tensor(C, tis_red_vec)

    sch.deallocate(b_red, c_red).execute()

    # -------------------------------------------------------------------------
    # retile_tamm_tensor
    # -------------------------------------------------------------------------
    b_ret = pt.TensorDouble([tis_red, tis_red, tis_red, tis_red])
    c_ret = pt.TensorComplexDouble([tis_red, tis_red, tis_red, tis_red])

    sch.allocate(b_ret, c_ret).execute()

    pt.retile_tamm_tensor(B, b_ret)
    pt.retile_tamm_tensor(C, c_ret)

    sch.deallocate(b_ret, c_ret).execute()

    # -------------------------------------------------------------------------
    # Final deallocation
    # -------------------------------------------------------------------------
    sch.deallocate(A, B, C).execute()


def main():
    pt.initialize(["Test_Utils.py"], False)

    try:
        pg = pt.ProcGroup.create_world_coll()

        ec = pt.ExecutionContext(
            pg,
            pt.DistributionKind.nw,
            pt.MemoryManagerKind.ga,
        )

        ex_hw = ec.exhw()
        sch = pt.Scheduler(ec)

        rprint("[ RUN      ] Test_Utils")
        test_utils(sch, ex_hw)
        rprint("[       OK ] Test_Utils")

    finally:
        pt.finalize(True)


if __name__ == "__main__":
    main()