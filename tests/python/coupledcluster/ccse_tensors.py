#!/usr/bin/env python3

import pytamm as tamm


class CCSE_Tensors:
    """
    Python translation of CCSE_Tensors<T> for Tensor<double>.

    Usage examples:
      cctens = CCSE_Tensors(MO, [V, O], "tensor_name", ["aa", "bb"])
      cctens = CCSE_Tensors(MO, [V, O, CI], "tensor_name", ["aa", "bb"])
      cctens = CCSE_Tensors(MO, [V, O, V, O], "tensor_name",
                            ["aaaa", "baba", "baab", "bbbb"])
    """

    def __init__(self, MO=None, tis=None, tensor_name="", blocks=None):
        self.tmap = {}
        self.allocated_tensors = []
        self.tname = tensor_name
        self.is_mo_3d = False
        self.vblocks = []

        if MO is None:
            return

        if tis is None:
            tis = []

        if blocks is None:
            blocks = []

        self.vblocks = list(blocks)

        ndims = len(tis)
        err_msg = f"Error in tensor [{self.tname}] declaration"

        if ndims < 2 or ndims > 4:
            raise RuntimeError(err_msg + ": Only 2,3,4D tensors are allowed")

        self.is_mo_3d = True

        O = MO("occ")
        V = MO("virt")

        for x in tis:
            if x != O and x != V:
                if ndims == 3:
                    self.is_mo_3d = False
                else:
                    raise RuntimeError(err_msg + ": Only O,V tiled index spaces can be specified")

        allowed_blocks = ["aa", "bb"]
        if ndims == 3 and self.is_mo_3d:
            allowed_blocks = ["aaa", "baa", "abb", "bbb"]
        elif ndims == 4:
            allowed_blocks = ["aaaa", "abab", "bbbb", "abba", "baab", "baba"]

        if len(blocks) == 0:
            raise RuntimeError(err_msg + ": Please specify the tensor blocks to be allocated")

        for x in blocks:
            if x not in allowed_blocks:
                if ndims == 2 or (ndims == 3 and not self.is_mo_3d):
                    raise RuntimeError(
                        err_msg + f": Invalid block [{x}] specified, allowed blocks are [aa|bb]"
                    )
                elif ndims == 3 and self.is_mo_3d:
                    raise RuntimeError(
                        err_msg
                        + f": Invalid block [{x}] specified, allowed blocks are [aaa|baa|abb|bbb]"
                    )
                else:
                    raise RuntimeError(
                        err_msg
                        + f": Invalid block [{x}] specified, allowed blocks are "
                        "[aaaa|abab|bbbb|abba|baab|baba]"
                    )

        if ndims == 2 or (ndims == 3 and not self.is_mo_3d):
            if "aa" in blocks:
                aa = tamm.TensorDouble(self.construct_tis(MO, tis, [0, 0]))
                self.tmap["aa"] = aa
                self.allocated_tensors.append(aa)

            if "bb" in blocks:
                bb = tamm.TensorDouble(self.construct_tis(MO, tis, [1, 1]))
                self.tmap["bb"] = bb
                self.allocated_tensors.append(bb)

        elif ndims == 3 and self.is_mo_3d:
            if "aaa" in blocks:
                aaa = tamm.TensorDouble(self.construct_tis(MO, tis, [0, 0, 0]))
                self.tmap["aaa"] = aaa
                self.allocated_tensors.append(aaa)

            if "baa" in blocks:
                baa = tamm.TensorDouble(self.construct_tis(MO, tis, [1, 0, 0]))
                self.tmap["baa"] = baa
                self.allocated_tensors.append(baa)

            if "abb" in blocks:
                abb = tamm.TensorDouble(self.construct_tis(MO, tis, [0, 1, 1]))
                self.tmap["abb"] = abb
                self.allocated_tensors.append(abb)

            if "bbb" in blocks:
                bbb = tamm.TensorDouble(self.construct_tis(MO, tis, [1, 1, 1]))
                self.tmap["bbb"] = bbb
                self.allocated_tensors.append(bbb)

        else:
            if "aaaa" in blocks:
                aaaa = tamm.TensorDouble(self.construct_tis(MO, tis, [0, 0, 0, 0]))
                self.tmap["aaaa"] = aaaa
                self.allocated_tensors.append(aaaa)

            if "abab" in blocks:
                abab = tamm.TensorDouble(self.construct_tis(MO, tis, [0, 1, 0, 1]))
                self.tmap["abab"] = abab
                self.allocated_tensors.append(abab)

            if "bbbb" in blocks:
                bbbb = tamm.TensorDouble(self.construct_tis(MO, tis, [1, 1, 1, 1]))
                self.tmap["bbbb"] = bbbb
                self.allocated_tensors.append(bbbb)

            if "abba" in blocks:
                abba = tamm.TensorDouble(self.construct_tis(MO, tis, [0, 1, 1, 0]))
                self.tmap["abba"] = abba
                self.allocated_tensors.append(abba)

            if "baab" in blocks:
                baab = tamm.TensorDouble(self.construct_tis(MO, tis, [1, 0, 0, 1]))
                self.tmap["baab"] = baab
                self.allocated_tensors.append(baab)

            if "baba" in blocks:
                baba = tamm.TensorDouble(self.construct_tis(MO, tis, [1, 0, 1, 0]))
                self.tmap["baba"] = baba
                self.allocated_tensors.append(baba)

    def __call__(self, block):
        if block not in self.tmap:
            raise RuntimeError(
                f"Error: tensor [{self.tname}]: block [{block}] requested does not exist"
            )
        return self.tmap[block]

    def construct_tis(self, MO, tis, btype):
        ndims = len(tis)

        O = MO("occ")
        V = MO("virt")

        o_alpha = MO("occ_alpha")
        o_beta = MO("occ_beta")
        v_alpha = MO("virt_alpha")
        v_beta = MO("virt_beta")

        btis = []

        for x in range(ndims):
            if tis[x] == O:
                btis.append(o_alpha if btype[x] == 0 else o_beta)
            elif tis[x] == V:
                btis.append(v_alpha if btype[x] == 0 else v_beta)
            elif ndims == 3 and not self.is_mo_3d:
                btis.append(tis[x])

        return btis

    def allocate(self, ec):
        sch = tamm.Scheduler(ec)
        for x in self.allocated_tensors:
            sch.allocate(x)
        sch.execute()

    def deallocate(self):
        if len(self.allocated_tensors) == 0:
            return

        ec = self.allocated_tensors[0].execution_context()
        if ec is None:
            return

        sch = tamm.Scheduler(ec)
        for x in self.allocated_tensors:
            sch.deallocate(x)
        sch.execute()

    def sum_tensor_sizes(self):
        total_size = 0.0
        for x in self.allocated_tensors:
            total_size += (float(x.size()) * 8.0) / (1024.0 * 1024.0 * 1024.0)
        return total_size

    @staticmethod
    def allocate_list(sch, *ccsets):
        for ccset in ccsets:
            for x in ccset.allocated_tensors:
                sch.allocate(x)

    @staticmethod
    def deallocate_list(sch, *ccsets):
        for ccset in ccsets:
            for x in ccset.allocated_tensors:
                sch.deallocate(x)

    @staticmethod
    def sum_tensor_sizes_list(*ccsets):
        return sum(x.sum_tensor_sizes() for x in ccsets)

    @staticmethod
    def copy(sch, src, dest, update=False):
        for x in src.vblocks:
            if update:
                sch(dest(x)(), "+=", src(x)())
            else:
                sch(dest(x)(), "=", src(x)())

    @staticmethod
    def initialize(sch, value, *ccsets):
        for ccset in ccsets:
            for x in ccset.vblocks:
                sch(ccset(x)(), "=", value)