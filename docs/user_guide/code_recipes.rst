Code Recipes
=================

CCSD Singles
-------------

.. code-block:: cpp

  template<typename T>
  void ccsd_t1(Scheduler& sch, const TiledIndexSpace& MO, Tensor<T>& i0, const Tensor<T>& t1,
              const Tensor<T>& t2, const Tensor<T>& f1, const Tensor<T>& v2) {

    const TiledIndexSpace& O = MO("occ");
    const TiledIndexSpace& V = MO("virt");

    Tensor<T> t1_2_1{{O, O}, {1, 1}};
    Tensor<T> t1_2_2_1{{O, V}, {1, 1}};
    Tensor<T> t1_3_1{{V, V}, {1, 1}};
    Tensor<T> t1_5_1{{O, V}, {1, 1}};
    Tensor<T> t1_6_1{{O, O, O, V}, {2, 2}};

    TiledIndexLabel p2, p3, p4, p5, p6, p7;
    TiledIndexLabel h1, h4, h5, h6, h7, h8;

    std::tie(p2, p3, p4, p5, p6, p7) = MO.labels<6>("virt");
    std::tie(h1, h4, h5, h6, h7, h8) = MO.labels<6>("occ");

    sch
      .allocate(t1_2_1, t1_2_2_1, t1_3_1, t1_5_1, t1_6_1)
      ( t1_2_1(h7, h1)       = 0)
      ( t1_3_1(p2, p3)       = 0)
      ( i0(p2,h1)            =        f1(p2,h1))
      ( t1_2_1(h7,h1)        =        f1(h7,h1))
      ( t1_2_2_1(h7,p3)      =        f1(h7,p3))
      ( t1_2_2_1(h7,p3)     += -1   * t1(p5,h6)       * v2(h6,h7,p3,p5))
      ( t1_2_1(h7,h1)       +=        t1(p3,h1)       * t1_2_2_1(h7,p3))
      ( t1_2_1(h7,h1)       += -1   * t1(p4,h5)       * v2(h5,h7,h1,p4))
      ( t1_2_1(h7,h1)       += -0.5 * t2(p3,p4,h1,h5) * v2(h5,h7,p3,p4))
      ( i0(p2,h1)           += -1   * t1(p2,h7)       * t1_2_1(h7,h1))
      ( t1_3_1(p2,p3)        =        f1(p2,p3))
      ( t1_3_1(p2,p3)       += -1   * t1(p4,h5)       * v2(h5,p2,p3,p4))
      ( i0(p2,h1)           +=        t1(p3,h1)       * t1_3_1(p2,p3))
      ( i0(p2,h1)           += -1   * t1(p3,h4)       * v2(h4,p2,h1,p3))
      ( t1_5_1(h8,p7)        =        f1(h8,p7))
      ( t1_5_1(h8,p7)       +=        t1(p5,h6)       * v2(h6,h8,p5,p7))
      ( i0(p2,h1)           +=        t2(p2,p7,h1,h8) * t1_5_1(h8,p7))
      ( t1_6_1(h4,h5,h1,p3)  =        v2(h4,h5,h1,p3))
      ( t1_6_1(h4,h5,h1,p3) += -1   * t1(p6,h1)       * v2(h4,h5,p3,p6))
      ( i0(p2,h1)           += -0.5 * t2(p2,p3,h4,h5) * t1_6_1(h4,h5,h1,p3))
      ( i0(p2,h1)           += -0.5 * t2(p3,p4,h1,h5) * v2(h5,p2,p3,p4))
    .deallocate(t1_2_1, t1_2_2_1, t1_3_1, t1_5_1, t1_6_1);
  }


View Tensor Construction
------------------------

View tensor constructs a lambda tensor that can access to the reference tensors using get and put lambda methods along with a block id translation lambda. Main usage of view tensors is for defining different shaped tensors that can use a reference tensor for any operations. Below example gives the view tensor usage for constructing a tensor with different constraints on the `Sijkl` tensor. The constructed `Sijki` tensor that has the values from `Sijkl` if the first index of the first pair matches the second index of the second pair. For this case there is no need for translation of the blocks as well as the put method as the tensor is used read-only.

.. code-block:: cpp

    // Translate lambda for translating BlockId to reference tensor
    auto no_translate = [](IndexVector blockid) -> IndexVector {
      return blockid;
    };
    
    // Put lambda for specifying how to do a put operation to the reference tensor
    auto put_copy_func = [](const BlockSpan<T>& in, BlockSpan<T>& out, const IndexVector& blockid) -> void {};
    
    // Get lambda for specifying how to do a get operation from the reference tensor
    auto get_S_is_si = [=](const BlockSpan<T> &from_span, BlockSpan<T> &to_span,
                           const IndexVector &blockid) {
      auto is_tid = blockid[0];
      auto si_tid = blockid[1];
      auto a_is_tid = blockid[2];
      auto b_si_tid = blockid[3];
      auto is_offset = LMOP.tile_offset(is_tid);
      auto si_offset = LMOP.tile_offset(si_tid);
      auto is_tsize = LMOP.tile_size(is_tid);
      auto si_tsize = LMOP.tile_size(si_tid);
      auto a_is_tsize = PNO.tile_size(a_is_tid);
      auto b_si_tsize = PNO.tile_size(b_si_tid);

      auto *to_buf = to_span.buf();
      auto *from_buf = from_span.buf();

      for (int is = 0, to_idx = 0; is < is_tsize; is++) {
        auto [i0, s0] = lmo_pairs[is_offset + is];
        for (int si = 0; si < si_tsize; si++) {
          auto [s1, i1] = lmo_pairs[si_offset + si];
          for (int a_is = 0; a_is < a_is_tsize; a_is++) {
            for (int b_si = 0; b_si < b_si_tsize; b_si++, to_idx++) {
              if (i0 != i1) {
                to_buf[to_idx] = 0;
              } else {
                to_buf[to_idx] = from_buf[to_idx];
              }
            }
          }
        }
      }
    };

    // Tensor(Tensor<T> ref_tensor, IndexLabelVec t_labels, MapFunc translate_func, CopyFunc get_copy, CopyFunc put_copy)
    Tensor<T> Sijki{Sijkl, {is, si, a_is, a_si}, no_translate, get_S_is_si, put_copy_func};


Unit Tiled View Tensor Construction
-----------------------------------

As a specialized view tensor, TAMM also provides unit tiled views of tensors that are build with tiled index spaces. This is especially useful to be used on sliced updates on specific dimension/s of a tensor that is already tiled for efficiency. Due to the distribution of the tensor blocks, you can only have unit tiled views for the consecutive dimensions from the left. Below example gives some usage details for the unit tiled view tensors. Users can chose to unit tile the whole tensor as in Example 2 or chose number dimensions from left to be unit tiled. As these tensors are view tensors, there is no need to do allocation for unit tiled view tensors, they simply use the storage from the reference tensors.


.. code-block:: cpp
  
  // Example 1
  // Construct MO Index Space
  const IndexSpace MO_IS{
    range(0, 50),
      {
        {"occ",  {range(0, 15)}},
        {"virt", {range(15, 50)}}
      }
  };
  // Use a set of specialized tiles for tiling
  const std::vector<Tile> mo_tiles = {10,5,10,10,10,5};
  const TiledIndexSpace MO{MO_IS, mo_tiles};

  // Construct tensors on tiled index spaces
  auto [h1,h2,h3] = MO.labels<3>("occ");
  Tensor<double> t1{h1,h2};
  Tensor<double> t2{h1,h2};
  Tensor<double> tmp{h1};
  
  // Allocate tensors
  sch.allocate(t1,t2,tmp).execute();

  // Construct a unit tiled view for t1 for only left-most dimension
  Tensor<double> t1_ut{t1,1};
  // Get the TiledIndexSpace and the labels from the unit tiled dimension
  TiledIndexSpace t1_utis{t1_ut.tiled_index_spaces()[0],range(2,3)};
  auto t1_ut_l1 = t1_utis.label();

  // Do computation over the unit tiled view of t1
  sch(tmp(h3) = t1_ut(t1_ut_l1,h2) * t2(h2,h3)).execute();
  
  // Example 2 
  // Construct specialized tiles for AO tiled space
  const std::vector<Tile> ao_tiles = {1,3};
  TiledIndexSpace AO{IndexSpace{range(4)}, ao_tiles};

  // Construct T on tiled AO space
  Tensor<double> T{AO, AO};
  // Allocate tensor
  sch.allocate(T).execute();
  // Fill tensor random values
  random_ip(T);

  print_tensor(T);

  // Construct unit tiled view of the full T tensor (both dimensions)
  Tensor<double> T_ut{T, 2};

  // Construct and allocate a scalar tensor
  Tensor<double> tmp2{};
  sch.allocate(tmp2).execute();

  // Loop over AO slices
  for(int i = 0; i < 4; i++) {
    for(int j = 0; j < 4; j++) {
      // Get unit tiled index spaces for i and j slice
      TiledIndexSpace tis1{T_ut.tiled_index_spaces()[0], range(i, i + 1)};
      TiledIndexSpace tis2{T_ut.tiled_index_spaces()[1], range(j, j + 1)};
      // Construct labels
      auto l1 = tis1.label();
      auto l2 = tis2.label();
      // Get each value from the unit tiled view tensor
      sch(tmp2() = T_ut(l1, l2)).execute();
      
      // print the values for each i and j
      auto val = get_scalar(tmp2);
      if(ec.pg().rank() == 0)
        std::cout << i << " " << j << " "  << val << std::endl;
    }
  }
