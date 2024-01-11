:orphan:

TiledIndexSpace Operations
==========================

``TiledIndexSpace`` object provides some operations to be used in
dependent ``TiledIndexSpace`` operations. The **generic** operations,
*union* and *intersection*, can be applied to all ``TiledIndexSpace``
objects. On the other hand, *composition*, *inverse*, and *projection*
can be only applied to dependent index spaces. For both kind of
operations, the corresponding ``TiledIndexSpace`` object has to be
constructed from the same root object via sub-sequencing or using
dependency maps. We also provide interface that accepts and returns
``TiledIndexLabel`` objects.

Generic Operations
------------------

-  **Union:** combines two ``TiledIndexSpace``\ s into a new one that
   includes all tiles from ``lhs`` and ``rhs``.

   :math:`ret_{tis} = lhs_{tis} \cup rhs_{tis}`

   .. code:: cpp

      TiledIndexSpace union_tis(const TiledIndexSpace& lhs, 
                                const TiledIndexSpace& rhs) 
      [[expects: lhs.root_tis() == rhs.root_tis()]]
      [[expects: lhs.is_dependent() == rhs.is_dependent()]];

      TiledIndexLabel union_lbl(const TiledIndexLabel& lhs, 
                                const TiledIndexLabel& rhs) 
      [[expects: lhs.tiled_index_space().root_tis() == rhs.tiled_index_space().root_tis()]]
      [[expects: lhs.tiled_index_space().is_dependent() == rhs.tiled_index_space().is_dependent()]];

-  **Intersection:** intersects two ``TiledIndexSpace``\ s into a new
   one that only includes tiles that are in both ``lhs`` and ``rhs``.
   :math:`ret_{tis} = lhs_{tis} \cap rhs_{tis}`

   .. code:: cpp

      TiledIndexSpace intersect_tis(const TiledIndexSpace& lhs, 
                                    const TiledIndexSpace& rhs)
      [[expects: lhs.root_tis() == rhs.root_tis()]]
      [[expects: lhs.is_dependent() == rhs.is_dependent()]];

      TiledIndexLabel intersect_lbl(const TiledIndexLabel& lhs, 
                                    const TiledIndexLabel& rhs)
      [[expects: lhs.tiled_index_space().root_tis() == rhs.tiled_index_space().root_tis()]]
      [[expects: lhs.tiled_index_space().is_dependent() == rhs.tiled_index_space().is_dependent()]];

Dependent Space Operations
--------------------------

-  **Composition:** composes two dependent ``TiledIndexSpace``\ s into
   one composing the dependency.
   :math:`lhs_{tis_{\beta}} = tis_{\alpha} \mapsto tis_{\beta}`

   :math:`rhs_{tis_{\gamma}} = tis_{\beta} \mapsto tis_{\gamma}`

   :math:`ret_{tis_{\gamma}} = tis_{\alpha} \mapsto tis_{\beta} \mapsto tis_{\gamma} = tis_{\alpha} \mapsto tis_{\gamma}`

   .. code:: cpp

      TiledIndexSpace compose_tis(const TiledIndexSpace& lhs,
                                  const TiledIndexSpace* rhs)
      [[expects: lhs.root_tis() == rhs.root_tis()]]
      [[expects: lhs.is_dependent()]]
      [[expects: rhs.is_dependent()]];

      TiledIndexLabel compose_lbl(const TiledIndexLabel& lhs,
                                  const TiledIndexLabel* rhs)
      [[expects: lhs.tiled_index_space().root_tis() == rhs.tiled_index_space().root_tis()]]
      [[expects: lhs.tiled_index_space().is_dependent()]]
      [[expects: rhs.tiled_index_space().is_dependent()]];

-  **Inverse:** inverts the dependency in ``tis`` creating a new
   dependent ``TiledIndexSpace``.
   :math:`tis_{\beta} = tis_{\alpha} \mapsto tis_{\beta}`
   :math:`ret_{tis_{\alpha}} = tis_{\beta} \mapsto tis_{\alpha}`

   .. code:: cpp

      TiledIndexSpace inverse_tis(const TiledIndexSpace& tis)
      [[expects: tis.is_dependent()]]
      [[expects: tis.dep_vec.size() == 1]];

      TiledIndexLabel inverse_lbl(const TiledIndexLabel& tlbl)
      [[expects: tlbl.tiled_index_space().is_dependent()]]
      [[expects: tlbl.tiled_index_space().dep_vec.size() == 1]];

-  **Projection:** constructs a new ``TiledIndexSpace`` by projecting
   the dependent space into one or more of the dependent spaces

   .. code:: cpp

      TiledIndexSpace project_tis(const TiledIndexSpace& lhs,
                                  const TiledIndexSpace& rhs)
      [[expects: lhs.is_dependent()]]
      [[expects: !rhs.is_dependent()]]
      [[expects: lhs.dep_vec.includes(rhs)]];

      TiledIndexLabel project_lbl(const TiledIndexLabel& lhs,
                                  const TiledIndexLabel& rhs)
      [[expects: lhs.tiled_index_space().is_dependent()]]
      [[expects: !rhs.tiled_index_space().is_dependent()]]
      [[expects: lhs.tiled_index_space().dep_vec.includes(rhs)]];

   --------------

   .. rubric:: Helper Methods
      :name: helper-methods

-  **Domain:** returns the key values (IndexVector) for the dependent
   ``TiledIndexSpace`` dependencies

   .. code:: cpp

      std::vector<IndexVector> domain_tis(const TiledIndexSpace& tis)
      [[expects: tis.is_dependent()]];

-  **Range:** returns set of indices or a new ``TiledIndexSpace`` from
   the values for the dependent ``TiledIndexSpace`` dependencies.

   .. code:: cpp

      std::vector<Index> range_tis(const TiledIndexSpace& tis)
      [[expects: tis.is_dependent()]];
      TiledIndexSpace range_tis(const TiledIndexSpace& tis)
      [[expects: tis.is_dependent()]];
