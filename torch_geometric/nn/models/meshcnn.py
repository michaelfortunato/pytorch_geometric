from typing import cast

import torch
from torch import BoolTensor, Tensor

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


class MeshCNN(torch.nn.Module):
    r"""The neural network from the paper `"MeshCNN: A Network With An Edge"
    <https://arxiv.org/abs/1809.05910>`_.

    .. ATTENTION::
        MeshCNN is not yet ready for use, and its ongoing development
        is being tracked in issue `#283
        <https://github.com/pyg-team/pytorch_geometric/issues/283>`__.
        Any usage of this class will raise a :obj:`NotImplementedError`
        error.

    """
    def __init__(self):
        raise NotImplementedError

    def forward(self):
        r"""
        ..
            Can't hide the documentation of this method.
            This is next best option.

        Raises:
            NotImplementedError: Always, until `#283
              <https://github.com/pyg-team/pytorch_geometric/issues/283>`__
              is finished.
        """  # noqa: D212, D419
        raise NotImplementedError

    @functional_transform('meshcnn_feature_extraction_layer')
    class FeatureExtractionLayer(BaseTransform):
        r"""This is the first layer of MeshCNN and transforms
        the input mesh :math:`\mathcal{m} = (V, F)` into the feature tensor
        and the adjacency tensor that define the features and adjacencies
        of the abstract graph processed by the rest of MeshCNN.

        Given a mesh :math:`\mathcal{m} = (V, F)`,
        this function computes:

        1. **Edge Feature Matrix**
        :math:`X^{(1)} \in \mathbb{R}^{|E| \times 5}`:
          TODO: The edge feature tensor, where row i holds the 5
          geometric features associated with edge :math:`e_i` in mesh
          :math:`m`. Using the Aggregate Update definition of a message
          passing GNN. This tensor can be thought of as node features
          of the line graph of :math:`m`.
          `MeshCNN <https://arxiv.org/abs/1809.05910>`_.

        2. **Edge Adjacency Matrix** :math:`A \in
        \{0, ..., |E| - 1\}^{2 \times 4*|E|}`:
          TODO:

        We can write this layer as

        .. math::
            \text{FeatureExtraction}(\mathcal{m}) \mapsto (X^{(1)}, A)

        where:
            - Input: :math:`\mathcal{m} = (V, F) \in \mathbb{R}^{|V| \times 3}
              \times \{0,...,|E|-1|\}^{3 \times |F|}`
            - Output:
              :math:`X^{(1)} \in \mathbb{R}^{|E| \times 5}`,
              :math:`A \in \{0, ..., |E| - 1\}^{2 \times 4*|E|}`.

        We now precisely define the *edge feature matrix*
        :math:`X^{(1)} \in \mathbb{R}^{|E| \times 3}` and the
        *edge adjacency matrix*
        :math:`A \in \{0, ..., |E| - 1\}^{2 \times 4*|E|}`.

        **Edge Feature Matrix:**

        The :math:`i` th row of the
        *edge feature matrix* :math:`X^{(1)} \in \mathbb{R}^{|E| \times 3}`,
        denoted as :math:`x^{(1)}_i \in \mathbb{R}^{5}`, holds the :math:`5`
        features associated with edge :math:`e_i` of the input mesh
        :math:`\mathcal{m} = (V, F)`. The 5 components of :math:`x^{(1)}_i`
        are:

        1. **Dihedral Angle** :math:`\eta`: The angle between
        the two faces incident to edge :math:`e_i`.

        2. **Min Face Angle**:
        :math:`\min(\alpha, \beta)`.

        3. **Max Face Angle**:
        :math:`\max(\alpha, \beta)`.

        4. **Min Edge Ratio**:
        :math:`\min(\frac{|\overline{OC}|}{|\overline{AB}|},
        \frac{|\overline{OD}|}{|\overline{AB}|})`
        where :math:`O` is midpoint of edge :math:`e_i`
        aka :math:`\overline{AB}`.

        5. **Max Edge Ratio**:
        :math:`\max(\frac{|\overline{OC}|}{|\overline{AB}|},
        \frac{|\overline{OD}|}{|\overline{AB}|})`
        where :math:`C,D` are opposite vertices.

        Please see *Figure 1* for an illustration.

        .. figure:: ../_figures/meshcnn_edge_features.svg
            :align: center
            :width: 60%

            **Figure 1:** Let :math:`e_i = \overline{AB}`.
            Then :math:`x_i^{(1)} = \bigl(\eta,
            \text{Min}(\alpha, \beta),
            \text{Max}(\alpha, \beta),
            \text{Min}(\frac{\overline{OC}}{\overline{AB}},
            \frac{\overline{OD}}{\overline{AB}}),
            \text{Max}(\frac{\overline{OC}}{\overline{AB}},
            \frac{\overline{OD}}{\overline{AB}})`.
            That is to say, for every edge :math:`\overline{AB}`
            in the triangular mesh, this layer (the first layer of MeshCNN)
            computes 1. its dihedral angle :math:`\eta`,
            2. its minimal face pair angle, :math:`\text{Min}(\alpha, \beta)`,
            3. its maximal face pair angle, :math:`\text{Max}(\alpha, \beta)`,
            4. its minmal edge ratio
            :math:`\text{Min}(\frac{\overline{OC}}{\overline{AB}},
            \frac{\overline{OD}}{\overline{AB}})`, and
            5. its maximal edge ratio
            :math:`\text{Max}(\frac{\overline{OC}}{\overline{AB}},
            \frac{\overline{OD}}{\overline{AB}})`.


        **Edge Adjacency Matrix:**

        We now give the definition for the *edge adjacency matrix*
        :math:`A` of the given input mesh :math:`\mathcal{m}`. We complete
        :math:`A` 's definition by explicitly defining
        what it means for two edges to be adjacent in :math:`\mathcal{m}`
        along with defining
        the ordering we impose on  :math:`A`.

        The :math:`j` th column of :math:`A` returns a pair of indices
        :math:`k,l \in \{0,...,|E|-1\}`, which means that edge
        :math:`e_k` is adjacent to edge :math:`e_l`
        in our input mesh :math:`\mathcal{m}`.
        The definition of edge adjacency in a triangular
        mesh is illustrated in Figure 2.
        In a triangular
        mesh, each edge :math:`e_i` is expected to be adjacent to
        exactly :math:`4`
        neighboring edges, hence the number of columns of
        :math:`A`: :math:`4*|E|`.
        We write *the neighborhood* of edge :math:`e_i` as
        :math:`\mathcal{N}(i) = (a(i), b(i), c(i), d(i))` where

        1. :math:`a(i)` denotes the index of the *first* counter-clockwise
        edge of the face *above* :math:`e_i`.

        2. :math:`b(i)` denotes the index of the *second* counter-clockwise
        edge of the face *above* :math:`e_i`.

        3. :math:`c(i)` denotes the index of the *first* counter-clockwise edge
        of the face *below* :math:`e_i`.

        4. :math:`d(i)` denotes the index of the *second*
        counter-clockwise edge of the face *below* :math:`e_i`.

        .. figure:: ../_figures/meshcnn_edge_adjacency.svg
            :align: center
            :width: 60%

            **Figure 2:** The neighbors of edge :math:`\mathbf{e_1}`
            are :math:`\mathbf{e_2}, \mathbf{e_3}, \mathbf{e_4}` and
            :math:`\mathbf{e_5}`, respectively.
            We write this as
            :math:`\mathcal{N}(1) = (a(1), b(1), c(1), d(1)) = (2, 3, 4, 5)`.
            As another example,
            :math:`\mathcal{N}(9) = (a(9), b(9), c(9), d(9)) = (10, 7, 5, 6)`.

        Because of this ordering constrait, :obj:`FeatureExtractionLayer`
        returns :math:`A` with *the following ordering*:

        .. math::
            &A[:,0] = (0, \text{The index of the "a" edge for edge } 0) \\
            &A[:,1] = (0, \text{The index of the "b" edge for edge } 0) \\
            &A[:,2] = (0, \text{The index of the "c" edge for edge } 0) \\
            &A[:,3] = (0, \text{The index of the "d" edge for edge } 0) \\
            \vdots \\
            &A[:,4*|E|-4] =
                \bigl(|E|-1,
                    a\bigl(|E|-1\bigr)\bigr) \\
            &A[:,4*|E|-3] =
                \bigl(|E|-1,
                    b\bigl(|E|-1\bigr)\bigr) \\
            &A[:,4*|E|-2] =
                \bigl(|E|-1,
                    c\bigl(|E|-1\bigr)\bigr) \\
            &A[:,4*|E|-1] =
                \bigl(|E|-1,
                    d\bigl(|E|-1\bigr)\bigr)


        Stated a bit more compactly, for every edge :math:`e_i` in the
        input mesh,
        :math:`A`, should have the following entries

        .. math::
            A[:, 4*i] &= (i, a(i)) \\
            A[:, 4*i + 1] &= (i, b(i)) \\
            A[:, 4*i + 2] &= (i, c(i)) \\
            A[:, 4*i + 3] &= (i, d(i))

        ..
            FIXME:
            We use the 2 below directives
            to render the docs for the forward method,
            which I argue is important. This can be changed universally
            for all torch_geometric.transforms by modifying
            pytorch_gemoetric/source/modules/transforms.rst?plain=1#L30
            to use autosummary/class.rst as its template instead of
            autosummary/only_class.rst.
            Because MeshCNN.FeatureExtractionLayer is a nested class in
            torch_geometric.nn.models, autosummary/nn.rst needs to be
            modified to handle nested subclasses with the forward method.
            In particular, problems are happening because autosummary/nn.rst
            has `:exclude-members: forward` but then re-includes
            it with `..automethod:: forward`, which works for top level
            classes for modules in torch_geometric.nn.models.* but does
            not work for classes inside a class.

        .. seealso::
            - :meth:`edge_adjacency`: Computes the edge adjacency matrix.
            - :meth:`edge_features`: Computes the 5-dimensional geometric
              features for each edge.
            - :meth:`forward`: Main entry point that combines both operations.

        .. currentmodule:: torch_geometric.nn.models.meshcnn.MeshCNN
        .. automethod:: FeatureExtractionLayer.forward

        """
        def __init__(self) -> None:
            return super().__init__()

        def forward(self, data: Data) -> Data:
            r"""Computes :math:`f(\mathcal{m}) = (X, A)`.

            .. ATTENTION::
                :obj:`face` MUST have consistent winding. See
                `here <https://trimesh.org/trimesh.base.html
                #trimesh.base.Trimesh.is_winding_consistent>`__) to learn
                what winding is. Fortunately, most programs such as Blender
                and trimesh ensure that their triangular meshes
                (a.k.a. the faces of the triangular meshes) have
                proper winding.

            Args:
                data (Data):
                    A :obj:`Data` object representing a
                    triangular mesh :math:`\mathcal{m} = (V, F)`.
                    It MUST have the two attributes:

                        - :obj:`data.pos` :math:`V`. A :obj:`torch.Tensor`
                            of shape
                            :obj:`[|V|, 3]`.

                        - :obj:`data.face`: :math:`F`. A :obj:`torch.Tensor` of
                            shape
                            :obj:`[3, |F|]`.

            Returns:
                Data: Returns a :obj:`Data` object that
                has the two attributes:

                    * :obj:`data.x`
                      :math:`X^{(1)} \in \mathbb{R}^{|E| \times 5}`.
                        A :obj:`torch.Tensor` of shape :obj:`[|E|, 5]`,
                        which is
                        also known as is the *edge feature matrix*.

                    * :obj:`data.edge_index`
                      :math:`A \in \{0, ..., |E|-1\}^{2 \times 4 * |E|}`.
                        A :obj:`torch.Tensor` of shape :obj:`[2, 4*|E|]`,
                        which is
                        also known as is the *edge adjacency matrix*.

            Example:
                >>> import torch
                >>> from torch_geometric.data import Data
                >>> from torch_geometric.nn.models import MeshCNN
                >>> # tetrahedral mesh
                >>> # pos.shape=(num_vertices, 3)=(4, 3)
                >>> pos = torch.tensor([[0., 0., 0.], [1., 1., 0.],
                ...                    [1., 0., 1.], [0., 1., 1.]])
                >>> # face.shape=(3, num_faces)=(3, 4)
                >>> face = torch.tensor([[0, 3, 0, 3], [1, 1, 2, 2],
                ...                                    [2, 0, 3, 1]])
                >>> data = Data(pos = pos, face = face)
                >>> meshcnn_layer_1 = MeshCNN.FeatureExtractionLayer()
                >>> data = meshcnn_layer1.forward(data)
                >>> print(data)
                Data(x=[6, 5], edge_index=[2, 24])
                >>> print(f"X^(1) = {data.x}")
                X^(1) = tensor([[1.9106, 1.0472, 1.0472, 0.8660, 0.8660],
                    [1.9106, 1.0472, 1.0472, 0.8660, 0.8660],
                    [1.9106, 1.0472, 1.0472, 0.8660, 0.8660],
                    [1.9106, 1.0472, 1.0472, 0.8660, 0.8660],
                    [1.9106, 1.0472, 1.0472, 0.8660, 0.8660],
                    [1.9106, 1.0472, 1.0472, 0.8660, 0.8660]])
                >>> print(f"A = {data.edge_index}")
                A = tensor([[ 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
                    2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5 ],
                        [3, 1, 2, 4, 0, 3, 5, 2, 4, 0, 1, 5, 1, 0,
                    4, 5, 0, 2, 5, 3, 2, 1, 3, 4 ]])
            """
            pos, face = self._assert_mesh(data)
            edges, edge_adjacency, _ = self.edge_adjacency(face)
            X = self.edge_features(pos, edges, edge_adjacency)
            A = torch.repeat_interleave(
                torch.arange(edge_adjacency.size(0),
                             device=edge_adjacency.device), 4)
            A = torch.stack([A, edge_adjacency.view(-1)], dim=0)
            return Data(x=X, edge_index=A)

        @staticmethod
        def edge_adjacency(
                face: torch.Tensor) -> tuple[Tensor, Tensor, BoolTensor]:
            r"""Computes the edge adjacency matrix of a
            mesh :math:`\mathcal{m} = (V, F)` whose face :math:`F` is given by
            tensor :obj:`face` (and so has shape :obj:`(3, |F|)`).

            Given a triangular mesh :math:`\mathcal{M} = (V, F)` with
            faces :math:`F`, this method computes the edges and
            their adjacency relationships required
            by MeshCNN. Each edge can be adjacent to either 2 neighboring edges
            (boundary edges) or 4 neighboring edges (interior edges).

            .. ATTENTION::
                The input :obj:`face` tensor MUST have consistent
                winding order.
                Most mesh processing libraries (Blender, trimesh) ensure proper
                winding by default. See `trimesh documentation
                <https://trimesh.org/trimesh.base.html#trimesh.base.Trimesh.is_winding_consistent>`__ .. # noqa: E501
                for more details.

            **Edge Adjacency Definition:**

            An **incident face** to an edge is a triangular
            face that contains that edge
            as one of its three sides.
            Each edge can have 1 or 2 incident faces.

            For each edge :math:`e_i` in the mesh, we define its neighborhood
            :math:`\mathcal{N}(i) = (a(i), b(i), c(i), d(i))` where:

            - :math:`a(i)`: First counter-clockwise edge of face 1
            - :math:`b(i)`: Second counter-clockwise edge of face 1
            - :math:`c(i)`: First counter-clockwise edge of face 2
            - :math:`d(i)`: Second counter-clockwise edge of face 2

            **Boundary vs Interior Edges:**

            - **Interior edges**: Adjacent to exactly 2 faces
              (face 1 ≠ face 2), have 4 unique neighboring edges
            - **Boundary edges**: Adjacent to only 1 face (face 1 = face 2),
              have 2 neighboring edges which are
              duplicated in the adjacency matrix following
              MeshCNN's convention: :math:`c(i) = a(i)`, :math:`d(i) = b(i)`.

            ..
                Recall that MeshCNN acts on meshes of the form
                :math:`\mathcal{m} = (V, F)`. Here,
                :math:`V = (v_1, ..., v_n)` for :math:`v_i \in \mathbb{R}^3`
                represents the vertices of the mesh and
                :math:`F = \bigcup^o (i,j,k)
                \quad 0 \leq i,j,k < n, i \neq j \neq k`
                are the faces of the mesh. If :math:`f = (1, 2, 3) \in F`,
                we say that face :math:`f` in mesh :math:`\mathcal{m}` is
                constructed by vertices :math:`v_1, v_2, v_3 \in V`.
                We use :math:`|V| = n` to denote the number of vertices of
                :math:`\mathcal{m}` and :math:`|F| = o` to denote the number
                of faces of :math:`\mathcal{m}`.
                The edge adjacency matrix of :math:`\mathcal{m} = (V, F)`
                is described in detail in :class:`FeatureExtractionLayer`.
                However, we will breifly reiterate its definition here.
                :obj:`face` is the :obj:`(3, num_faces)` tensor representing
                the faces :math:`F` of a mesh :math:`\mathcal{m} = (V, F)`.
                :obj:`edge_adjacency` is a function that given the faces
                :obj:`face` of a mesh, computes the :obj:`edges`, and
                the :obj:`edge_adjacency` of the mesh (it also returns
                the set of edge ids that are interior edges of the mesh, but
                that is not really used).

            Args:
                face (torch.Tensor): Shape: :obj:`(3, |F|)`. The faces
                  :math:`F` of the input
                  mesh :math:`\mathcal{m} = (V, F)` with shape
                  :obj:`(3, |F|)`. For instance, :obj:`face[:,i] = [1, 2, 3]`
                  says that face :math:`i` of input mesh :math:`\mathcal{m}`
                  is constructed by vertices :math:`v_1, v_2, v_3 \in V`.

            Returns:
                tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:
                A triplet of tensors
                :obj:`(edges, edge_adjacency, _interior_edge_mask)`.
                Each tensor is documented in full below.

                1. :obj:`edges` (torch.Tensor): Shape: :obj:`(2, |E|)`.
                This tensor represents the edges of the input mesh
                :math:`\mathcal{m} = (V, F)` whose face :math:`F` is given by
                :obj:`face`. :obj:`edges` is a :obj:`torch.Tensor` of shape
                :obj:`(2, |E|)`, where :obj:`|E|` denotes the number
                of edges in the mesh :math:`\mathcal{m}`.

                2. :obj:`edge_adjacency` (torch.Tensor). Shape:
                :obj:`(|E|, 4)`.
                This tensor represents the edge adjacencies of the input mesh
                :math:`\mathcal{m} = (V, F)` whose face :math:`F` is given by
                :obj:`face`.

                3. :obj:`_interior_edge_mask` (torch.BoolTensor).
                Shape: :obj:`(|E|, 1)`. This is output is usually ignored
                but we return it because it might be useful for debugging.
                :obj:`_interior_edge_mask[i] = True` if edge i is an interior
                edge, that is if edge i is adjacent to four unique edges.
                This can also be checked in code by seeing if
                :obj:`edge_adjacency[i, 0] != edge_adjacency[i, 2]`.

            """
            # edges: shape (2, |E|)
            #   Think of this tensor as edge id to edge definition map
            #   In other words, edges[:, i] returns the two vertex
            #   ids in our mesh that construct edge i.
            #   Note that we deduplicate edges (v1, v2) and edge (v2, v1).
            #   Why do we do this? We want to assign each edge (v1,v2)
            #   an id. unique_edges associates each unique pair (v1, v2)
            #   with an index (edges[:,i] = (v1, v2)), thus implicitly
            #   assigning an id to each edge. To summarize, to see the two
            #   vertices that construct edge with id i, inspect
            #   edges[:, i].
            # edge_ids: shape (1, 3|F|)
            #   unique_edges[edge_ids] == torch.sort(edges, dim=0)[0]
            #   In words, the ith entry of edge_ids tells us which edge_id
            #   the edge in edges (technically, torch.sort(edges, dim=0)[0])
            #   corresponds to. This is crucial because edges has the structure
            #   that every 3 consecutive edges forms a face. For instance,
            #   edges[:, i], edges[:, i+1], edges[:, i+2] form face i.
            # edge_counts: shape (|E|, 1)
            #   edge_counts[i] tells us how many times edge with id i
            #   appears in our mesh. If edge with id i is a boundary edge,
            #   edge_counts[i] is 1. Other, the edge with id i is an
            #   interior edge, and so edge_counts[i] is 2.
            edges = torch.stack(
                [
                    face[[0, 1], :],  # v0 -> v1
                    face[[1, 2], :],  # v1 -> v2
                    face[[2, 0], :],  # v2 -> v0
                ],
                dim=2).reshape(2, -1)  # (2, 3|F|)

            edges, edge_ids, edge_counts = torch.unique(
                # sort the vertex indices within each edge so that
                # unique works. Recall that edges is shape (2, 3|F|)
                # torch.sort sorts the vertex indices within each edge
                # so that edge_a = (v1, v2) can be deduplicated with
                # edge_b = (v2, v3).
                # For instance, say edges = [[1, 2], [[2, 1]], [0,3], [0, 5]]
                # then torch.sort(edges, dim=0)[0] returns
                # torch.sort(edges, dim=0)[0] =
                #   [[1, 2], [[1, 2]], [0,3], [0, 5]]
                # This then allows torch.unique to deduplicate [[1,2]]
                # and [[1,2]]. Note that torch.sort(edges, dim=0) returns a
                # the indices along with the sorted edges themselves. We only
                # care about the edges themselves, hence
                # torch.sort(edges, dim=0)[0]
                torch.sort(edges, dim=0)[0],
                # Our input edges have shape (2, |E|), and we wish to
                # deduplicate (v1, v2) from (v2, v1), hence we call unique on
                # dim=1.
                dim=1,
                return_inverse=True,
                return_counts=True)

            # face_edges: shape (|F|, 3)
            #   Recall that edge_ids[i] returns the edge id associated
            #   with edges[:,i]. Furthermore, every 3 consecutive edges,
            #   edges[:, i], edges[:, i+1], edges[:, i+2] construct face i.
            #   Therefore, edge_ids[i], edge_ids[i+1], edge_ids[i+2]
            #   are the three edge ids that construct face i. Therefore,
            #   face_edges[i] returns the 3 edge ids that construct face i in
            #   our input mesh. (Remember, to get the two vertex ids
            #   that construct edge with id i), use unique_edges[:, i])
            face_edges = edge_ids.view(-1, 3)

            # neighbor_pairs: shape (3|F|, 2)
            #   Reecal that every three consecutive edges in face_edges
            #   holds the three edge ids that construct face i. That is,
            #   face_edges[i, 0], face_edges[i, 1], face_edges[i:, 2]
            #   construct face i.
            #   Given a face f_i = (e_0, e_1, e_2), the edges can be ordered
            #   in three possible ways:
            #       (e_0, e_1, e_2), (e_2, e_0, e_1), and (e_1, e_2, e_0).
            #   When we are trying to find the two edges adjacent to e_0,
            #   we want to use (e_0, e_1, e_2), and similarly we want to use
            #   (e_1, e_2, e_0) and (e_2, e_0, e_1), when considering edge e_1
            #   and edge e_2 as the base edge of the face.
            #   By enumerating all 3 windings of a face, we set ourselves up
            #   for considering the neighbhorood. Then, we use stack and view
            #   to set up the key ingredient.
            #   At last the following is the key ingredient to
            #   neighbhor_pairs. Let as before, let us consider
            #   f_i = (e_0, e_1, e_2). neighbor_pairs has the following order
            #       neighbor_pairs[i] = [e_0, e_1]
            #       neighbor_pairs[i + |F|] = [e_1, e_2]
            #       neighbor_pairs[i + 2|F|] = [e_2, e_0]
            neighbor_pairs = torch.stack(
                [face_edges[:, [(i + 1) % 3, (i + 2) % 3]] for i in range(3)],
                dim=1).reshape(-1, 2)

            # edge_ids tells us the edge id of the ith edge in edges.
            sorted_edge_ids, order = torch.sort(edge_ids)
            sorted_neighbors = neighbor_pairs[order]

            first_occurrence_mask = torch.cat([
                torch.tensor([True], device=face.device), sorted_edge_ids[1:]
                != sorted_edge_ids[:-1]
            ])  # shape (|edges|, 1)

            # NOTE: An edge can be a boundary edge or an interior edge.
            # If edge i is a boundary edge, it has only two neighbors,
            # with edge_adjacency[i,3], edge_adjacency[i, 4] = -1.
            # Otherwise edge is an interior edge with 4 distinct neighbors.
            # In the case edge i is a boundary edge,
            # the MeshCNN paper simply considers
            # edge_adjacency[i, 3] = edge_adjacency[i, 0]
            # and edge_adjacency[i, 4] = edge_adjacency[i, 1]
            # when computing the features of edge i.
            # So we modify edge_adjacency accordingly here.
            # ref: https://github.com/ranahanocka/MeshCNN/blob/5bf0b899d48eb204b9b73bc1af381be20f4d7df1/models/layers/mesh_prepare.py#L384 # noqa: E501
            interior_edge_mask = edge_counts == 2
            # (|edges|, 2)
            # Think of sorted_neighbors[first_occurrence_mask] as
            # edges a, b
            first_neighbors = sorted_neighbors[first_occurrence_mask]
            # (|edges|, 4)
            # For our adjacency tensor, we know that every edge
            # will have its first two neigbors. So we just
            # copy the a,b edges to match MeshCNN's implementation
            # (see above).
            adjacency = first_neighbors.repeat(1, 2)
            # Overwrite interior edges with their actual second neighbors
            # Think of sorted_neighbors[~first_occurrence_mask] as
            # edges c, d
            # Only interior edges have c and d
            adjacency[interior_edge_mask,
                      2:4] = sorted_neighbors[~first_occurrence_mask]
            return edges, adjacency, interior_edge_mask

        @staticmethod
        def edge_features(
            pos: Tensor,
            edges: Tensor,
            edge_adjacency: Tensor,
        ) -> Tensor:
            r"""Computes the edge features."""
            #        C
            #       /|\
            #      / | \
            #     /  |  \
            #    /   |   \
            #   /    |    \
            #  /     |     \
            # A------O------B
            #  \     |     /
            #   \    |    /
            #    \   |   /
            #     \  |  /
            #      \ | /
            #       \|/
            #        D
            edge_AC_ids = edge_adjacency[:, 0]
            edge_BD_ids = edge_adjacency[:, 2]

            edge_AC_vertex_ids = edges.t()[edge_AC_ids]
            edge_BD_vertex_ids = edges.t()[edge_BD_ids]
            edge_AB_vertex_ids = edges.t()
            # the vertex ids of the whole neighborhood
            neighborhood_vertex_ids = torch.stack(
                [edge_AC_vertex_ids, edge_BD_vertex_ids],
                dim=-1).view(-1, 4)  # (|E|, 4)
            __alias = neighborhood_vertex_ids  # Just to keep it 79 width
            edge_AB_v1_mask = __alias == edge_AB_vertex_ids[:, 0].view(-1, 1)
            edge_AB_v2_mask = __alias == edge_AB_vertex_ids[:, 1].view(-1, 1)
            edge_AB_vertex_id_mask = edge_AB_v1_mask | edge_AB_v2_mask
            opposite_vertex_ids = neighborhood_vertex_ids[
                ~edge_AB_vertex_id_mask].view(-1, 2)

            pos_A = pos[edge_AB_vertex_ids][:, 0, :]
            pos_B = pos[edge_AB_vertex_ids][:, 1, :]
            pos_O = (pos_A +
                     pos_B) / 2  # i.e. the midpoint pos_A + (pos_B - pos_A)/2
            pos_C = pos[opposite_vertex_ids][:, 0, :]
            pos_D = pos[opposite_vertex_ids][:, 1, :]

            vec_AB = pos_B - pos_A
            vec_AC = pos_C - pos_A
            vec_BC = pos_C - pos_B
            vec_AD = pos_D - pos_A
            vec_BD = pos_D - pos_B

            normal_1 = torch.cross(vec_AB, vec_AC, dim=-1)
            normal_2 = torch.cross(vec_AB, vec_AD, dim=-1)
            normal_1_norm = normal_1 / torch.norm(normal_1, p=2, dim=-1,
                                                  keepdim=True)
            normal_2_norm = normal_2 / torch.norm(normal_2, p=2, dim=-1,
                                                  keepdim=True)
            dihedral_angle = torch.pi - torch.acos(
                torch.clamp(torch.sum(normal_1_norm * normal_2_norm, dim=-1),
                            -1, 1))

            # Compute the angles
            vec_AC_norm = vec_AC / torch.norm(vec_AC, dim=-1, keepdim=True)
            vec_BC_norm = vec_BC / torch.norm(vec_BC, dim=-1, keepdim=True)
            cos_alpha = torch.sum(vec_AC_norm * vec_BC_norm, dim=-1)
            cos_alpha = torch.clamp(cos_alpha, -1, 1)
            alpha = torch.acos(cos_alpha)  # Shape: (750,)

            vec_AD_norm = vec_AD / torch.norm(vec_AD, dim=-1, keepdim=True)
            vec_BD_norm = vec_BD / torch.norm(vec_BD, dim=-1, keepdim=True)
            cos_beta = torch.sum(vec_AD_norm * vec_BD_norm, dim=-1)
            cos_beta = torch.clamp(cos_beta, -1, 1)
            beta = torch.acos(cos_beta)

            OC_AB_ratio = torch.norm(pos_C - pos_O, dim=1) / torch.norm(
                pos_B - pos_A, dim=1)
            OD_AB_ratio = torch.norm(pos_D - pos_O, dim=1) / torch.norm(
                pos_B - pos_A, dim=1)

            features = torch.stack([
                dihedral_angle,
                torch.min(alpha, beta),
                torch.max(alpha, beta),
                torch.min(OC_AB_ratio, OD_AB_ratio),
                torch.max(OC_AB_ratio, OD_AB_ratio)
            ], dim=1)
            return features

        @staticmethod
        def _assert_mesh(data: Data) -> tuple[Tensor, Tensor]:
            """Validate and return mesh tensors with proper shapes.

            Error otherwise.
            Like unwrap in Rust
            """
            assert data.pos is not None, "Data must have `pos` attribute"
            assert data.face is not None, "Data must have `face` attribute"
            if data.pos.size(1) != 3:
                raise ValueError(f"pos must be [|V|, 3], got {data.pos.shape}")
            if data.face.size(0) != 3:
                raise ValueError(
                    f"face must be [3, |F|], got {data.face.shape}")
            return cast(Tensor, data.pos), cast(Tensor, data.face)
