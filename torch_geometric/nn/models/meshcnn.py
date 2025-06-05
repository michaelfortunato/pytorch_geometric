from typing import Callable, List, Union, cast

import torch
from torch import Tensor

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


class MeshCNN(torch.nn.Module):
    r"""The neural network from the paper `"MeshCNN: A Network With An Edge"
    <https://arxiv.org/abs/1809.05910>`_.


    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
        depth (int): The depth of the U-Net architecture.
        pool_ratios (float or [float], optional): Graph pooling ratio for each
            depth. (default: :obj:`0.5`)
        sum_res (bool, optional): If set to :obj:`False`, will use
            concatenation for integration of skip connections instead
            summation. (default: :obj:`True`)
        act (torch.nn.functional, optional): The nonlinearity to use.
            (default: :obj:`torch.nn.functional.relu`)
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        depth: int,
        pool_ratios: Union[float, List[float]] = 0.5,
        sum_res: bool = True,
        act: Union[str, Callable] = 'relu',
    ):
        raise NotImplementedError

    @functional_transform('meshcnn_feature_extraction_layer')
    class FeatureExtractionLayer(BaseTransform):
        r"""Given a mesh :math:`\mathcal{m} = (V, F)`,
        this method computes :math:`X^{(1)} \in \mathbb{R}^{|E| \times 5}`,
        the *edge feature* matrix required by
        `MeshCNN <https://arxiv.org/abs/1809.05910>`_, along with
        :math:`A \in \{0, ..., |E| - 1\}^{2 \times 4*|E|}`,
        the *edge adjacency* matrix required by MeshCNN.

        This layer, therefore,
        can be expressed as the function.

        .. math::
            &\text{FeatureExtraction}(\mathcal{m}) \mapsto (X^{(1)}, A) \\
            &\text{where, } \mathcal{m} = (V, F) \in \mathbb{R}^{|V| \times 3}
            \times \{0,...,|E|-1|\}^{3 \times |F|}, \\
            &X^{(1)} \in \mathbb{R}^{|E| \times 5}, \\
            &A \in \{0, ..., |E| - 1\}^{2 \times 4*|E|}.

        We now explain the exact form of the *edge feature matrix*
        :math:`X^{(1)} \in \mathbb{R}^{|E| \times 3}` and the
        *edge adjacency matrix*
        :math:`A \in \{0, ..., |E| - 1\}^{2 \times 4*|E|}`.

        The :math:`i` th row of the
        *edge feature matrix* :math:`X^{(1)} \in \mathbb{R}^{|E| \times 3}`,
        denoted as :math:`x^{(1)}_i \in \mathbb{R}^{5}` holds the :math:`5`
        features associated with edge :math:`e_i` of the input mesh
        :math:`\mathcal{m} = (V, F)`. The 5 components of :math:`x^{(1)}_i`
        are:

        1. The dihedral angle of the two faces incident to edge :math:`e_i`.

        2. The minimum of :obj:`Inner Angle 1` and
        `Inner Angle 2` , where
        `Inner Angle 1 ` represents the angle formed
        by the two other edges of face 1.

        3. The maximum of `Inner Angle 1` and `Inner Angle 2` .

        4. The minimum of `Edge Ratio 1` and Edge Ratio 2`, where
        `Edge Ratio 1` is the ratio
        between the edge and the line perpendicular
        to the edge from the opposite
        vertex of face 1.

        5. The maximum of `Edge Ratio 1` and Edge Ratio 2`.

        Please see *Figure 1* for an illustration.

        .. figure:: ../_figures/meshcnn_edge_features.svg
            :align: center
            :width: 60%

            **Figure 1:** Let :math:`e_i = \overline{AB}`.
            Then :math:`x_i^{(1)} = \bigl(\eta,
            \text{Min}(\alpha, \beta),
            \text{Max}(\alpha, \beta),
            \text{Min}(\frac{\overline{OC}}{\overline{AB}}),
            \text{Max}(\frac{\overline{OD}}{\overline{AB}}) \bigr)`.
            That is to say, for every edge :math:`\overline{AB}`
            in the triangular mesh, this layer (the first layer of MeshCNN)
            computes 1. its dihedral angle :math:`\eta`,
            2. its minimal face angle, :math:`\text{Min}(\alpha, \beta)`,
            3. its maximal face angle, :math:`\text{Max}(\alpha, \beta)`,
            4. its minmal edge ratio
            :math:`\text{Min}(\frac{\overline{OC}}{\overline{AB}})`, and
            5. its maximal edge ratio
            :math:`\text{Max}(\frac{\overline{OD}}{\overline{AB}})`.

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
            :math:`\mathcal{N}(1) = (a(1), b(1), c(1), d(1)) = (2, 3, 4, 5)`

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

        Args:
            data(Data): A :obj:`Data` object representing a
                triangular mesh :math:`\mathcal{m} = (V, F)`
                It MUST have the two attributes:

                * :obj:`mesh.pos`: :math:`V`. The :obj:`Tensor` of shape
                  :obj:`[|V|, 3]`

                * :obj:`mesh.face`: :math:`F`. The :obj:`Tensor` of shape
                  :obj:`[3, |F|]`.

        Returns:
            :math:`(X^{(1)}, A)`. In particular, this is a :obj:`Data` object
            that has the two attributes:

                * :obj:`data.x`: :math:`X^{(1)} \in \mathbb{R}^{|E| \times 5}`.
                  The :obj:`Tensor` of shape :obj:`[|E| \times 5]`, which is
                  also known as is the *edge feature matrix*.

                * :obj:`data.edge_index`:
                  :math:`A \in \{0, ..., |E|-1\}^{2 \times 4 * |E|}`.
                  The :obj:`Tensor` of shape :obj:`[2, 4*|E|]`, which is
                  also known as is the *edge adjacency matrix*.

        .. warning::
            We assume that the vertex indices given in :math:`F` are already
            ordered counter-clockwise. This requirement is also known
            as ensuring that the mesh has a valid *winding*.
        """
        def __init__(self) -> None:
            return super().__init__()

        def forward(self, data: Data) -> Data:
            r"""Computes :math:`f(\mathcal{m}) = (X, A)`."""
            pos, face = self._assert_mesh(data)
            edge_adjacency, unique_edges = self.adj(face)
            X = self.features(pos, edge_adjacency, unique_edges)
            A = torch.repeat_interleave(torch.arange(edge_adjacency.size(0)),
                                        4)
            A = torch.stack([A, edge_adjacency.view(-1)], dim=0)
            return Data(x=X, edge_index=A)

        @staticmethod
        def adj(face: torch.Tensor):
            edges = torch.stack(
                [
                    face[[0, 1], :],  # v0 -> v1
                    face[[1, 2], :],  # v1 -> v2
                    face[[2, 0], :],  # v2 -> v0
                ],
                dim=2).reshape(2, -1)

            sorted_edges, _ = torch.sort(edges, dim=0)

            unique_edges, edge_ids, edge_counts = torch.unique(
                sorted_edges, dim=1, return_inverse=True, return_counts=True)

            face_edges = edge_ids.view(-1, 3)

            neighbor_pairs = torch.stack(
                [face_edges[:, [(i + 1) % 3, (i + 2) % 3]] for i in range(3)],
                dim=1).reshape(-1, 2)

            sorted_edge_ids, order = torch.sort(edge_ids)
            sorted_neighbors = neighbor_pairs[order]

            first_occurrence_mask = torch.cat([
                torch.tensor([True], device=edges.device), sorted_edge_ids[1:]
                != sorted_edge_ids[:-1]
            ])

            first_positions = first_occurrence_mask.nonzero(as_tuple=True)[0]

            adjacency = torch.full((unique_edges.size(1), 4), -1,
                                   dtype=torch.long, device=edges.device)

            adjacency[:, :2] = sorted_neighbors[first_positions]

            interior_edge_indices = (edge_counts == 2).nonzero(
                as_tuple=True)[0]
            second_positions = first_positions[interior_edge_indices] + 1
            adjacency[interior_edge_indices,
                      2:] = sorted_neighbors[second_positions]

            return adjacency, unique_edges

        @staticmethod
        def features(pos: Tensor, edge_adjacency: Tensor,
                     unique_edges: Tensor):
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

            edge_AC_vertex_ids = unique_edges.t()[edge_AC_ids]
            edge_BD_vertex_ids = unique_edges.t()[edge_BD_ids]
            edge_AB_vertex_ids = unique_edges.t()
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
            # TODO: Verify this is right
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
