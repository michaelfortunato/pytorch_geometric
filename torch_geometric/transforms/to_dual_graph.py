from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform('to_dual_graph')
class ToDualGraph(BaseTransform):
    r"""Constructs the dual graph  :math:`G^* = (V^*, E^*)` of the given
    mesh :math:`G=(V,F)`, where a unique vertex :math:`v^*`
    in :math:`V^*` corresponds to a unique face :math:`f` in :math:`F`,
    and an edge :math:`e = (v^*, v^{*'})` is in :math:`E^*` if and only if
    the two corresponding faces :math:`f, f' \in F` in are adjacent to
    each other in the orginal mesh :math:`G`.

    The mesh should be a :obj:`torch_geometric.data.Data` object
    which has attributes :obj:`data.pos` of shape
    :obj:`[number_of_vertices,3]` (vertex positions)
    and by :obj:`data.face` of shape :obj:`[number_of_faces, 3]`
    (three vertices in data.post).

    In the dual graph, each face in the original mesh is represented by a node,
    and an edge exists between two nodes if their corresponding
    faces share an edge in the original graph. Explicitly, let
    :math:`\phi: V^* \mapsto F` assign each node in the dual graph to a unique
    face in :math:`G`. An edge :math:`e^* = (v^*, v'^*)` is in the dual graph
    when the corresponding faces :math:`\phi(v^*), \phi(v'^*)` in :math:`G`,
    are adjacent to each other (i.e) they share an edge.

    Args:
        data (torch_geometric.data.Data): The input mesh data object where
            :obj:`data.pos` are the vertex positions of shape
            :obj:`[num_vertices, 3]`, and :obj:`data.face` has
            shape :obj:`[3, num_faces]`.
        sorted (bool, optional): If set to :obj:`True`, will sort
            neighbor indices for each node in ascending order.
            (default: :obj:`False`)

    Returns:
        torch_geometric.data.Data: A new :obj:`Data` object where
        :obj:`edge_index`
        is of shape :math:`[2,|V^*|]`, and defines the adjacency of
        vertices in the dual graph and :obj:`num_nodes`
        equals the number of faces in the original mesh.

    Example:
        >>> import torch
        >>> from torch_geometric.data import Data
        >>> from torch_geometric.transforms import to_dual_graph
        >>>
        >>> # Define a simple square mesh made of two triangles:
        >>> pos = torch.tensor([
        ...     [0, 0, 0],
        ...     [1, 0, 0],
        ...     [1, 1, 0],
        ...     [0, 1, 0]
        ... ], dtype=torch.float)
        >>> face = torch.tensor([
        ...     [0, 1, 2],
        ...     [0, 2, 3]
        ... ], dtype=torch.long)
        >>>
        >>> # Wrap in a PyG Data object (face might need to be transposed
        depending on shape)
        >>> data = Data(pos=pos, face=face.t())
        >>>
        >>> # Convert to the dual graph
        >>> dual_data = dual_graph(data)
        >>>
        >>> # dual_data.edge_index holds the connectivity between the two faces
        >>> print(dual_data)
        Data(edge_index=[2, 1], num_nodes=2)

    .. note::

        - Technically, dual graphs are often defined for planar graphs,
          but the concept is commonly applied to meshes as well.
        - This function assumes that all faces in :obj:`data.face` are valid
          triangles.
        - The dual graph is only fully planar if the original mesh is planar,
          but
          constructing duals of 3D meshes is still commonly used for mesh
          analysis.

    .. seealso::

        - `Dual Graph (Wikipedia) <https://en.wikipedia.org/wiki/Dual_graph>`_
        - `Gross, Jonathan L.; Yellen, Jay, eds. (2003),
          Handbook of Graph Theory,
          CRC Press, p. 724, ISBN 978-1-58488-090-5.
          <https://books.google.com/books?id=mKkIGIea_BkC&lpg=PA724>`_
    """
    def __init__(self) -> None:
        raise NotImplementedError

    def forward(self, data: Data) -> Data:
        raise NotImplementedError
