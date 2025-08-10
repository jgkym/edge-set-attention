import tempfile
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import (
    get_laplacian,
    to_dense_adj,
    to_scipy_sparse_matrix,
    to_undirected,
)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from yacs.config import CfgNode as CN


def get_default_cfg_posenc(cfg):
    """Extend configuration with positional encoding options."""

    # Argument group for each Positional Encoding class.
    cfg.posenc_LapPE = CN()
    cfg.posenc_SignNet = CN()
    cfg.posenc_RWSE = CN()
    cfg.posenc_HKdiagSE = CN()
    cfg.posenc_ElstaticSE = CN()
    cfg.posenc_EquivStableLapPE = CN()

    # Effective Resistance Embeddings
    cfg.posenc_ERN = CN()  # Effective Resistance for Nodes
    cfg.posenc_ERE = CN()  # Effective Resistance for Edges

    # Common arguments to all PE types.
    for name in [
        "posenc_LapPE",
        "posenc_SignNet",
        "posenc_RWSE",
        "posenc_HKdiagSE",
        "posenc_ElstaticSE",
        "posenc_ERN",
        "posenc_ERE",
    ]:
        pecfg = getattr(cfg, name)

        # Use extended positional encodings
        pecfg.enable = False

        # Neural-net model type within the PE encoder:
        # 'DeepSet', 'Transformer', 'Linear', 'none', ...
        pecfg.model = "none"

        # Size of Positional Encoding embedding
        pecfg.dim_pe = 8

        # Number of layers in PE encoder model
        pecfg.layers = 3

        # Number of attention heads in PE encoder when model == 'Transformer'
        pecfg.n_heads = 4

        # Number of layers to apply in LapPE encoder post its pooling stage
        pecfg.post_layers = 0

        # Choice of normalization applied to raw PE stats: 'none', 'BatchNorm'
        pecfg.raw_norm_type = "none"

        # In addition to appending PE to the node features, pass them also as
        # a separate variable in the PyG graph batch object.
        pecfg.pass_as_var = False

    # Config for EquivStable LapPE
    cfg.posenc_EquivStableLapPE.enable = False
    cfg.posenc_EquivStableLapPE.raw_norm_type = "none"

    # Config for Laplacian Eigen-decomposition for PEs that use it.
    for name in ["posenc_LapPE", "posenc_SignNet", "posenc_EquivStableLapPE"]:
        pecfg = getattr(cfg, name)
        pecfg.eigen = CN()

        # The normalization scheme for the graph Laplacian: 'none', 'sym', or 'rw'
        pecfg.eigen.laplacian_norm = "sym"

        # The normalization scheme for the eigen vectors of the Laplacian
        pecfg.eigen.eigvec_norm = "L2"

        # Maximum number of top smallest frequencies & eigenvectors to use
        pecfg.eigen.max_freqs = 10

    # Config for SignNet-specific options.
    cfg.posenc_SignNet.phi_out_dim = 4
    cfg.posenc_SignNet.phi_hidden_dim = 64

    for name in ["posenc_RWSE", "posenc_HKdiagSE", "posenc_ElstaticSE"]:
        pecfg = getattr(cfg, name)

        # Config for Kernel-based PE specific options.
        pecfg.kernel = CN()

        # List of times to compute the heat kernel for (the time is equivalent to
        # the variance of the kernel) / the number of steps for random walk kernel
        # Can be overridden by `posenc.kernel.times_func`
        pecfg.kernel.times = []

        # Python snippet to generate `posenc.kernel.times`, e.g. 'range(1, 17)'
        # If set, it will be executed via `eval()` and override posenc.kernel.times
        pecfg.kernel.times_func = ""

    # Override default, electrostatic kernel has fixed set of 10 measures.
    cfg.posenc_ElstaticSE.kernel.times_func = "range(10)"

    # Setting accuracy for Effective Resistance Calculations:
    cfg.posenc_ERN.accuracy = 0.1
    cfg.posenc_ERE.accuracy = 0.1

    # To be set during the calculations:
    cfg.posenc_ERN.er_dim = "none"

    return cfg


# def dict_to_namespace(d):
#     if isinstance(d, dict):
#         return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
#     elif isinstance(d, list):
#         return [dict_to_namespace(i) for i in d]
#     else:
#         return d

yaml_LapPE = """posenc_LapPE:
  enable: True
  eigen:
    laplacian_norm: none
    eigvec_norm: L2
    max_freqs: 1
  model: DeepSet
  dim_pe: 4
  layers: 2
  raw_norm_type: none"""

yaml_RWSE = """posenc_RWSE:
  enable: True
  kernel:
    times_func: range(1,21)
  model: Linear
  dim_pe: 24
  raw_norm_type: BatchNorm"""

yaml_ESLapPE = """posenc_EquivStableLapPE:
  enable: True
  eigen:
    laplacian_norm: none
    eigvec_norm: L2
    max_freqs: 8
  raw_norm_type: none"""


def compute_posenc_stats(data, pe_types, is_undirected):
    """Precompute positional encodings for the given graph.

    Supported PE statistics to precompute, selected by `pe_types`:
    'LapPE': Laplacian eigen-decomposition.
    'RWSE': Random walk landing probabilities (diagonals of RW matrices).
    'HKfullPE': Full heat kernels and their diagonals. (NOT IMPLEMENTED)
    'HKdiagSE': Diagonals of heat kernel diffusion.
    'ElstaticSE': Kernel based on the electrostatic interaction between nodes.

    Args:
        data: PyG graph
        pe_types: Positional encoding types to precompute statistics for.
            This can also be a combination, e.g. 'eigen+rw_landing'
        is_undirected: True if the graph is expected to be undirected
        cfg: Main configuration node

    Returns:
        Extended PyG Data object.
    """
    # Verify PE types.
    for t in pe_types:
        if t not in ["LapPE", "EquivStableLapPE", "RWSE"]:
            raise ValueError(f"Unexpected PE stats selection {t} in {pe_types}")

    # cfg = ""
    cfg = CN()
    cfg = get_default_cfg_posenc(cfg)

    if "LapPE" in pe_types:
        # cfg += yaml_LapPE + '\n'
        with tempfile.NamedTemporaryFile(
            delete=False, mode="w", suffix=".yaml"
        ) as temp_file:
            temp_file.write(yaml_LapPE)
            temp_file_path = temp_file.name

        cfg.merge_from_file(temp_file_path)

    if "RWSE" in pe_types:
        # cfg += yaml_RWSE + '\n'
        with tempfile.NamedTemporaryFile(
            delete=False, mode="w", suffix=".yaml"
        ) as temp_file:
            temp_file.write(yaml_RWSE)
            temp_file_path = temp_file.name

        cfg.merge_from_file(temp_file_path)

        cfg.posenc_RWSE.kernel.times = list(eval(cfg.posenc_RWSE.kernel.times_func))

    if "EquivStableLapPE" in pe_types:
        # cfg += yaml_ESLapPE + '\n'

        with tempfile.NamedTemporaryFile(
            delete=False, mode="w", suffix=".yaml"
        ) as temp_file:
            temp_file.write(yaml_ESLapPE)
            temp_file_path = temp_file.name

        cfg.merge_from_file(temp_file_path)

    # cfg = yaml.safe_load(cfg)
    # cfg = dict_to_namespace(cfg)

    # Basic preprocessing of the input graph.
    if hasattr(data, "num_nodes"):
        N = data.num_nodes  # Explicitly given number of nodes, e.g. ogbg-ppa
    else:
        N = data.x.shape[0]  # Number of nodes, including disconnected nodes.
    if is_undirected:
        undir_edge_index = data.edge_index
    else:
        undir_edge_index = to_undirected(data.edge_index)

    # Eigen values and vectors.
    evals, evects = None, None
    if "LapPE" in pe_types or "EquivStableLapPE" in pe_types:
        laplacian_norm_type = cfg.posenc_LapPE.eigen.laplacian_norm.lower()
        if laplacian_norm_type == "none":
            laplacian_norm_type = None
        # Eigen-decomposition with numpy, can be reused for Heat kernels.
        L = to_scipy_sparse_matrix(
            *get_laplacian(
                undir_edge_index, normalization=laplacian_norm_type, num_nodes=N
            )
        )
        evals, evects = np.linalg.eigh(L.toarray())

        if "LapPE" in pe_types:
            max_freqs = cfg.posenc_LapPE.eigen.max_freqs
            eigvec_norm = cfg.posenc_LapPE.eigen.eigvec_norm
        elif "EquivStableLapPE" in pe_types:
            max_freqs = cfg.posenc_EquivStableLapPE.eigen.max_freqs
            eigvec_norm = cfg.posenc_EquivStableLapPE.eigen.eigvec_norm

        data.EigVals, data.EigVecs = get_lap_decomp_stats(
            evals=evals, evects=evects, max_freqs=max_freqs, eigvec_norm=eigvec_norm
        )

    # Random Walks.
    if "RWSE" in pe_types:
        kernel_param = cfg.posenc_RWSE.kernel
        if len(kernel_param.times) == 0:
            raise ValueError("List of kernel times required for RWSE")
        rw_landing = get_rw_landing_probs(
            ksteps=kernel_param.times, edge_index=data.edge_index, num_nodes=N
        )
        data.pestat_RWSE = rw_landing

    return data


def get_lap_decomp_stats(evals, evects, max_freqs, eigvec_norm="L2"):
    """Compute Laplacian eigen-decomposition-based PE stats of the given graph.

    Args:
        evals, evects: Precomputed eigen-decomposition
        max_freqs: Maximum number of top smallest frequencies / eigenvecs to use
        eigvec_norm: Normalization for the eigen vectors of the Laplacian
    Returns:
        Tensor (num_nodes, max_freqs, 1) eigenvalues repeated for each node
        Tensor (num_nodes, max_freqs) of eigenvector values per node
    """
    N = len(evals)  # Number of nodes, including disconnected nodes.

    # Keep up to the maximum desired number of frequencies.
    idx = evals.argsort()[:max_freqs]
    evals, evects = evals[idx], np.real(evects[:, idx])
    evals = torch.from_numpy(np.real(evals)).clamp_min(0)

    # Normalize and pad eigen vectors.
    evects = torch.from_numpy(evects).float()
    evects = eigvec_normalizer(evects, evals, normalization=eigvec_norm)
    if N < max_freqs:
        EigVecs = F.pad(evects, (0, max_freqs - N), value=float("nan"))
    else:
        EigVecs = evects

    # Pad and save eigenvalues.
    if N < max_freqs:
        EigVals = F.pad(evals, (0, max_freqs - N), value=float("nan")).unsqueeze(0)
    else:
        EigVals = evals.unsqueeze(0)
    EigVals = EigVals.repeat(N, 1).unsqueeze(2)

    return EigVals, EigVecs


def get_rw_landing_probs(
    ksteps, edge_index, edge_weight=None, num_nodes=None, space_dim=0
):
    """Compute Random Walk landing probabilities for given list of K steps.

    Args:
        ksteps: List of k-steps for which to compute the RW landings
        edge_index: PyG sparse representation of the graph
        edge_weight: (optional) Edge weights
        num_nodes: (optional) Number of nodes in the graph
        space_dim: (optional) Estimated dimensionality of the space. Used to
            correct the random-walk diagonal by a factor `k^(space_dim/2)`.
            In euclidean space, this correction means that the height of
            the gaussian distribution stays almost constant across the number of
            steps, if `space_dim` is the dimension of the euclidean space.

    Returns:
        2D Tensor with shape (num_nodes, len(ksteps)) with RW landing probs
    """
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    source, dest = edge_index[0], edge_index[1]
    deg = torch.zeros(num_nodes, dtype=edge_weight.dtype, device=edge_index.device)
    deg.index_add_(0, source, edge_weight)  # Out degrees.
    deg_inv = deg.pow(-1.0)
    deg_inv.masked_fill_(deg_inv == float("inf"), 0)

    if edge_index.numel() == 0:
        P = edge_index.new_zeros((1, num_nodes, num_nodes))
    else:
        # P = D^-1 * A
        P = torch.diag(deg_inv) @ to_dense_adj(
            edge_index, max_num_nodes=num_nodes
        )  # 1 x (Num nodes) x (Num nodes)
    rws = []
    if ksteps == list(range(min(ksteps), max(ksteps) + 1)):
        # Efficient way if ksteps are a consecutive sequence (most of the time the case)
        Pk = P.clone().detach().matrix_power(min(ksteps))
        for k in range(min(ksteps), max(ksteps) + 1):
            rws.append(torch.diagonal(Pk, dim1=-2, dim2=-1) * (k ** (space_dim / 2)))
            Pk = Pk @ P
    else:
        # Explicitly raising P to power k for each k \in ksteps.
        for k in ksteps:
            rws.append(
                torch.diagonal(P.matrix_power(k), dim1=-2, dim2=-1)
                * (k ** (space_dim / 2))
            )
    rw_landing = torch.cat(rws, dim=0).transpose(0, 1)  # (Num nodes) x (K steps)

    return rw_landing


def get_heat_kernels_diag(evects, evals, kernel_times=[], space_dim=0):
    """Compute Heat kernel diagonal.

    This is a continuous function that represents a Gaussian in the Euclidean
    space, and is the solution to the diffusion equation.
    The random-walk diagonal should converge to this.

    Args:
        evects: Eigenvectors of the Laplacian matrix
        evals: Eigenvalues of the Laplacian matrix
        kernel_times: Time for the diffusion. Analogous to the k-steps in random
            walk. The time is equivalent to the variance of the kernel.
        space_dim: (optional) Estimated dimensionality of the space. Used to
            correct the diffusion diagonal by a factor `t^(space_dim/2)`. In
            euclidean space, this correction means that the height of the
            gaussian stays constant across time, if `space_dim` is the dimension
            of the euclidean space.

    Returns:
        2D Tensor with shape (num_nodes, len(ksteps)) with RW landing probs
    """
    heat_kernels_diag = []
    if len(kernel_times) > 0:
        evects = F.normalize(evects, p=2.0, dim=0)

        # Remove eigenvalues == 0 from the computation of the heat kernel
        idx_remove = evals < 1e-8
        evals = evals[~idx_remove]
        evects = evects[:, ~idx_remove]

        # Change the shapes for the computations
        evals = evals.unsqueeze(-1)  # lambda_{i, ..., ...}
        evects = evects.transpose(0, 1)  # phi_{i,j}: i-th eigvec X j-th node

        # Compute the heat kernels diagonal only for each time
        eigvec_mul = evects**2
        for t in kernel_times:
            # sum_{i>0}(exp(-2 t lambda_i) * phi_{i, j} * phi_{i, j})
            this_kernel = torch.sum(
                torch.exp(-t * evals) * eigvec_mul, dim=0, keepdim=False
            )

            # Multiply by `t` to stabilize the values, since the gaussian height
            # is proportional to `1/t`
            heat_kernels_diag.append(this_kernel * (t ** (space_dim / 2)))
        heat_kernels_diag = torch.stack(heat_kernels_diag, dim=0).transpose(0, 1)

    return heat_kernels_diag


def get_heat_kernels(evects, evals, kernel_times=[]):
    """Compute full Heat diffusion kernels.

    Args:
        evects: Eigenvectors of the Laplacian matrix
        evals: Eigenvalues of the Laplacian matrix
        kernel_times: Time for the diffusion. Analogous to the k-steps in random
            walk. The time is equivalent to the variance of the kernel.
    """
    heat_kernels, rw_landing = [], []
    if len(kernel_times) > 0:
        evects = F.normalize(evects, p=2.0, dim=0)

        # Remove eigenvalues == 0 from the computation of the heat kernel
        idx_remove = evals < 1e-8
        evals = evals[~idx_remove]
        evects = evects[:, ~idx_remove]

        # Change the shapes for the computations
        evals = evals.unsqueeze(-1).unsqueeze(-1)  # lambda_{i, ..., ...}
        evects = evects.transpose(0, 1)  # phi_{i,j}: i-th eigvec X j-th node

        # Compute the heat kernels for each time
        eigvec_mul = evects.unsqueeze(2) * evects.unsqueeze(
            1
        )  # (phi_{i, j1, ...} * phi_{i, ..., j2})
        for t in kernel_times:
            # sum_{i>0}(exp(-2 t lambda_i) * phi_{i, j1, ...} * phi_{i, ..., j2})
            heat_kernels.append(
                torch.sum(torch.exp(-t * evals) * eigvec_mul, dim=0, keepdim=False)
            )

        heat_kernels = torch.stack(
            heat_kernels, dim=0
        )  # (Num kernel times) x (Num nodes) x (Num nodes)

        # Take the diagonal of each heat kernel,
        # i.e. the landing probability of each of the random walks
        rw_landing = torch.diagonal(heat_kernels, dim1=-2, dim2=-1).transpose(
            0, 1
        )  # (Num nodes) x (Num kernel times)

    return heat_kernels, rw_landing


def get_electrostatic_function_encoding(edge_index, num_nodes):
    """Kernel based on the electrostatic interaction between nodes."""
    L = to_scipy_sparse_matrix(
        *get_laplacian(edge_index, normalization=None, num_nodes=num_nodes)
    ).todense()
    L = torch.as_tensor(L)
    Dinv = torch.eye(L.shape[0]) * (L.diag() ** -1)
    A = deepcopy(L).abs()
    A.fill_diagonal_(0)
    DinvA = Dinv.matmul(A)

    electrostatic = torch.pinverse(L)
    electrostatic = electrostatic - electrostatic.diag()
    green_encoding = torch.stack(
        [
            electrostatic.min(dim=0)[0],  # Min of Vi -> j
            electrostatic.max(dim=0)[0],  # Max of Vi -> j
            electrostatic.mean(dim=0),  # Mean of Vi -> j
            electrostatic.std(dim=0),  # Std of Vi -> j
            electrostatic.min(dim=1)[0],  # Min of Vj -> i
            electrostatic.max(dim=0)[0],  # Max of Vj -> i
            electrostatic.mean(dim=1),  # Mean of Vj -> i
            electrostatic.std(dim=1),  # Std of Vj -> i
            (DinvA * electrostatic).sum(
                dim=0
            ),  # Mean of interaction on direct neighbour
            (DinvA * electrostatic).sum(
                dim=1
            ),  # Mean of interaction from direct neighbour
        ],
        dim=1,
    )

    return green_encoding


def eigvec_normalizer(EigVecs, EigVals, normalization="L2", eps=1e-12):
    """
    Implement different eigenvector normalizations.
    """

    EigVals = EigVals.unsqueeze(0)

    if normalization == "L1":
        # L1 normalization: eigvec / sum(abs(eigvec))
        denom = EigVecs.norm(p=1, dim=0, keepdim=True)

    elif normalization == "L2":
        # L2 normalization: eigvec / sqrt(sum(eigvec^2))
        denom = EigVecs.norm(p=2, dim=0, keepdim=True)

    elif normalization == "abs-max":
        # AbsMax normalization: eigvec / max|eigvec|
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values

    elif normalization == "wavelength":
        # AbsMax normalization, followed by wavelength multiplication:
        # eigvec * pi / (2 * max|eigvec| * sqrt(eigval))
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom * 2 / np.pi

    elif normalization == "wavelength-asin":
        # AbsMax normalization, followed by arcsin and wavelength multiplication:
        # arcsin(eigvec / max|eigvec|)  /  sqrt(eigval)
        denom_temp = (
            torch.max(EigVecs.abs(), dim=0, keepdim=True)
            .values.clamp_min(eps)
            .expand_as(EigVecs)
        )
        EigVecs = torch.asin(EigVecs / denom_temp)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = eigval_denom

    elif normalization == "wavelength-soft":
        # AbsSoftmax normalization, followed by wavelength multiplication:
        # eigvec / (softmax|eigvec| * sqrt(eigval))
        denom = (F.softmax(EigVecs.abs(), dim=0) * EigVecs.abs()).sum(
            dim=0, keepdim=True
        )
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom

    else:
        raise ValueError(f"Unsupported normalization `{normalization}`")

    denom = denom.clamp_min(eps).expand_as(EigVecs)
    EigVecs = EigVecs / denom

    return EigVecs
