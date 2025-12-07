import argparse
import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelBinarizer
from sklearn.cluster import KMeans
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import lil_matrix, bmat, issparse
import ot
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

# JAX for Optimal Transport
import jax
import jax.numpy as jnp
from ott.geometry import geometry
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
device="cuda:0" if torch.cuda.is_available() else "cpu" 

# ==========================================
# 1. 指标计算 (对应论文 Fig 2H)
# ==========================================
def calculate_benchmark_metrics(pred_adata, gt_adata, scalers, w=0.9):
    """
    使用 POT 库计算精确的 Wasserstein 距离 (W2)。
    无需自己写 Solver，一行代码解决。
    """
    print("  [Metric] Calculating 2-Wasserstein metrics (Exact EMD)...")
    
    # 1. 准备数据 (Numpy)
    # 预测数据
    pred_coords = pred_adata.obsm['spatial']  # [N, 2]
    pred_expr = pred_adata.X     # [N, D]
    
    # 真实数据
    gt_coords = gt_adata['coords']            # [M, 2]
    gt_expr = gt_adata['expr']                # [M, 50]
    
    # 2. 定义权重 (假设每个细胞权重相等)
    n = pred_coords.shape[0]
    m = gt_coords.shape[0]
    a = np.ones((n,)) / n  # 预测分布权重
    b = np.ones((m,)) / m  # 真实分布权重

    # =========================================================
    # 指标 A: Gene Expression Only (W2_gene)
    # =========================================================
    # 计算代价矩阵 M (Squared Euclidean Distance)
    # cdist 直接计算两组点之间的距离矩阵，'sqeuclidean' 表示平方欧氏距离
    M_gene = cdist(pred_expr, gt_expr, metric='sqeuclidean')
    
    # ot.emd2 直接返回 Wasserstein 距离的平方 (因为我们输入的是平方距离矩阵)
    # 这里的 emd2 求解的是标准的线性规划问题，给出精确解
    w2_sq_gene = ot.emd2(a, b, M_gene)
    w2_gene = np.sqrt(w2_sq_gene)

    # =========================================================
    # 指标 B: Combined (Gene + Spatial)
    # =========================================================
    # 计算空间代价矩阵
    M_spatial = cdist(pred_coords, gt_coords, metric='sqeuclidean')
    
    # 混合代价矩阵 (根据论文公式)
    M_combined = (1 - w) * M_gene + w * M_spatial
    
    # 计算混合 W2
    w2_sq_combined = ot.emd2(a, b, M_combined)
    w2_combined = np.sqrt(w2_sq_combined)
    
    return float(w2_gene), float(w2_combined)

# ==========================================
# 2. 数据预处理与加载 (保持不变)
# ==========================================
def preprocess_multislice(adata_list, time_points, n_top_genes=2000, n_pca=50, use_spatial_split=False,use_pca=False):
    """处理多个切片，统一进行 PCA 和标准化。"""
    print("Concatenating all datasets for global normalization...")
    adata_list = align_spatial_slices(adata_list, time_points)
    for adata, t in zip(adata_list, time_points):
        adata.obs['time_point'] = t
        
    adata_concat = ad.concat(adata_list, label="batch_time", keys=[str(t) for t in time_points], index_unique="-")
    
    if 'Annotation' in adata_concat.obs:
        raw_annotations = adata_concat.obs['Annotation'].astype(str).fillna("Unknown").values
        lb = LabelBinarizer()
        one_hot_all = lb.fit_transform(raw_annotations)
    else:
        # 如果没有注释，创建一个假的全 1
        one_hot_all = np.zeros((adata_concat.n_obs, 1))

    # 基础预处理
    if 'log1p' not in adata_concat.uns:
        sc.pp.normalize_total(adata_concat, target_sum=1e4)
        sc.pp.log1p(adata_concat)
        
    if n_top_genes is not None:
        sc.pp.highly_variable_genes(adata_concat, n_top_genes=n_top_genes, subset=True)
        
    if use_pca:
        print(f"Running PCA (n_comps={n_pca})...")
        sc.tl.pca(adata_concat, n_comps=n_pca)
        expr_all = adata_concat.obsm['X_pca']
    else:
        print("Skipping PCA, using HVG expression directly...")
        # 如果是稀疏矩阵，转换为密集矩阵
        if issparse(adata_concat.X):
            expr_all = adata_concat.X.toarray()
        else:
            expr_all = adata_concat.X
    feat_dim = expr_all.shape[1]
    print(f"Feature dimension: {feat_dim}")
    
    mass_all = adata_concat.obs['mass'].values if 'mass' in adata_concat.obs else np.ones(adata_concat.n_obs)
    e_mean, e_std = np.mean(expr_all, axis=0), np.std(expr_all, axis=0) + 1e-6
    expr_norm_all = (expr_all - e_mean) / e_std
    scalers = { 'e_mean': e_mean, 'e_std': e_std}
    
    processed_data_dict = {} # 改成字典方便按时间存取
    start_idx = 0
    for t, adata in zip(time_points, adata_list):
        n = adata.n_obs
        coords_aligned = adata.obsm['spatial_aligned']
        processed_data_dict[t] = {
            'coords': coords_aligned,
            'expr': expr_norm_all[start_idx : start_idx+n],
            'mass': mass_all[start_idx : start_idx+n],
            'obs_names': adata.obs_names,
            'type_onehot': one_hot_all[start_idx : start_idx+n],
            'n_cells': n
        }
        start_idx += n
        
    return processed_data_dict, scalers,feat_dim

# ==========================================
# 3. OT 计算辅助函数
# ==========================================
def _pairwise_squared_distances(x, y):
    x_sq = jnp.sum(jnp.square(x), axis=1, keepdims=True)
    y_sq = jnp.sum(jnp.square(y), axis=1)
    distances = x_sq + y_sq - 2.0 * jnp.matmul(x, y.T)
    return jnp.maximum(distances, 0.0)
# ==========================================
# 新增：空间对齐辅助函数
# ==========================================
def align_spatial_slices(adata_list, time_points):
 
    print("Aligning spatial coordinates across slices...")
    
    aligned_coords_list = []
    
    for i, adata in enumerate(adata_list):
        coords = adata.obsm['spatial'].copy()
        
        # Center
        c_mean = np.mean(coords, axis=0)
        coords = coords - c_mean
        
        scale = np.max(np.std(coords, axis=0)) + 1e-8
        coords = coords / scale
        
        adata.obsm['spatial_aligned'] = coords # 暂存
    
    
    for i in range(len(adata_list)):
        coords = adata_list[i].obsm['spatial_aligned']
        
        pca =  KMeans(n_clusters=1, n_init=1, max_iter=1).fit(coords) # 只是为了拿到中心，其实已经是0了
        u, s, vh = np.linalg.svd(coords.T @ coords)
        
        angle = np.arctan2(vh[0, 1], vh[0, 0])
        R = np.array([
            [np.cos(-angle), -np.sin(-angle)],
            [np.sin(-angle),  np.cos(-angle)]
        ])
        coords_rotated = coords @ R.T
        
        if i > 0:
            prev_coords = adata_list[i-1].obsm['spatial_aligned']
            
            
            best_coords = coords_rotated
            min_dist = np.inf
            
            # 测试 4 种翻转: (x,y), (-x,y), (x,-y), (-x,-y)
            transforms = [
                np.array([1, 1]), np.array([-1, 1]), 
                np.array([1, -1]), np.array([-1, -1])
            ]
            
            n_sub = min(500, coords_rotated.shape[0], prev_coords.shape[0])
            idx_curr = np.random.choice(coords_rotated.shape[0], n_sub, replace=False)
            idx_prev = np.random.choice(prev_coords.shape[0], n_sub, replace=False)
            sub_curr = coords_rotated[idx_curr]
            sub_prev = prev_coords[idx_prev]
            
            for t_scale in transforms:
                temp_coords = sub_curr * t_scale
                d_matrix = cdist(temp_coords, sub_prev)
                chamfer_dist = np.mean(np.min(d_matrix, axis=1)) + np.mean(np.min(d_matrix, axis=0))
                
                if chamfer_dist < min_dist:
                    min_dist = chamfer_dist
                    best_transform = t_scale
            
            coords_rotated = coords_rotated * best_transform
            
        adata_list[i].obsm['spatial_aligned'] = coords_rotated
        print(f"  Slice t={time_points[i]}: Aligned. (std={np.std(coords_rotated):.3f})")
    return adata_list
def visualize_geodesic_pairing(coords1, coords2, cross_graph, dist_geo, ot_matrix=None, save_path=None):
    """
    可视化测地线配对关系 (修改版：加入人工平移以解决坐标重叠问题)
    """
    fig = plt.figure(figsize=(20, 5))
    
    # 1. 计算平移偏移量 (Shift)
    # 获取 coords1 的 X 轴范围
    x_min, x_max = np.min(coords1[:, 0]), np.max(coords1[:, 0])
    width = x_max - x_min
    if width == 0: width = 1.0
    
    # 设置偏移量，让 Target 显示在 Source 的右侧，中间留一点空隙
    shift_x = width * 1.5
    
    # 创建用于绘图的 Target 坐标副本
    coords2_vis = coords2.copy()
    coords2_vis[:, 0] += shift_x
    
    # =========================================
    # 子图1: 空间分布 + Cross Graph 连接
    # =========================================
    ax1 = plt.subplot(1, 4, 1)
    
    # 画点
    ax1.scatter(coords1[:, 0], coords1[:, 1], c='red', s=10, alpha=0.5, label='Source (t)', marker='o', edgecolors='none')
    ax1.scatter(coords2_vis[:, 0], coords2_vis[:, 1], c='blue', s=10, alpha=0.5, label='Target (t+1)', marker='s', edgecolors='none')
    
    # 标注 Time Point
    ax1.text(np.mean(coords1[:, 0]), np.max(coords1[:, 1]) + width*0.1, "Time t", ha='center', fontsize=10, fontweight='bold')
    ax1.text(np.mean(coords2_vis[:, 0]), np.max(coords2_vis[:, 1]) + width*0.1, "Time t+1", ha='center', fontsize=10, fontweight='bold')
    
    # 绘制 cross_graph 连接 (Geodesic 构建时的临近连接)
    # 为了避免太乱，只显示部分点的连接
    n1 = coords1.shape[0]
    n_show = min(100, n1)
    indices_to_show = np.linspace(0, n1-1, n_show, dtype=int)
    
    # 获取稀疏矩阵的连接关系
    # cross_graph 是 [N1, N2] 的稀疏矩阵
    # 我们需要找到非零元素
    
    cx = cross_graph.tocsr() # 转为 CSR 加速切片
    
    for i in indices_to_show:
        # 找到第 i 个 Source 点连接的所有 Target 点
        # cross_graph[i, :] 非零的索引
        row = cx.getrow(i)
        target_indices = row.indices # 连接到的 Target 索引
        
        for j in target_indices:
            ax1.plot([coords1[i, 0], coords2_vis[j, 0]], 
                    [coords1[i, 1], coords2_vis[j, 1]], 
                    'g-', alpha=0.2, linewidth=0.5)
    
    ax1.set_title('Geodesic Graph Construction\n(Green lines = Cross Neighbors)', fontsize=10)
    ax1.legend(loc='lower right', fontsize=8)
    ax1.axis('off') #以此模式通常不需要坐标轴刻度
    
    # =========================================
    # 子图2: 测地线距离矩阵热力图 (不变)
    # =========================================
    ax2 = plt.subplot(1, 4, 2)
    n_show1 = min(200, dist_geo.shape[0])
    n_show2 = min(200, dist_geo.shape[1])
    # 这里的距离是真实的测地线距离，不需要平移
    im = ax2.imshow(dist_geo[:n_show1, :n_show2], aspect='auto', cmap='viridis', origin='lower')
    ax2.set_title('Geodesic Distance Matrix\n(Subset)', fontsize=10)
    ax2.set_xlabel('Target Index')
    ax2.set_ylabel('Source Index')
    plt.colorbar(im, ax=ax2, label='Distance')
    
    # =========================================
    # 子图3: OT 配对关系 (应用平移)
    # =========================================
    ax3 = plt.subplot(1, 4, 3)
    if ot_matrix is not None:
        # 画点
        ax3.scatter(coords1[:, 0], coords1[:, 1], c='red', s=10, alpha=0.5, marker='o', edgecolors='none')
        ax3.scatter(coords2_vis[:, 0], coords2_vis[:, 1], c='blue', s=10, alpha=0.5, marker='s', edgecolors='none')
        
        # 找到主要配对
        # 这里的 OT Matrix 通常比较稠密(entropic)，我们只画权重最高的连线
        
        # 策略：对于每个 Source 点，画出权重最大的那个 Target 连接
        # 为了避免画几千条线，我们只随机采样一些 Source 点
        n_lines_show = 150
        sample_indices = np.random.choice(ot_matrix.shape[0], min(n_lines_show, ot_matrix.shape[0]), replace=False)
        
        for i in sample_indices:
            # 找到该行最大的权重的列索引 (argmax)
            j = np.argmax(ot_matrix[i, :])
            weight = ot_matrix[i, j]
            
            # 只有当权重足够大时才画 (相对值)
            if weight > 1e-8:
                ax3.plot([coords1[i, 0], coords2_vis[j, 0]], 
                        [coords1[i, 1], coords2_vis[j, 1]], 
                        'purple', alpha=0.4, linewidth=0.8)
        
        ax3.set_title(f'Optimal Transport Map\n(Top-1 connection for random {min(n_lines_show, ot_matrix.shape[0])} cells)', fontsize=10)
    else:
        ax3.text(0.5, 0.5, 'No OT Matrix', ha='center', va='center')
        
    ax3.axis('off')
    
    # =========================================
    # 子图4: OT 矩阵热力图 (不变)
    # =========================================
    ax4 = plt.subplot(1, 4, 4)
    if ot_matrix is not None:
        n_show1 = min(200, ot_matrix.shape[0])
        n_show2 = min(200, ot_matrix.shape[1])
        im = ax4.imshow(ot_matrix[:n_show1, :n_show2], aspect='auto', cmap='magma', origin='lower')
        ax4.set_title('OT Coupling Matrix\n(Subset)', fontsize=10)
        ax4.set_xlabel('Target Index')
        ax4.set_ylabel('Source Index')
        plt.colorbar(im, ax=ax4, label='Probability')
    else:
        ax4.text(0.5, 0.5, 'No OT Matrix', ha='center', va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [Visual] Saved OT visualization to: {save_path}")
    else:
        plt.show()
    
    plt.close(fig)

def compute_geodesic_cost_matrix(coords1, coords2, n_neighbors=15, n_cross_neighbors=5, return_intermediate=False):
    """
    计算测地线距离，并强制保证 Source 和 Target 之间的连通性。
    
    Parameters:
    -----------
    coords1 : np.ndarray
        Source 点集坐标
    coords2 : np.ndarray
        Target 点集坐标
    n_neighbors : int
        每个点集内部的邻居数
    n_cross_neighbors : int
        跨集连接的邻居数
    return_intermediate : bool
        是否返回中间结果（cross_graph, dist_geo）用于可视化
    
    Returns:
    --------
    result : np.ndarray
        测地线距离矩阵的平方 [N1, N2]
    (cross_graph, dist_geo) : tuple, optional
        如果 return_intermediate=True，返回中间结果
    """
    n1 = coords1.shape[0]
    n2 = coords2.shape[0]
    
    g1 = kneighbors_graph(coords1, n_neighbors=n_neighbors, mode='distance', include_self=False)
    g2 = kneighbors_graph(coords2, n_neighbors=n_neighbors, mode='distance', include_self=False)
    
    nbrs_cross = NearestNeighbors(n_neighbors=n_cross_neighbors, algorithm='auto').fit(coords2)
    distances_cross, indices_cross = nbrs_cross.kneighbors(coords1)
    
    cross_graph = lil_matrix((n1, n2))
    

    for i in range(n1):
        for k in range(n_cross_neighbors):
            j = indices_cross[i, k]
            d = distances_cross[i, k]
            cross_graph[i, j] = d
            
    from scipy.sparse import bmat
    combined_graph = bmat([
        [g1, cross_graph], 
        [cross_graph.T, g2]
    ])
    
    dist_matrix_all = shortest_path(csgraph=combined_graph, directed=False, return_predecessors=False)
    dist_geo = dist_matrix_all[:n1, n1:]
    finite_mask = np.isfinite(dist_geo)
    if not np.any(finite_mask):
        print(" [Warning] Geodesic graph still totally disconnected! Fallback to Euclidean.")
        result = cdist(coords1, coords2, metric='sqeuclidean')
        if return_intermediate:
            return result, (None, None)
        return result
    
    max_dist = np.max(dist_geo[finite_mask])
    dist_geo[~finite_mask] = max_dist * 1.5
    result = dist_geo ** 2
    
    if return_intermediate:
        return result, (cross_graph, dist_geo)
    return result
def compute_unbalanced_ot(d0, d1, tau=0.8, epsilon=0.05, lambda_type=0.0, use_geodesic=False, visualize=False, save_path=None):
    """
    封装 OT 计算
    
    Parameters:
    -----------
    d0 : dict
        Source 数据字典，包含 'coords', 'expr', 'type_onehot', 'mass'
    d1 : dict
        Target 数据字典，包含 'coords', 'expr', 'type_onehot', 'mass'
    tau : float
        Unbalanced OT 参数
    epsilon : float
        Sinkhorn 正则化参数
    lambda_type : float
        类型约束权重
    use_geodesic : bool
        是否使用测地线距离
    visualize : bool
        是否进行可视化
    save_path : str, optional
        可视化保存路径
    
    Returns:
    --------
    ot_matrix : np.ndarray
        最优传输矩阵 [N1, N2]
    """
    coords_x, coords_y = jnp.array(d0['coords']), jnp.array(d1['coords'])
    coords_x_np,coords_y_np=np.array(d0['coords']),np.array(d1['coords'])
    
    expr_x, expr_y = jnp.array(d0['expr']), jnp.array(d1['expr'])
    type_x, type_y = jnp.array(d0['type_onehot']), jnp.array(d1['type_onehot'])
    
    cross_graph = None
    dist_geo = None
    
    if use_geodesic:
        if visualize:
            dist_c_np, (cross_graph, dist_geo) = compute_geodesic_cost_matrix(
                coords_x_np, coords_y_np, n_neighbors=15, return_intermediate=True)
        else:
            dist_c_np = compute_geodesic_cost_matrix(coords_x_np, coords_y_np, n_neighbors=15)
        dist_c = jnp.array(dist_c_np)
    else:
        coords_x, coords_y = jnp.array(coords_x_np), jnp.array(coords_y_np)
        dist_c = _pairwise_squared_distances(coords_x, coords_y)
    dist_e = _pairwise_squared_distances(expr_x, expr_y)
    
    similarity_matrix = jnp.matmul(type_x, type_y.T)
    dist_type = 1.0 - similarity_matrix
    mean_c = jnp.mean(dist_c) + 1e-8
    mean_e = jnp.mean(dist_e) + 1e-8
    cost = (dist_c / mean_c) + (dist_e / mean_e) + lambda_type * dist_type
    # raw_cost = dist_c + dist_e + lambda_type * dist_type
    # scale_factor = jnp.mean(raw_cost) + 1e-8
    # cost = raw_cost / scale_factor

    a = jnp.array(d0['mass']) / np.sum(d0['mass'])
    b = jnp.array(d1['mass']) / np.sum(d1['mass'])
    
    geom = geometry.Geometry(cost_matrix=cost, epsilon=epsilon)
    prob = linear_problem.LinearProblem(geom, a=a, b=b, tau_a=tau, tau_b=tau)
    solver = sinkhorn.Sinkhorn()
    out = solver(prob)
    ot_matrix = np.array(out.matrix)
    
    # 可视化
    if visualize and use_geodesic and cross_graph is not None and dist_geo is not None:
        if save_path is None:
            save_path = "geodesic_pairing_visualization.png"
        visualize_geodesic_pairing(coords_x_np, coords_y_np, cross_graph, dist_geo, ot_matrix, save_path)
    
    return ot_matrix

# ==========================================
# 4. 模型与训练 (支持传入 subset list)
# ==========================================
class GaussianFourierProjection(nn.Module):
    """用于对时间 t 进行高维映射，这是 Flow Matching/Diffusion 收敛的关键"""
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.act = nn.GELU() # GELU 比 ReLU 表现更好
        self.linear2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.ln1(x)
        h = self.linear1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.ln2(h)
        h = self.linear2(h)
        return x + h  # Skip Connection

class MassFlowMatching(nn.Module):
    def __init__(self, coord_dim=2, expression_dim=50, hidden_dim=512, time_embed_dim=64):
        super().__init__()
        
        # 输入维度：坐标 + 基因 + 质量
        input_dim = coord_dim + expression_dim + 1
        
        # 时间编码
        self.time_embed = GaussianFourierProjection(time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, hidden_dim)
        )
        
        # 初始映射
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # 核心网络 (3个残差块)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim)
        ])
        
        self.final_norm = nn.LayerNorm(hidden_dim)
        
        # 输出头
        self.spatial_head = nn.Linear(hidden_dim, coord_dim)
        self.expression_head = nn.Linear(hidden_dim, expression_dim)
        self.mass_head = nn.Linear(hidden_dim, 1)

    def forward(self, coords, expression, mass, t):
        # t 应该是归一化后的 [0, 1]
        if t.dim() == 1: t = t.unsqueeze(-1)
        if mass.dim() == 1: mass = mass.unsqueeze(-1)
        
        # 1. 处理时间
        t_emb = self.time_embed(t.squeeze(-1))
        t_emb = self.time_mlp(t_emb) # [B, hidden_dim]
        
        # 2. 处理状态输入
        x = torch.cat([coords, expression, mass], dim=-1)
        h = self.input_proj(x) # [B, hidden_dim]
        
        # 3. 融合时间与状态 (AdaGN 的简化版，直接相加)
        h = h + t_emb
        
        # 4. 残差网络
        for block in self.blocks:
            h = block(h)
            
        h = self.final_norm(h)
        
        # 5. 输出
        return self.spatial_head(h), self.expression_head(h), self.mass_head(h)

class OTSampler:
    def __init__(self, ot_matrix):
        self.ot_matrix = np.array(ot_matrix, dtype=np.float32)
        self.row_sums = self.ot_matrix.sum(axis=1) + 1e-10
        self.conditional_probs = self.ot_matrix / self.row_sums[:, None]
        zero_rows = (self.row_sums < 1e-8)
        self.conditional_probs[zero_rows] = 1.0 / self.ot_matrix.shape[1]
        self.conditional_probs_tensor = torch.tensor(self.conditional_probs, device=device)
        self.row_weights = torch.tensor(self.row_sums, device=device)
        self.n_source = self.ot_matrix.shape[0]

    def sample(self, batch_size):
        row_indices = torch.randint(0, self.n_source, (batch_size,), device=device)
        col_indices = torch.multinomial(self.conditional_probs_tensor[row_indices], num_samples=1).squeeze()
        weights = self.row_weights[row_indices]
        return row_indices.cpu().numpy(), col_indices.cpu().numpy(), weights.cpu().numpy()

def train_model(train_data_list, train_times, feat_dim,epochs=1000, batch_size=256, visualize_ot=False):
    ot_matrices = []
    samplers = []
    
    # 计算全局时间范围用于归一化
    t_min = min(train_times)
    t_max = max(train_times)
    t_range = t_max - t_min
    if t_range < 1e-6:
        t_range = 1.0  # 避免除零
    
    # print(f"  Computing OT matrices for sequence: {train_times}")
    for i in range(len(train_data_list) - 1):
        d0 = train_data_list[i]
        d1 = train_data_list[i+1]
        # 只可视化第一个 OT 矩阵
        vis = visualize_ot
        save_path = f"geodesic_pairing_t{train_times[i]}_to_t{train_times[i+1]}.png" if vis else None
        ot = compute_unbalanced_ot(d0, d1, epsilon=0.01, visualize=vis, save_path=save_path)
        samplers.append(OTSampler(ot))
        ot_matrices.append(ot)
        
    model = MassFlowMatching(expression_dim=feat_dim, hidden_dim=256)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    n_intervals = len(ot_matrices)
    
    model.train() # 确保在训练模式
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss = 0
        interval_idx = np.random.randint(0, n_intervals)
        sampler = samplers[interval_idx]
        
        idx0, idx1, weights = sampler.sample(batch_size)
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device).unsqueeze(1)
        
        d0 = train_data_list[interval_idx]
        d1 = train_data_list[interval_idx+1]
        
        # 获取当前时间区间的归一化时间范围
        t_start = train_times[interval_idx]
        t_end = train_times[interval_idx+1]
        t_start_norm = (t_start - t_min) / t_range
        t_end_norm = (t_end - t_min) / t_range
        
        # B. 准备数据
        c0 = torch.tensor(d0['coords'][idx0], dtype=torch.float32, device=device)
        e0 = torch.tensor(d0['expr'][idx0], dtype=torch.float32, device=device)
        m0 = torch.tensor(d0['mass'][idx0], dtype=torch.float32, device=device).unsqueeze(1)
        
        c1 = torch.tensor(d1['coords'][idx1], dtype=torch.float32, device=device)
        e1 = torch.tensor(d1['expr'][idx1], dtype=torch.float32, device=device)
        m1 = torch.tensor(d1['mass'][idx1], dtype=torch.float32, device=device).unsqueeze(1)
        
        # C. 构造中间态 - 使用全局归一化的绝对时间
        # alpha 是区间内的相对位置 [0, 1]
        alpha = torch.rand(batch_size, 1, device=device)
        # u 是全局归一化的绝对时间 [t_start_norm, t_end_norm]
        u = t_start_norm + alpha * (t_end_norm - t_start_norm)
        
        c_t = (1 - alpha) * c0 + alpha * c1
        e_t = (1 - alpha) * e0 + alpha * e1
        m_t = (1 - alpha) * m0 + alpha * m1
        
        # D. 计算 Target
        # D.1 速度 Target
        v_s_target = c1 - c0
        v_e_target = e1 - e0
        n_cells_0 = d0['n_cells']
        n_cells_1 = d1['n_cells']
        
        raw_growth = weights_tensor * n_cells_1
        global_ratio = n_cells_1 / n_cells_0
        relative_growth = raw_growth / (global_ratio + 1e-6)
        
        target_growth_factor = torch.clamp(relative_growth * global_ratio, 0.1, 10.0)
        
        k_target = torch.log(target_growth_factor)
        
        # E. 前向传播
        # 注意这里接收 k_pred
        v_s_pred, v_e_pred, k_pred = model(c_t, e_t, m_t, u)
        
        loss_s = torch.mean((v_s_pred - v_s_target)**2)
        loss_e = torch.mean((v_e_pred - v_e_target)**2)
        loss_m = torch.mean((k_pred - k_target)**2)
        
        # F.4 总 Loss (加权求和，防止权重总和波动)
        lambda_s = 1.0
        lambda_e = 1.0
        lambda_m = 1.0
        loss = (lambda_s * loss_s + lambda_e * loss_e + lambda_m * loss_m) 
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.6f} | S: {loss_s.item():.6f} | E: {loss_e.item():.6f} | M: {loss_m.item():.6f}")
      
    return model, t_min, t_range
# ==========================================
# 5. 预测函数 (返回结果对象)
# ==========================================
def predict_state(model, start_data, start_time, target_time, t_min, t_range):
    """
    从 start_time 演化到 target_time (包含细胞分裂与凋亡逻辑)
    使用与训练时相同的全局时间归一化
    """
    # 1. 初始化细胞列表
    # 将 numpy 数组转为 list of dicts，方便动态增删
    current_cells = []
    n_start = start_data['n_cells']
    for i in range(n_start):
        current_cells.append({
            'coords': start_data['coords'][i],
            'expr': start_data['expr'][i],
            'mass': start_data['mass'][i],
            'id': f"cell_{i}"
        })
    
    # 2. 计算归一化的时间范围（与训练时保持一致）
    start_time_norm = (start_time - t_min) / t_range
    target_time_norm = (target_time - t_min) / t_range
    total_time_delta_norm = target_time_norm - start_time_norm
    
    simulation_steps = 20 # 将总时间切分为 20 步
    dt_u = total_time_delta_norm / simulation_steps # 归一化时间的步长
    
    model.eval()
    
    print(f"    Simulating {start_time} -> {target_time} (Delta T={target_time - start_time})")
    print(f"    Normalized: {start_time_norm:.4f} -> {target_time_norm:.4f} (Delta U={total_time_delta_norm:.4f})")
    
    u = start_time_norm
    for step in range(simulation_steps):
        if len(current_cells) == 0: break
            
        # 将当前活着的细胞堆叠成 Tensor
        coords = torch.tensor(np.array([c['coords'] for c in current_cells]), dtype=torch.float32, device=device)
        expr = torch.tensor(np.array([c['expr'] for c in current_cells]), dtype=torch.float32, device=device)
        mass = torch.tensor(np.array([c['mass'] for c in current_cells]), dtype=torch.float32, device=device).unsqueeze(1)
        t_tensor = torch.ones(len(current_cells), 1, device=device) * u
        
        # B. 模型预测
        with torch.no_grad():
            v_s, v_e, k_log_rate = model(coords, expr, mass, t_tensor)
        
        coords_new = coords + v_s * dt_u
        expr_new = expr + v_e * dt_u
        mass_new = mass * torch.exp(k_log_rate * dt_u)
        
        # 转回 Numpy
        coords_np = coords_new.cpu().numpy()
        expr_np = expr_new.cpu().numpy()
        mass_np = mass_new.squeeze(1).cpu().numpy()
        
        next_cells = []
        for i, old_c in enumerate(current_cells):
            m = mass_np[i]
            if m > 2.0: # 分裂
                child1 = {'coords': coords_np[i], 'expr': expr_np[i], 'mass': m/2, 'id': old_c['id']+"_1"}
                child2 = {'coords': coords_np[i], 'expr': expr_np[i], 'mass': m/2, 'id': old_c['id']+"_2"}
                next_cells.append(child1)
                next_cells.append(child2)
            elif m < 0.1: # 凋亡
                continue
            else:
                old_c['coords'] = coords_np[i]
                old_c['expr'] = expr_np[i]
                old_c['mass'] = m
                next_cells.append(old_c)
        
        current_cells = next_cells
        u += dt_u
        
    print(f"Simulation finished. Reconstructing AnnData with {len(current_cells)} cells...")
    if len(current_cells) == 0:
        print("Error: All cells died.")
        return

    final_expr = np.array([c['expr'] for c in current_cells])
    final_coords = np.array([c['coords'] for c in current_cells])
    final_mass = np.array([c['mass'] for c in current_cells])
    
    print(f"    Simulation finished. Final cell count: {len(current_cells)}")
    
    # 创建预测的AnnData对象
    pred_adata = ad.AnnData(X=final_expr)
    pred_adata.obsm['spatial'] = final_coords
    pred_adata.obs['mass'] = final_mass
    if final_expr.shape[1] == 50:
        pred_adata.obsm['X_pca'] = final_expr
    return pred_adata

# ==========================================
# 6. Leave-One-Out 主流程
# ==========================================
def run_leave_one_out_benchmark(args):
    files = args.files
    times = args.times
    
    # 1. 全局加载与预处理
    adata_list = [sc.read_h5ad(f) for f in files]
    
    # 按时间排序输入
    sorted_pairs = sorted(zip(times, adata_list))
    sorted_times = [p[0] for p in sorted_pairs]
    sorted_adatas = [p[1] for p in sorted_pairs]
    
    use_pca = not args.no_pca
    data_dict, scalers, feat_dim = preprocess_multislice(sorted_adatas, sorted_times, use_pca=use_pca)
    
    results = []
    
    # 2. 循环：依次将中间的时间点作为测试集
    # 我们不能去掉第一个(t_0)和最后一个(t_N)，因为那样就变成外推(Extrapolation)了
    # 这里的 Benchmark 主要是测插值(Interpolation)能力
    possible_targets = sorted_times[1:-1] 
    
    print(f"\n==============================================")
    print(f"Starting Leave-One-Out Benchmark")
    print(f"Mode: {'PCA (dim=50)' if use_pca else f'Raw Genes (dim={feat_dim})'}")
    print(f"All times: {sorted_times}")
    print(f"Targets to test: {possible_targets}")
    print(f"==============================================\n")
    
    for target_t in possible_targets:
        print(f"--- Processing Target: Day {target_t} ---")
        
        # A. 构建训练集 (移除 target_t)
        train_times = [t for t in sorted_times if t != target_t]
        train_data_list = [data_dict[t] for t in train_times]
        
        print(f"1. Training on sequence: {train_times}")
        # B. 训练
        model, t_min, t_range = train_model(train_data_list, train_times, feat_dim=feat_dim, epochs=args.epochs, visualize_ot=args.visualize_ot)
        
        # C. 预测
        # 找到训练集中最近的"过去"时间点
        past_times = [t for t in train_times if t < target_t]
        start_t = past_times[-1]
        start_data = data_dict[start_t]
        
        print(f"2. Predicting Day {target_t} (starting from Day {start_t})...")
        pred_adata = predict_state(model, start_data, start_t, target_t, t_min, t_range)
        
        # D. 计算指标
        gt_data = data_dict[target_t] # Ground Truth (Raw normalized dict)
        w2_gene, w2_comb = calculate_benchmark_metrics(pred_adata, gt_data, scalers)
        pred_adata.write_h5ad(f"data/day{target_t}_pred.h5ad")
        obs_df = pd.DataFrame(index=gt_data['obs_names'])
        obs_df['mass'] = gt_data['mass']
        obs_df['x'] =  gt_data['coords'][:, 0]
        obs_df['y'] =  gt_data['coords'][:, 1]
        gt_adata = ad.AnnData(X=gt_data['expr'], obs=obs_df)
        gt_adata.obsm['type_onehot'] = gt_data['type_onehot']
        gt_adata.obsm['spatial'] = gt_data['coords']
        gt_adata.uns['n_cells'] = gt_data['n_cells']
        gt_adata.write_h5ad(f"data/day{target_t}_gt.h5ad")
        print(f"3. Result for Day {target_t}:")
        print(f"   W2 (Gene):     {w2_gene:.4f}")
        print(f"   W2 (Combined): {w2_comb:.4f}")
        
        results.append({
            'Target_Time': target_t,
            'W2_Gene': w2_gene,
            'W2_Combined': w2_comb,
            'Train_Sequence': str(train_times)
        })
        print("----------------------------------------------")

    # 3. 输出总结
    df = pd.DataFrame(results)
    print("\nBenchmark Summary:")
    print(df)
    
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs='+', required=True, help="List of all h5ad files")
    parser.add_argument("--times", nargs='+', type=float, required=True, help="List of all time points")
    parser.add_argument("--epochs", type=int, default=1000, help="Epochs for each training fold")
    parser.add_argument("--output", type=str, default="benchmark_results.csv")
    parser.add_argument("--no_pca", action="store_true", help="If set, skip PCA and use HVGs directly")
    parser.add_argument("--visualize_ot", action="store_true", help="If set, visualize geodesic pairing for the first OT matrix")
    args = parser.parse_args()
    run_leave_one_out_benchmark(args)