import argparse
import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import NearestNeighbors

# JAX for Optimal Transport
import jax
import jax.numpy as jnp
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. 数据预处理与加载
# ==========================================

def preprocess_data(adata, n_top_genes=2000, n_pca=50, basis='spatial'):
    """
    标准化、对数化、PCA降维，并提取坐标和特征。
    """
    print(f"Preprocessing AnnData with {adata.n_obs} cells...")
    adata = adata.copy()
    
    # 基础预处理
    if 'log1p' not in adata.uns:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        
    # 高变基因选择
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, subset=True)
    
    # PCA 降维 (Flow Matching 在低维空间效果更好)
    sc.tl.pca(adata, n_comps=n_pca)
    
    # 提取特征
    # 使用 PCA 后的特征作为表达特征 (Expression Representation)
    expression = adata.obsm['X_pca']
    
    # 提取空间坐标
    if basis in adata.obsm:
        coords = adata.obsm[basis]
    else:
        print(f"Warning: {basis} not found, using first 2 PCs as spatial coords.")
        coords = adata.obsm['X_pca'][:, :2]
        
    # 提取或初始化质量 (Mass)
    if 'mass' in adata.obs:
        mass = adata.obs['mass'].values
    else:
        # 默认为1.0
        mass = np.ones(adata.n_obs)
        
    return coords, expression, mass, adata

# ==========================================
# 2. OT 阶段：计算 Unbalanced OT
# ==========================================

def compute_unbalanced_ot(coords_0, expr_0, coords_1, expr_1, mass_0, mass_1, tau=0.5, epsilon=0.1):
    """
    使用 OTT-JAX 计算 Unbalanced Sinkhorn Transport Matrix。
    """
    print("Computing Unbalanced Optimal Transport...")
    
    # 拼接坐标和表达特征用于计算代价矩阵
    # 注意：通常会对坐标和表达进行加权归一化，这里简化处理
    features_0 = np.hstack([coords_0, expr_0])
    features_1 = np.hstack([coords_1, expr_1])
    
    # 转换为 JAX 数组
    x = jnp.array(features_0)
    y = jnp.array(features_1)
    a = jnp.array(mass_0) / np.sum(mass_0) # 归一化分布
    b = jnp.array(mass_1) / np.sum(mass_1)
    
    # 定义几何 (Geometry)
    geom = pointcloud.PointCloud(x, y, epsilon=epsilon)
    
    # 定义线性问题 (Unbalanced)
    # tau_a, tau_b 控制边缘分布的松弛程度
    prob = linear_problem.LinearProblem(geom, a=a, b=b, tau_a=tau, tau_b=tau)
    
    # 求解 Sinkhorn
    try:
        solver = sinkhorn.Sinkhorn()
        out = solver(prob)
        transport_matrix = np.array(out.matrix)
    except Exception as err:
        print(f"Sinkhorn solver failed with {err}. Falling back to kNN transport.")
        transport_matrix = None
    
    transport_matrix = np.nan_to_num(transport_matrix, nan=0.0) if transport_matrix is not None else None
    if transport_matrix is not None:
        transport_matrix[transport_matrix < 0] = 0.0
        total_mass = transport_matrix.sum()
    else:
        total_mass = 0.0
    
    if transport_matrix is None or total_mass <= 0:
        print("Sinkhorn transport plan is empty. Using kNN-based heuristic transport.")
        transport_matrix = build_knn_transport(coords_0, expr_0, coords_1, expr_1, mass_0, mass_1)
        total_mass = transport_matrix.sum()
    
    transport_matrix = transport_matrix / max(total_mass, 1e-8)
    
    return transport_matrix


def build_knn_transport(coords_0, expr_0, coords_1, expr_1, mass_0, mass_1, k=10):
    """
    Fallback heuristic transport using feature-space kNN with Gaussian weights.
    """
    features_0 = np.hstack([coords_0, expr_0])
    features_1 = np.hstack([coords_1, expr_1])
    
    n0, n1 = features_0.shape[0], features_1.shape[0]
    k = min(k, n1)
    
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean')
    nbrs.fit(features_1)
    distances, indices = nbrs.kneighbors(features_0, return_distance=True)
    
    # 避免 sigma 为 0
    sigma = np.median(distances[distances > 0]) if np.any(distances > 0) else 1.0
    sigma = max(sigma, 1e-6)
    
    weights = np.exp(- (distances ** 2) / (2 * sigma ** 2))
    weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-8)
    
    transport = np.zeros((n0, n1), dtype=np.float64)
    for i in range(n0):
        transport[i, indices[i]] = weights[i] * mass_0[i]
    
    total = transport.sum()
    if total <= 0:
        # fallback to uniform coupling
        transport = np.ones((n0, n1), dtype=np.float64) / (n0 * n1)
    
    return transport

# ==========================================
# 3. 模型定义：Mass Flow Matching
# ==========================================

class MassFlowMatching(nn.Module):
    def __init__(self, coord_dim=2, expression_dim=50, hidden_dim=256):
        super().__init__()
        
        input_dim = coord_dim + expression_dim + 1 + 1 # coords + expr + mass + t
        
        self.state_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.spatial_head = nn.Linear(hidden_dim, coord_dim)
        self.expression_head = nn.Linear(hidden_dim, expression_dim)
        
        # 质量演化网络
        self.mass_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh() # 输出 [-1, 1]，表示相对变化率
        )

    def forward(self, coords, expression, mass, t):
        # 确保 t 的形状正确
        if t.dim() == 1:
            t = t.unsqueeze(-1) # (B, 1)
        if mass.dim() == 1:
            mass = mass.unsqueeze(-1) # (B, 1)
            
        x = torch.cat([coords, expression, mass, t], dim=-1)
        h = self.state_encoder(x)
        
        v_spatial = self.spatial_head(h)
        v_expression = self.expression_head(h)
        
        # 质量变化率系数，假设范围在 [-5, 5] (指数级增长/衰减)
        mass_rate = self.mass_head(h) * 5.0 
        
        return v_spatial, v_expression, mass_rate

# ==========================================
# 4. 训练逻辑
# ==========================================

def sample_matched_pair(ot_matrix, batch_size=128):
    """
    根据 OT 矩阵的权重采样 (source, target) 索引对。
    """
    # 将矩阵展平为一维概率分布
    flat_prob = ot_matrix.flatten().astype(np.float64)
    flat_prob = np.nan_to_num(flat_prob, nan=0.0)
    flat_prob[flat_prob < 0] = 0.0
    
    total = flat_prob.sum()
    if total <= 0:
        raise ValueError("OT matrix has no positive mass; cannot sample matched pairs.")
    
    flat_prob = flat_prob / total
    
    # 采样索引
    indices = np.random.choice(len(flat_prob), size=batch_size, p=flat_prob)
    
    # 转换回 (i, j) 坐标
    row_indices = indices // ot_matrix.shape[1]
    col_indices = indices % ot_matrix.shape[1]
    
    # 获取对应的 OT 权重用于 Loss 加权
    weights = ot_matrix[row_indices, col_indices]
    weights = np.nan_to_num(weights, nan=0.0)
    weight_mean = weights.mean()
    if weight_mean <= 0:
        weights = np.ones_like(weights)
    else:
        weights = weights / weight_mean
    
    return row_indices, col_indices, weights

def train_model(
    model, 
    coords_0, expr_0, mass_0, 
    coords_1, expr_1, mass_1, 
    ot_matrix, 
    epochs=1000, batch_size=256, lr=1e-3
):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 计算 OT 边缘分布以估计每个细胞的“真实”增殖潜力
    # 如果 OT 矩阵行和 sum > 1/N，说明该细胞需要分裂才能满足传输
    # 这是一个简单的启发式：estimated_growth = row_sum / uniform_mass
    row_sums = np.nan_to_num(ot_matrix.sum(axis=1), nan=0.0)
    mean_row_sum = row_sums.mean()
    if mean_row_sum <= 0:
        target_mass_ratios = np.ones_like(row_sums)
    else:
        target_mass_ratios = np.clip(row_sums / mean_row_sum, 1e-6, None) # 相对增长率
    
    print("Starting training...")
    model.train()
    
    for epoch in range(epochs):
        # 1. 采样
        idx_0, idx_1, ot_weights = sample_matched_pair(ot_matrix, batch_size)
        
        # 准备数据 (转为 Tensor)
        c0 = torch.tensor(coords_0[idx_0], dtype=torch.float32)
        e0 = torch.tensor(expr_0[idx_0], dtype=torch.float32)
        m0 = torch.tensor(mass_0[idx_0], dtype=torch.float32)
        
        c1 = torch.tensor(coords_1[idx_1], dtype=torch.float32)
        e1 = torch.tensor(expr_1[idx_1], dtype=torch.float32)
        m1 = torch.tensor(mass_1[idx_1], dtype=torch.float32)
        
        weights = torch.tensor(ot_weights, dtype=torch.float32).unsqueeze(1)
        
        # 2. 时间插值 t ~ U[0, 1]
        t = torch.rand(batch_size, 1)
        
        # 3. 构建 Flow Matching 目标 (Conditional FM)
        # 线性插值路径
        c_t = (1 - t) * c0 + t * c1
        e_t = (1 - t) * e0 + t * e1
        
        # 质量插值：这里比较 trick。
        # 如果仅仅线性插值 m0 到 m1，且 m0=m1=1，网络学不到东西。
        # 我们引入 OT 估计的增长潜力。
        # 假设目标质量 m_target = m0 * target_mass_ratios[idx_0]
        # 或者简单使用 m_t 线性插值，但在 loss 里计算变化率
        
        # 这里使用简单的线性插值作为输入状态
        m_t = (1 - t) * m0 + t * m1
        
        # 4. 模型预测
        v_spatial_pred, v_expression_pred, mass_rate_pred = model(c_t, e_t, m_t, t)
        
        # 5. 计算 Ground Truth 速度 (Conditional Vector Field)
        v_spatial_true = c1 - c0
        v_expression_true = e1 - e0
        
        # 质量变化率真值：dm/dt = (m1 - m0) / 1.0 (假设时间跨度归一化为1)
        # 修正：使用对数变化率更稳定 d(log m)/dt
        # 实际上，如果 m0=1, m1=1，这里是0。
        # 我们使用 OT 导出的 target_mass_ratio 来指导
        growth_ratio = torch.tensor(target_mass_ratios[idx_0], dtype=torch.float32)
        # 目标变化率：log(growth_ratio)
        mass_rate_true = torch.log(growth_ratio + 1e-6)
        
        # 6. Loss 计算
        loss_spatial = torch.mean(weights * (v_spatial_pred - v_spatial_true)**2)
        loss_expr = torch.mean(weights * (v_expression_pred - v_expression_true)**2)
        loss_mass = torch.mean(weights * (mass_rate_pred.squeeze() - mass_rate_true)**2)
        
        loss = loss_spatial + loss_expr + loss_mass
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss={loss.item():.4f} (S={loss_spatial.item():.3f}, E={loss_expr.item():.3f}, M={loss_mass.item():.3f})")
            
    return model

# ==========================================
# 5. 采样与预测：支持增殖的 ODE Solver
# ==========================================

def predict_evolution(
    model, 
    initial_cells, # dict list or DF
    t_start=0.0, 
    t_end=1.0, 
    dt=0.05,
    threshold_proliferation=1.8, # 质量达到 1.8 倍分裂
    threshold_apoptosis=0.2      # 质量低于 0.2 倍凋亡
):
    """
    模拟细胞演化，显式处理质量驱动的分裂与凋亡。
    使用 Euler 方法步进 (比 odeint 更容易处理拓扑变化)。
    """
    model.eval()
    
    # 初始化状态
    current_cells = initial_cells.copy() # List of dicts: {'coords', 'expr', 'mass', 'id'}
    
    t = t_start
    steps = int((t_end - t_start) / dt)
    
    print(f"Simulating evolution from t={t_start} to {t_end} with {steps} steps...")
    
    with torch.no_grad():
        for step in range(steps):
            if len(current_cells) == 0:
                break
                
            # 1. 准备当前批次数据
            coords = torch.tensor(np.array([c['coords'] for c in current_cells]), dtype=torch.float32)
            expr = torch.tensor(np.array([c['expr'] for c in current_cells]), dtype=torch.float32)
            mass = torch.tensor(np.array([c['mass'] for c in current_cells]), dtype=torch.float32)
            t_tensor = torch.ones(len(current_cells)) * t
            
            # 2. 预测变化率
            v_s, v_e, m_rate = model(coords, expr, mass, t_tensor)
            
            # 3. 更新状态 (Euler Step)
            # 空间和表达：x_{t+1} = x_t + v * dt
            coords_new = coords + v_s * dt
            expr_new = expr + v_e * dt
            
            # 质量更新：dm/dt = rate * m => m_{t+1} = m_t * exp(rate * dt)
            # 这种对数更新保证质量非负
            mass_new = mass * torch.exp(m_rate.squeeze() * dt)
            
            # 4. 处理分裂与凋亡 (Updating the list)
            next_cells = []
            
            coords_np = coords_new.numpy()
            expr_np = expr_new.numpy()
            mass_np = mass_new.numpy()
            
            for i, cell_data in enumerate(current_cells):
                m = mass_np[i]
                c = coords_np[i]
                e = expr_np[i]
                pid = cell_data['parent_id'] if 'parent_id' in cell_data else cell_data['id']
                
                if m > threshold_proliferation:
                    # === 分裂 ===
                    # 分裂成2个 (简单二分裂)
                    # 添加微小噪声防止重叠
                    for child_idx in range(2):
                        noise_c = np.random.normal(0, 0.01, size=c.shape)
                        noise_e = np.random.normal(0, 0.01, size=e.shape)
                        next_cells.append({
                            'coords': c + noise_c,
                            'expr': e + noise_e,
                            'mass': m / 2.0, # 质量减半
                            'id': f"{cell_data['id']}_d{step}_{child_idx}",
                            'parent_id': pid
                        })
                elif m < threshold_apoptosis:
                    # === 凋亡 ===
                    # 也可以选择保留但标记为 dead，这里直接移除
                    pass 
                else:
                    # === 正常存活 ===
                    next_cells.append({
                        'coords': c,
                        'expr': e,
                        'mass': m,
                        'id': cell_data['id'],
                        'parent_id': pid
                    })
            
            current_cells = next_cells
            t += dt
            
            if step % 5 == 0:
                print(f"  Step {step}/{steps}: {len(current_cells)} cells (Mass mean: {np.mean([c['mass'] for c in current_cells]):.2f})")

    return current_cells

# ==========================================
# 6. 主程序
# ==========================================

def main(day0_path, day3_path, output_path, target_time=0.66):
    """
    Main pipeline.
    target_time: 预测的时间点 (0.0 = day0, 1.0 = day3)。例如 Day 2 约等于 2/3 = 0.66
    """
    
    # 1. 加载数据
    print("Loading datasets...")
    ad0 = sc.read_h5ad(day0_path)
    ad3 = sc.read_h5ad(day3_path)
    
    # 2. 预处理 (确保两个数据集使用相同的 PCA 空间)
    # 实际操作中应合并后做 PCA 再拆分，这里简化为分别处理或投影
    # 这里的逻辑是：合并 -> 预处理 -> 拆分
    ad_concat = ad.concat([ad0, ad3], label="batch", keys=["day0", "day3"])
    c_all, e_all, m_all, ad_processed = preprocess_data(ad_concat, n_pca=50)
    
    # 拆分回 numpy array
    n0 = ad0.n_obs
    coords_0, expr_0, mass_0 = c_all[:n0], e_all[:n0], m_all[:n0]
    coords_3, expr_3, mass_3 = c_all[n0:], e_all[n0:], m_all[n0:]
    
    print(f"Day 0: {len(coords_0)} cells, Day 3: {len(coords_3)} cells")
    
    # 3. 计算 Unbalanced OT
    ot_matrix = compute_unbalanced_ot(
        coords_0, expr_0, coords_3, expr_3, mass_0, mass_3, tau=0.8
    )
    
    # 4. 初始化并训练模型
    model = MassFlowMatching(coord_dim=coords_0.shape[1], expression_dim=expr_0.shape[1])
    model = train_model(
        model, 
        coords_0, expr_0, mass_0,
        coords_3, expr_3, mass_3,
        ot_matrix,
        epochs=2000 # 实际需更多
    )
    
    # 5. 预测 (Day 0 -> Target Time)
    # 构建初始状态列表
    initial_state = []
    obs_indices = ad0.obs_names
    for i in range(n0):
        initial_state.append({
            'coords': coords_0[i],
            'expr': expr_0[i],
            'mass': mass_0[i],
            'id': obs_indices[i]
        })
        
    predicted_cells = predict_evolution(
        model, 
        initial_state, 
        t_start=0.0, 
        t_end=target_time, 
        dt=0.05
    )
    
    # 6. 重构 AnnData 输出
    print("Reconstructing AnnData...")
    pred_coords = np.array([c['coords'] for c in predicted_cells])
    pred_expr_pca = np.array([c['expr'] for c in predicted_cells])
    pred_mass = np.array([c['mass'] for c in predicted_cells])
    pred_ids = [c['id'] for c in predicted_cells]
    
    # 创建新的 AnnData
    # 注意：这里只有 PCA 特征。如果需要原始基因表达，需要训练一个 Decoder 或使用 kNN 映射回原始空间。
    # 这里我们演示 kNN 映射回原始基因表达空间 (利用 Day 0 和 Day 3 的参考数据)
    
    # 简单起见，我们创建一个只有 PCA 和 coords 的 AnnData
    ad_pred = ad.AnnData(X=pred_expr_pca) # 把 PCA 放 X 里暂存，或者放 obsm
    ad_pred.obsm['X_pca'] = pred_expr_pca
    ad_pred.obsm['spatial'] = pred_coords
    ad_pred.obs['mass'] = pred_mass
    ad_pred.obs_names = pred_ids
    ad_pred.obs['predicted_time'] = target_time
    
    # (可选) 恢复基因名，如果你有 decoder，这里略过
    
    print(f"Saving prediction to {output_path}")
    ad_pred.write_h5ad(output_path)
    print("Done.")

if __name__ == "__main__":
    # 示例调用
    # 请将路径替换为实际文件路径
    # main("data/day0.h5ad", "data/day3.h5ad", "output/day2_predicted.h5ad", target_time=0.66)
    
    parser = argparse.ArgumentParser(description="Mass-Flow-Matching Trajectory Inference")
    parser.add_argument("--input_t0", type=str, required=True, help="Path to Day 0 h5ad")
    parser.add_argument("--input_t1", type=str, required=True, help="Path to Day 3 (End) h5ad")
    parser.add_argument("--output", type=str, required=True, help="Path to save predicted h5ad")
    parser.add_argument("--time", type=float, default=0.66, help="Target interpolation time (0.0-1.0)")
    
    args = parser.parse_args()
    
    main(args.input_t0, args.input_t1, args.output, args.time)