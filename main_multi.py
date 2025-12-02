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
from ott.geometry import geometry
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. 数据预处理与加载
# ==========================================

def preprocess_data(adata, isLog=False, n_top_genes=2000, n_pca=50, basis='spatial'):
    """
    标准化、对数化、PCA降维，并提取坐标和特征。
    """
    print(f"Preprocessing AnnData with {adata.n_obs} cells...")
    adata = adata.copy()
    
    # 基础预处理
    if isLog:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    if n_top_genes is not None:    
        # 高变基因选择
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, subset=True)
    
    # PCA 降维
    sc.tl.pca(adata, n_comps=n_pca)
    
    # 提取特征
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
        mass = np.ones(adata.n_obs)
        
    return coords, expression, mass, adata


def inverse_pca_transform(pca_scores, reference_adata):
    """
    将 PCA 得分映射回基因表达空间。
    """
    if 'pca' not in reference_adata.uns:
        raise ValueError("reference_adata missing PCA information in .uns['pca']")
    if 'PCs' not in reference_adata.varm:
        raise ValueError("reference_adata missing loadings in .varm['PCs']")
    
    pca_scores = np.asarray(pca_scores)
    if pca_scores.ndim == 1:
        pca_scores = pca_scores.reshape(1, -1)
    
    loadings = np.asarray(reference_adata.varm['PCs'])
    n_components = pca_scores.shape[1]
    if loadings.shape[1] < n_components:
        raise ValueError(f"PCA loadings columns < provided scores")
    loadings = loadings[:, :n_components]
    
    mean = reference_adata.uns['pca'].get('mean')
    if mean is not None:
        mean = np.asarray(mean)
    
    expression = np.dot(pca_scores, loadings.T)
    if mean is not None:
        expression += mean
    
    return expression

# ==========================================
# 2. OT 阶段：计算 Unbalanced OT
# ==========================================

def _pairwise_squared_distances(x, y):
    x_sq = jnp.sum(jnp.square(x), axis=1, keepdims=True)
    y_sq = jnp.sum(jnp.square(y), axis=1)
    distances = x_sq + y_sq - 2.0 * jnp.matmul(x, y.T)
    return jnp.maximum(distances, 0.0)


def compute_unbalanced_ot(coords_0, expr_0, coords_1, expr_1, mass_0, mass_1, tau=0.5, epsilon=0.5, coord_weight=1.0, expr_weight=1.0):
    print("Computing Unbalanced Optimal Transport...")
    
    coords_0 = np.asarray(coords_0, dtype=np.float32)
    coords_1 = np.asarray(coords_1, dtype=np.float32)
    expr_0 = np.asarray(expr_0, dtype=np.float32)
    expr_1 = np.asarray(expr_1, dtype=np.float32)
    
    def _normalize_pair(arr0, arr1, eps=1e-8):
        stacked = np.vstack([arr0, arr1])
        mean = np.mean(stacked, axis=0, keepdims=True)
        std = np.std(stacked, axis=0, keepdims=True)
        std = np.where(std < eps, 1.0, std)
        return (arr0 - mean) / std, (arr1 - mean) / std
    
    coords_0, coords_1 = _normalize_pair(coords_0, coords_1)
    expr_0, expr_1 = _normalize_pair(expr_0, expr_1)
    
    coords_x = jnp.array(coords_0)
    coords_y = jnp.array(coords_1)
    expr_x = jnp.array(expr_0)
    expr_y = jnp.array(expr_1)
    a = jnp.array(mass_0) / np.sum(mass_0)
    b = jnp.array(mass_1) / np.sum(mass_1)
    
    coord_cost = coord_weight * _pairwise_squared_distances(coords_x, coords_y)
    expr_cost = expr_weight * _pairwise_squared_distances(expr_x, expr_y)
    combined_cost = coord_cost + expr_cost
    
    geom = geometry.Geometry(cost_matrix=combined_cost, epsilon=epsilon)
    prob = linear_problem.LinearProblem(geom, a=a, b=b, tau_a=tau, tau_b=tau)
    
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
    
    return transport_matrix

# ==========================================
# 3. 模型定义：Mass Flow Matching
# ==========================================

class MassFlowMatching(nn.Module):
    def __init__(self, coord_dim=2, expression_dim=50, hidden_dim=256):
        super().__init__()
        
        input_dim = coord_dim + expression_dim + 1 + 1 
        
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
        
        self.mass_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh() 
        )

    def forward(self, coords, expression, mass, t):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        if mass.dim() == 1:
            mass = mass.unsqueeze(-1)
            
        x = torch.cat([coords, expression, mass, t], dim=-1)
        h = self.state_encoder(x)
        
        v_spatial = self.spatial_head(h)
        v_expression = self.expression_head(h)
        mass_rate = self.mass_head(h) * 5.0 
        
        return v_spatial, v_expression, mass_rate

# ==========================================
# 4. 训练逻辑
# ==========================================

def sample_matched_pair(ot_matrix, batch_size=128):
    flat_prob = ot_matrix.flatten().astype(np.float64)
    flat_prob = np.nan_to_num(flat_prob, nan=0.0)
    flat_prob[flat_prob < 0] = 0.0
    
    total = flat_prob.sum()
    if total <= 0:
        raise ValueError("OT matrix has no positive mass")
    
    flat_prob = flat_prob / total
    indices = np.random.choice(len(flat_prob), size=batch_size, p=flat_prob)
    
    row_indices = indices // ot_matrix.shape[1]
    col_indices = indices % ot_matrix.shape[1]
    
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
    epochs=500, batch_size=256, lr=1e-3,
    lambda_spatial=1.0, lambda_expr=0.0, lambda_mass=0.0
):
    coord_dim = coords_0.shape[1]
    theta = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
    trans = nn.Parameter(torch.zeros(coord_dim, dtype=torch.float32))

    optimizer = optim.Adam(list(model.parameters()) + [theta, trans], lr=lr)
    
    # === 计算真实的质量增长率 (修复负数问题) ===
    ot_row_sums = np.nan_to_num(ot_matrix.sum(axis=1), nan=0.0)
    total_mass_t0 = np.sum(mass_0)
    total_mass_t1 = np.sum(mass_1)
    
    global_scale = total_mass_t1 / total_mass_t0
    
    ot_sum = ot_matrix.sum()
    if ot_sum == 0: ot_sum = 1.0
    relative_contribution = ot_row_sums / ot_sum
    
    estimated_target_mass = relative_contribution * total_mass_t1
    mass_0_safe = np.maximum(mass_0, 1e-8)
    target_mass_ratios = estimated_target_mass / mass_0_safe
    target_mass_ratios = np.clip(target_mass_ratios, 1e-4, 50.0)
    
    # === 简化后的诊断信息 ===
    print("\n=== Growth Calculation Info ===")
    print(f"Global Expansion Factor (T1/T0 Mass): {global_scale:.4f}")
    print(f"Mean Target Ratio per Cell: {target_mass_ratios.mean():.4f}")
    print("===============================\n")
    
    print("Starting training...")
    model.train()
    
    for epoch in range(epochs):
        # 1. 采样
        idx_0, idx_1, ot_weights = sample_matched_pair(ot_matrix, batch_size)
        
        c0 = torch.tensor(coords_0[idx_0], dtype=torch.float32)
        e0 = torch.tensor(expr_0[idx_0], dtype=torch.float32)
        m0 = torch.tensor(mass_0[idx_0], dtype=torch.float32).unsqueeze(1)
        
        c1_raw = torch.tensor(coords_1[idx_1], dtype=torch.float32)
        e1 = torch.tensor(expr_1[idx_1], dtype=torch.float32)
        m1 = torch.tensor(mass_1[idx_1], dtype=torch.float32).unsqueeze(1)
        
        weights = torch.tensor(ot_weights, dtype=torch.float32).unsqueeze(1)
        
        t = torch.rand(batch_size, 1)
        
        # 2. 对齐与插值
        if coord_dim == 2:
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)
            R = torch.stack([torch.stack([cos_t, -sin_t]), torch.stack([sin_t,  cos_t])])
            c1 = (R @ c1_raw.T).T + trans
        else:
            c1 = c1_raw + trans

        c_t = (1 - t) * c0 + t * c1
        e_t = (1 - t) * e0 + t * e1
        m_t = (1 - t) * m0 + t * m1
        
        # 3. 预测与 Loss
        v_spatial_pred, v_expression_pred, mass_rate_pred = model(c_t, e_t, m_t, t)
        
        v_spatial_true = c1 - c0
        v_expression_true = e1 - e0
        
        growth_ratio = torch.tensor(target_mass_ratios[idx_0], dtype=torch.float32)
        mass_rate_true = torch.log(growth_ratio + 1e-6)
        
        loss_spatial = torch.mean(weights * (v_spatial_pred - v_spatial_true)**2)
        loss_expr = torch.mean(weights * (v_expression_pred - v_expression_true)**2)
        loss_mass = torch.mean(weights * (mass_rate_pred.squeeze() - mass_rate_true)**2)
        
        loss = lambda_spatial * loss_spatial + lambda_expr * loss_expr + lambda_mass * loss_mass
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # === 详细的损失函数打印 ===
        if epoch % 100 == 0:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{epochs}")
            print(f"{'='*60}")
            print(f"总损失 (Total Loss):        {loss.item():.6f}")
            print(f"  空间损失 (Spatial Loss):  {loss_spatial.item():.6f} (λ={lambda_spatial:.2f}, weighted={lambda_spatial * loss_spatial.item():.6f})")
            print(f"  表达损失 (Expression):    {loss_expr.item():.6f} (λ={lambda_expr:.2f}, weighted={lambda_expr * loss_expr.item():.6f})")
            print(f"  质量损失 (Mass Loss):      {loss_mass.item():.6f} (λ={lambda_mass:.2f}, weighted={lambda_mass * loss_mass.item():.6f})")
            print(f"\n预测值统计:")
            print(f"  空间速度预测均值:         {v_spatial_pred.mean().item():.6f} (std: {v_spatial_pred.std().item():.6f})")
            print(f"  表达速度预测均值:         {v_expression_pred.mean().item():.6f} (std: {v_expression_pred.std().item():.6f})")
            print(f"  质量增长率预测均值:       {mass_rate_pred.mean().item():.6f} (std: {mass_rate_pred.std().item():.6f})")
            print(f"  质量增长率真实均值:       {mass_rate_true.mean().item():.6f} (std: {mass_rate_true.std().item():.6f})")
            print(f"{'='*60}\n")
            
    return model



# ==========================================
# 5. 采样与预测
# ==========================================

def predict_evolution(
    model, 
    initial_cells,
    t_start=0.0, 
    t_end=1.0, 
    dt=0.05,
    threshold_proliferation=1.8,
    threshold_apoptosis=0.2
):
    model.eval()
    current_cells = initial_cells.copy()
    
    t = t_start
    steps = int((t_end - t_start) / dt)
    
    print(f"Simulating evolution: t={t_start}->{t_end}, {len(current_cells)} start cells")
    
    # 统计数据容器 (仅用于最后的总结)
    stats_history = {'n_cells': [], 'total_mass': []}
    
    with torch.no_grad():
        for step in range(steps):
            if len(current_cells) == 0:
                print(f"  Step {step}: All died.")
                break
                
            coords = torch.tensor(np.array([c['coords'] for c in current_cells]), dtype=torch.float32)
            expr = torch.tensor(np.array([c['expr'] for c in current_cells]), dtype=torch.float32)
            mass = torch.tensor(np.array([c['mass'] for c in current_cells]), dtype=torch.float32)
            t_tensor = torch.ones(len(current_cells)) * t
            
            v_s, v_e, m_rate = model(coords, expr, mass, t_tensor)
            
            coords_new = coords + v_s * dt
            expr_new = expr + v_e * dt
            mass_new = mass * torch.exp(m_rate.squeeze() * dt)
            
            next_cells = []
            
            coords_np = coords_new.numpy()
            expr_np = expr_new.numpy()
            mass_np = mass_new.numpy()
            
            for i, cell_data in enumerate(current_cells):
                m = mass_np[i]
                c = coords_np[i]
                e = expr_np[i]
                pid = cell_data.get('parent_id', cell_data['id'])
                
                if m > threshold_proliferation:
                    # 分裂
                    for child_idx in range(2):
                        noise_c = np.random.normal(0, 0.01, size=c.shape)
                        noise_e = np.random.normal(0, 0.01, size=e.shape)
                        next_cells.append({
                            'coords': c + noise_c,
                            'expr': e + noise_e,
                            'mass': m / 2.0,
                            'id': f"{cell_data['id']}_d{step}_{child_idx}",
                            'parent_id': pid
                        })
                elif m < threshold_apoptosis:
                    # 凋亡 (跳过添加)
                    pass 
                else:
                    # 存活
                    next_cells.append({
                        'coords': c, 'expr': e, 'mass': m,
                        'id': cell_data['id'], 'parent_id': pid
                    })
            
            current_cells = next_cells
            t += dt
            
            # 记录简要历史
            curr_mass_sum = sum(c['mass'] for c in current_cells)
            stats_history['n_cells'].append(len(current_cells))
            stats_history['total_mass'].append(curr_mass_sum)
            
            # === 简化后的循环打印 (降低频率) ===
            if step % 10 == 0 or step == steps - 1:
                print(f"  Step {step}/{steps}: Cells={len(current_cells)}, Total Mass={curr_mass_sum:.2f}")

    print(f"Done. Final Cells: {len(current_cells)}, Growth Factor: {stats_history['total_mass'][-1]/stats_history['total_mass'][0]:.2f}x\n")
    return current_cells




def main(input_t0_path, input_t1_path, output_path, target_time=0.66):
    """
    原有的两个时间点插值函数（保持向后兼容）。
    """
    print("加载数据集 (t0, t1)...")
    ad_t0 = sc.read_h5ad(input_t0_path)
    ad_t1 = sc.read_h5ad(input_t1_path)
    
    # 预处理
    ad_concat = ad.concat([ad_t0, ad_t1], label="batch", keys=["t0", "t1"])
    c_all, e_all, m_all, ad_processed = preprocess_data(ad_concat, n_pca=50, n_top_genes=None)
    
    # === 标准化 (关键修复) ===
    e_mean, e_std = np.mean(e_all, axis=0), np.std(e_all, axis=0) + 1e-6
    e_all_norm = (e_all - e_mean) / e_std
    
    c_mean, c_std = np.mean(c_all, axis=0), np.std(c_all, axis=0) + 1e-6
    c_all_norm = (c_all - c_mean) / c_std
    
    scalers = {'e_mean': e_mean, 'e_std': e_std, 'c_mean': c_mean, 'c_std': c_std}
    
    n0 = ad_t0.n_obs
    coords_0, expr_0, mass_0 = c_all_norm[:n0], e_all_norm[:n0], m_all[:n0]
    coords_1, expr_1, mass_1 = c_all_norm[n0:], e_all_norm[n0:], m_all[n0:]
    
    print(f"t0: {len(coords_0)} cells, t1: {len(coords_1)} cells")
    
    # 计算 OT
    ot_matrix = compute_unbalanced_ot(coords_0, expr_0, coords_1, expr_1, mass_0, mass_1, tau=0.8)
    
    # 训练
    model = MassFlowMatching(coord_dim=coords_0.shape[1], expression_dim=expr_0.shape[1])
    model = train_model(model, coords_0, expr_0, mass_0, coords_1, expr_1, mass_1, ot_matrix, epochs=2000)
    
    # 预测
    initial_state = []
    obs_indices = ad_t0.obs_names
    for i in range(n0):
        initial_state.append({
            'coords': coords_0[i], 'expr': expr_0[i], 'mass': mass_0[i], 'id': obs_indices[i]
        })
        
    predicted_cells = predict_evolution(model, initial_state, t_start=0.0, t_end=target_time, dt=0.05)
    
    # 重构输出 (反标准化)
    print("Reconstructing AnnData...")
    pred_expr_norm = np.array([c['expr'] for c in predicted_cells])
    pred_coords_norm = np.array([c['coords'] for c in predicted_cells])
    
    pred_expr_pca = pred_expr_norm * scalers['e_std'] + scalers['e_mean']
    pred_coords = pred_coords_norm * scalers['c_std'] + scalers['c_mean']
    pred_mass = np.array([c['mass'] for c in predicted_cells])
    pred_ids = [c['id'] for c in predicted_cells]
    
    pred_expr = inverse_pca_transform(pred_expr_pca, ad_processed)
    
    ad_pred = ad.AnnData(X=pred_expr, var=ad_processed.var.copy())
    ad_pred.var_names = ad_processed.var_names.copy()
    ad_pred.obsm['X_pca'] = pred_expr_pca
    ad_pred.obsm['spatial'] = pred_coords
    ad_pred.obs['mass'] = pred_mass
    ad_pred.obs_names = pred_ids
    ad_pred.obs['predicted_time'] = target_time
    
    print(f"Saving prediction to {output_path}")
    ad_pred.write_h5ad(output_path)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mass-Flow-Matching Trajectory Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # 两种模式：原有的两个时间点模式，或新的多时间点模式
    parser.add_argument("--input_t0", type=str, help="Path to start-time (t0) h5ad (两时间点模式)")
    parser.add_argument("--input_t1", type=str, help="Path to end-time (t1) h5ad (两时间点模式)")
    parser.add_argument("--output", type=str, required=True, help="Path to save predicted h5ad")
    parser.add_argument("--time", type=float, default=0.66, help="Target interpolation time (0.0-1.0, 仅用于两时间点模式)")
    parser.add_argument("--epochs", type=int, default=2000, help="训练轮数")
    parser.add_argument("--use_all_segments", action="store_true", 
                       help="使用所有相邻时间段进行联合训练（可能提升预测效果，但需要更多计算）。默认只使用目标时间段。")
    
    args = parser.parse_args()
        
    main(args.input_t0, args.input_t1, args.output, args.time)