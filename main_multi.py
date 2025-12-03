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
# 1. 数据预处理与加载 (修改支持列表)
# ==========================================
def preprocess_multislice(adata_list, time_points, n_top_genes=2000, n_pca=50, use_spatial_split=False):
    """
    处理多个切片，统一进行 PCA 和标准化。
    adata_list: list of AnnData
    time_points: list of float (e.g. [2, 10, 15, 20]) 
    """
    print("Concatenating all datasets for global normalization...")
    
    # 给每个 adata 标记时间
    for adata, t in zip(adata_list, time_points):
        adata.obs['time_point'] = t
        
    # 拼接
    adata_concat = ad.concat(adata_list, label="batch_time", keys=[str(t) for t in time_points], index_unique="-")
    raw_annotations = adata_concat.obs['Annotation'].astype(str).fillna("Unknown").values
    if use_spatial_split:
        print("  [Preprocess] Applying spatial splitting to Annotations to preserve gaps...")
        if 'spatial' in adata_concat.obsm:
            coords_for_cluster = adata_concat.obsm['spatial']
        else:
            coords_for_cluster = adata_concat.obsm['X_pca'][:, :2]
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(coords_for_cluster)
        spatial_labels = kmeans.labels_.astype(str)
        final_annotations = np.char.add(raw_annotations, np.char.add("_Spatial", spatial_labels))
    else:
        final_annotations = raw_annotations
    lb = LabelBinarizer()
    one_hot_all = lb.fit_transform(final_annotations)
    print(f"  [Preprocess] Encoded {len(lb.classes_)} unique cell types (with spatial split).")
        
        
    
    # 基础预处理
    if 'log1p' not in adata_concat.uns:
        sc.pp.normalize_total(adata_concat, target_sum=1e4)
        sc.pp.log1p(adata_concat)
        
    if n_top_genes is not None:
        sc.pp.highly_variable_genes(adata_concat, n_top_genes=n_top_genes, subset=True)
    
    sc.tl.pca(adata_concat, n_comps=n_pca)
    
    # 提取全局特征
    coords_all = adata_concat.obsm['spatial'] if 'spatial' in adata_concat.obsm else adata_concat.obsm['X_pca'][:, :2]
    expr_all = adata_concat.obsm['X_pca']
    
    if 'mass' in adata_concat.obs:
        mass_all = adata_concat.obs['mass'].values
    else:
        mass_all = np.ones(adata_concat.n_obs)
        
    # 全局标准化 (Z-score)
    c_mean, c_std = np.mean(coords_all, axis=0), np.std(coords_all, axis=0) + 1e-6
    e_mean, e_std = np.mean(expr_all, axis=0), np.std(expr_all, axis=0) + 1e-6
    
    coords_norm = (coords_all - c_mean) / c_std
    expr_norm = (expr_all - e_mean) / e_std
    
    scalers = {'c_mean': c_mean, 'c_std': c_std, 'e_mean': e_mean, 'e_std': e_std}
    
    # 重新拆分回列表，保持顺序
    processed_data = []
    start_idx = 0
    for adata in adata_list:
        n = adata.n_obs
        processed_data.append({
            'coords': coords_norm[start_idx : start_idx+n],
            'expr': expr_norm[start_idx : start_idx+n],
            'mass': mass_all[start_idx : start_idx+n],
            'obs_names': adata.obs_names,
            'type_onehot': one_hot_all[start_idx : start_idx+n],
            'n_cells': n
        })
        start_idx += n
        
    return processed_data, scalers, adata_concat

# ==========================================
# 2. OT 阶段 (无需大幅修改，只需调用多次)
# ==========================================
def inspect_ot_matrix(ot_matrix, name="Interval"):
    """
    分析 OT 矩阵的稀疏性、最大值和分布情况，防止训练时‘瞎猜’。
    """
    matrix = np.array(ot_matrix)
    n_rows, n_cols = matrix.shape
    total_transport = matrix.sum()
    
    # 稀疏度
    n_zeros = np.sum(matrix < 1e-8)
    sparsity = n_zeros / (n_rows * n_cols)
    
    # 行列统计
    row_sums = matrix.sum(axis=1) # 每个起始细胞“存活并转移”的量
    col_sums = matrix.sum(axis=0) # 每个终点细胞接收的量
    
    # 有效连接统计 (假设阈值 1e-6)
    valid_mask = matrix > 1e-6
    # 每个起始细胞平均连接到多少个终点细胞（扩散程度）
    avg_connectivity = np.sum(valid_mask, axis=1).mean() 
    
    print(f"\n[OT Matrix Analysis] {name}")
    print(f"  Shape: {n_rows} x {n_cols}")
    print(f"  Total Mass Transported: {total_transport:.4f}")
    print(f"  Sparsity (values < 1e-8): {sparsity:.2%}")
    print(f"  Max Probability: {matrix.max():.6f}")
    print(f"  Avg targets per source cell: {avg_connectivity:.2f} (lower is sharper/better)")
    print(f"  Dead cells (Row sum < 1e-6): {np.sum(row_sums < 1e-6)} / {n_rows}")
    print(f"--------------------------------------------------")
    
def _pairwise_squared_distances(x, y):
    x_sq = jnp.sum(jnp.square(x), axis=1, keepdims=True)
    y_sq = jnp.sum(jnp.square(y), axis=1)
    distances = x_sq + y_sq - 2.0 * jnp.matmul(x, y.T)
    return jnp.maximum(distances, 0.0)

def compute_unbalanced_ot(coords_0, expr_0, coords_1, expr_1, mass_0, mass_1,type_0_one_hot,type_1_one_hot, tau=0.8, epsilon=0.05,lambda_type=1.0):
    # 建议将 epsilon 设得更小 (如 0.05 或 0.01)，以获得更“尖锐”的映射
    print("  Calculating OT matrix (Sinkhorn)...")
    coords_x = jnp.array(coords_0)
    coords_y = jnp.array(coords_1)
    expr_x = jnp.array(expr_0)
    expr_y = jnp.array(expr_1)
    type_x = jnp.array(type_0_one_hot)
    type_y = jnp.array(type_1_one_hot)
    
    # 距离计算
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
    
    
    a = jnp.array(mass_0) / np.sum(mass_0)
    b = jnp.array(mass_1) / np.sum(mass_1)
    
    # 注意：epsilon 越小，映射越确定，但计算越容易数值不稳定
    geom = geometry.Geometry(cost_matrix=cost, epsilon=epsilon)
    prob = linear_problem.LinearProblem(geom, a=a, b=b, tau_a=tau, tau_b=tau)
    
    solver = sinkhorn.Sinkhorn()
    out = solver(prob)
    ot_matrix = np.array(out.matrix)
    
    # 计算完立刻打印分析
    inspect_ot_matrix(ot_matrix)
    
    return ot_matrix

# ==========================================
# 3. 模型定义 (输入维度不变)
# ==========================================

class MassFlowMatching(nn.Module):
    def __init__(self, coord_dim=2, expression_dim=50, hidden_dim=256):
        super().__init__()
        # Input: coords + expr + mass + absolute_time(1)
        input_dim = coord_dim + expression_dim + 1 + 1 
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), # 加个 Norm 稍微稳一点
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.spatial_head = nn.Linear(hidden_dim, coord_dim)
        self.expression_head = nn.Linear(hidden_dim, expression_dim)
        self.mass_head = nn.Linear(hidden_dim, 1)

    def forward(self, coords, expression, mass, t):
        # t 应该是绝对时间或者全局归一化时间
        if t.dim() == 1: t = t.unsqueeze(-1)
        if mass.dim() == 1: mass = mass.unsqueeze(-1)
            
        x = torch.cat([coords, expression, mass, t], dim=-1)
        h = self.net(x)
        
        return self.spatial_head(h), self.expression_head(h), self.mass_head(h)

# ==========================================
# 4. 多切片训练逻辑 (核心修改)
# ==========================================

class OTSampler:
    """
    专门的采样器类，预先处理好概率分布，避免在训练循环中重复计算。
    """
    def __init__(self, ot_matrix):
        self.ot_matrix = np.array(ot_matrix, dtype=np.float32)
        
        # 1. 计算行和 (每个 t0 细胞的总转移质量)
        self.row_sums = self.ot_matrix.sum(axis=1) + 1e-10
        
        # 2. 归一化每一行，得到条件概率 P(t1 | t0)
        # 这样每一行的和都为 1 (除非原本全为0)
        self.conditional_probs = self.ot_matrix / self.row_sums[:, None]
        
        # 处理全 0 行 (即该细胞完全凋亡，不流向任何地方)
        # 我们给它设一个均匀分布，但标记权重为 0，这样 Loss 乘权重后不会影响模型
        zero_rows = (self.row_sums < 1e-8)
        self.conditional_probs[zero_rows] = 1.0 / self.ot_matrix.shape[1]
        
        # 转为 PyTorch Tensor 方便后续采样
        self.conditional_probs_tensor = torch.tensor(self.conditional_probs)
        self.row_weights = torch.tensor(self.row_sums) # 这将作为训练时的 Loss 权重
        
        self.n_source = self.ot_matrix.shape[0]

    def sample(self, batch_size):
        # 步骤 A: 随机均匀选择 t0 的细胞索引
        # 这样保证稀有细胞和常见细胞被选中的概率一样
        row_indices = torch.randint(0, self.n_source, (batch_size,))
        
        # 步骤 B: 根据选中行的概率分布，采样 t1 的细胞索引
        # torch.multinomial 作用于每一行
        selected_probs = self.conditional_probs_tensor[row_indices]
        col_indices = torch.multinomial(selected_probs, num_samples=1).squeeze()
        
        # 步骤 C: 获取权重 (原本 OT 矩阵中的行和)
        # 如果某行原本几乎为0 (死细胞)，权重就很小，Loss 就不计入
        weights = self.row_weights[row_indices]
        
        return row_indices.numpy(), col_indices.numpy(), weights

def train_multislice(
    model, 
    data_list,     
    time_points,   
    ot_matrices,   
    epochs=2000, 
    batch_size=256, 
    lr=1e-3
):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # === 修改点：预先初始化采样器 ===
    samplers = [OTSampler(ot) for ot in ot_matrices]
    n_intervals = len(ot_matrices)

    print(f"Starting Multi-slice training on {time_points}...")
    model.train()
    
    for epoch in range(epochs):
        # 1. 随机选择时间区间
        interval_idx = np.random.randint(0, n_intervals)
        sampler = samplers[interval_idx]
        
        t_start = time_points[interval_idx]
        t_end = time_points[interval_idx+1]
        dt = t_end - t_start
        n_cells_0 = data_list[interval_idx]['n_cells']
        n_cells_1 = data_list[interval_idx+1]['n_cells']
        
        # 2. === 修改点：使用新的采样逻辑 ===
        # idx0 是均匀选的，weights 反映了该细胞是否“存活”
        idx0, idx1, weights = sampler.sample(batch_size)
        weights_tensor = torch.tensor(weights, dtype=torch.float32).to(model.spatial_head.weight.device).unsqueeze(1)
        
        d0 = data_list[interval_idx]
        d1 = data_list[interval_idx+1]
        
        c0 = torch.tensor(d0['coords'][idx0], dtype=torch.float32)
        e0 = torch.tensor(d0['expr'][idx0], dtype=torch.float32)
        m0 = torch.tensor(d0['mass'][idx0], dtype=torch.float32).unsqueeze(1)
        
        c1 = torch.tensor(d1['coords'][idx1], dtype=torch.float32)
        e1 = torch.tensor(d1['expr'][idx1], dtype=torch.float32)
        m1 = torch.tensor(d1['mass'][idx1], dtype=torch.float32).unsqueeze(1)
        
        # 3. 构造插值
        alpha = torch.rand(batch_size, 1)
        t_current = t_start + alpha * dt
        
        c_t = (1 - alpha) * c0 + alpha * c1
        e_t = (1 - alpha) * e0 + alpha * e1
        m_t = (1 - alpha) * m0 + alpha * m1
        
        # 4. 计算 Target
        v_spatial_target = (c1 - c0) / dt
        v_expr_target = (e1 - e0) / dt
        
        raw_growth = weights_tensor * n_cells_1
        global_avg_growth=n_cells_1/n_cells_0
        relative_growth=raw_growth/global_avg_growth
        sharpening_factor = 2.0  # 调节这个参数！越大差异越明显
        sharpened_relative = torch.pow(relative_growth, sharpening_factor)
        target_growth_factor = sharpened_relative * global_avg_growth
        target_growth_factor = torch.clamp(target_growth_factor, min=1e-4, max=10.0)
        k_target = torch.log(target_growth_factor) / dt
        
        # 5. 前向传播
        v_s_pred, v_e_pred, k_pred = model(c_t, e_t, m_t, t_current)
        
        # 6. === 修改点：Loss 加权 ===
        # 我们希望模型关注那些在 OT 中确实发生了转移的细胞
        # weights 来源于 OT 矩阵的行和。如果某细胞在 OT 里完全消失(weights~0)，这组数据对训练无意义
        
        loss_s = torch.mean(weights_tensor * (v_s_pred - v_spatial_target)**2)
        loss_e = torch.mean(weights_tensor * (v_e_pred - v_expr_target)**2)
        loss_m = torch.mean(weights_tensor * (k_pred - k_target)**2)
        
        # 归一化权重 (防止 Loss 随 OT 总质量波动)
        weight_sum = weights.mean() + 1e-8
        lambda_s = 1.0
        lambda_e = 1.0
        lambda_m = 1.0
        loss = (lambda_s * loss_s + lambda_e * loss_e + lambda_m * loss_m) / weight_sum
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch} | Intv {t_start}->{t_end} | Loss: {loss.item():.10f} | Loss_s: {loss_s.item():.10f} | Loss_e: {loss_e.item():.10f} | Loss_m: {loss_m.item():.10f}")
            
    return model

# ==========================================
# 5. 预测逻辑 (支持从最近的时间点出发)
# ==========================================

def predict_multislice(model, data_list, time_points, target_time, scalers, output_path):
    """
    根据目标时间，自动选择最近的前序切片作为起点进行模拟。
    """
    # 1. 确定起点
    # 找到所有小于 target_time 的时间点
    past_times = [t for t in time_points if t < target_time]
    if not past_times:
        raise ValueError(f"Target time {target_time} is before the first observed time point {time_points[0]}")
    
    start_time = past_times[-1] # 最近的过去时间点
    start_idx = time_points.index(start_time)
    
    print(f"\nPrediction Task: {target_time}")
    print(f"-> Starting simulation from observed slice: Day {start_time} (Index {start_idx})")
    print(f"-> Simulation duration: {target_time - start_time} days")
    
    # 2. 准备初始细胞
    start_data = data_list[start_idx]
    current_cells = []
    for i in range(start_data['n_cells']):
        current_cells.append({
            'coords': start_data['coords'][i],
            'expr': start_data['expr'][i],
            'mass': start_data['mass'][i],
            'id': start_data['obs_names'][i]
        })
        
    # 3. 欧拉积分 (Euler Integration)
    dt = 0.05 # 模拟步长 (单位：绝对时间天数，可以稍微调小一点以提高精度)
    curr_t = float(start_time)
    
    model.eval()
    
    while curr_t < target_time:
        # 如果剩下时间小于 dt，就只走剩下的一点点
        step_dt = min(dt, target_time - curr_t)
        
        # 准备 Batch
        coords = torch.tensor(np.array([c['coords'] for c in current_cells]), dtype=torch.float32)
        expr = torch.tensor(np.array([c['expr'] for c in current_cells]), dtype=torch.float32)
        mass = torch.tensor(np.array([c['mass'] for c in current_cells]), dtype=torch.float32)
        t_tensor = torch.ones(len(current_cells), 1) * curr_t
        
        with torch.no_grad():
            v_s, v_e, k = model(coords, expr, mass, t_tensor)
            
        # 更新状态
        coords_new = coords + v_s * step_dt
        expr_new = expr + v_e * step_dt
        # mass_new = mass * exp(k * dt)
        mass_new = mass * torch.exp(k.squeeze() * step_dt)
        
        # 更新列表
        coords_np = coords_new.numpy()
        expr_np = expr_new.numpy()
        mass_np = mass_new.numpy()
        
        # 简单的分裂/凋亡逻辑 (可选，简化版)
        next_cells = []
        for i, old_c in enumerate(current_cells):
            # 更新数值
            new_c = old_c.copy()
            new_c['coords'] = coords_np[i]
            new_c['expr'] = expr_np[i]
            new_c['mass'] = mass_np[i]
            
            # 简单的分裂判定 (阈值需根据实际mass分布调整)
            if new_c['mass'] > 1.9: # 假设初始mass约为1
                # 分裂成两个
                child1 = new_c.copy()
                child1['mass'] /= 2
                child1['id'] = new_c['id'] + "_c1"
                
                child2 = new_c.copy()
                child2['mass'] /= 2
                child2['id'] = new_c['id'] + "_c2"
                
                next_cells.extend([child1, child2])
            elif new_c['mass'] < 0.1:
                # 凋亡
                continue
            else:
                next_cells.append(new_c)
        
        current_cells = next_cells
        curr_t += step_dt
        # print(f"  t={curr_t:.2f}, count={len(current_cells)}")

    # 4. 重构并保存
    print(f"Simulation finished. Reconstructing AnnData with {len(current_cells)} cells...")
    
    if len(current_cells) == 0:
        print("Error: All cells died.")
        return

    pred_expr_norm = np.array([c['expr'] for c in current_cells])
    pred_coords_norm = np.array([c['coords'] for c in current_cells])
    pred_mass = np.array([c['mass'] for c in current_cells])
    pred_ids = [c['id'] for c in current_cells]
    
    # 反归一化
    pred_expr_pca = pred_expr_norm * scalers['e_std'] + scalers['e_mean']
    pred_coords = pred_coords_norm * scalers['c_std'] + scalers['c_mean']
    
    # 这里我们只保存 PCA 结果，如果需要基因表达矩阵，可以用原始数据的 PCA loading 反推
    # 为了简化，这里创建一个新的 AnnData
    ad_pred = ad.AnnData(X=pred_expr_pca) # 暂时把PCA放X里，或者放obsm
    ad_pred.obsm['X_pca'] = pred_expr_pca
    ad_pred.obsm['spatial'] = pred_coords
    ad_pred.obs['mass'] = pred_mass
    ad_pred.obs_names = pred_ids
    ad_pred.obs['predicted_time'] = target_time
    
    print(f"Saving to {output_path}")
    ad_pred.write_h5ad(output_path)

# ==========================================
# 6. 主程序
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Multi-Slice Mass Flow Matching")
    
    # 接收多个文件和时间点
    parser.add_argument("--files", nargs='+', required=True, help="List of h5ad files (e.g. day2.h5ad day10.h5ad ...)")
    parser.add_argument("--times", nargs='+', type=float, required=True, help="List of time points corresponding to files (e.g. 2 10 15 20)")
    parser.add_argument("--predict_time", type=float, required=True, help="Target time to predict (e.g. 5)")
    parser.add_argument("--output", type=str, required=True, help="Output path")
    parser.add_argument("--epochs", type=int, default=3000)
    
    args = parser.parse_args()
    
    # 1. 整理输入
    files = args.files
    times = args.times
    if len(files) != len(times):
        raise ValueError("Number of files must match number of time points.")
    
    # 按时间排序
    sorted_pairs = sorted(zip(times, files))
    sorted_times = [p[0] for p in sorted_pairs]
    sorted_files = [p[1] for p in sorted_pairs]
    
    print(f"Sorted Input Sequence: {sorted_times}")
    
    # 2. 加载与预处理
    adata_list = [sc.read_h5ad(f) for f in sorted_files]
    data_list, scalers, adata_concat = preprocess_multislice(adata_list, sorted_times)
    
    # 3. 计算序列 OT (2->10, 10->15, 15->20)
    ot_matrices = []
    for i in range(len(data_list) - 1):
        print(f"\nComputing OT for interval {sorted_times[i]} -> {sorted_times[i+1]} ...")
        d0 = data_list[i]
        d1 = data_list[i+1]
        ot = compute_unbalanced_ot(
            d0['coords'], d0['expr'], 
            d1['coords'], d1['expr'], 
            d0['mass'], d1['mass'],
            type_0_one_hot=d0['type_onehot'],
            type_1_one_hot=d1['type_onehot']
        )
        ot_matrices.append(ot)
        
    # 4. 训练全局模型
    model = MassFlowMatching(coord_dim=2, expression_dim=50) # 假设PCA=50
    model = train_multislice(model, data_list, sorted_times, ot_matrices, epochs=args.epochs)
    
    # 5. 预测 Day 5
    predict_multislice(model, data_list, sorted_times, args.predict_time, scalers, args.output)

if __name__ == "__main__":
    main()