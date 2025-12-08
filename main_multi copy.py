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
# 1. 指标计算 (对应论文 Fig 2H)
# ==========================================
def calculate_benchmark_metrics(pred_adata, gt_adata, scalers, w=0.9):
    """
    计算预测结果(pred)与真实结果(gt)之间的 Wasserstein 距离。
    对应论文公式：
    Cost = (1-w)||q - q_hat||^2 + w||x - x_hat||^2
    """
    print("  [Metric] Calculating 2-Wasserstein metrics...")
    
    # 1. 获取数据并反归一化回原始/PCA空间，保证尺度有物理意义
    # (如果 scalers 是基于 Z-score 的，距离计算通常在归一化空间进行比较公平，
    # 但论文中使用了 w=0.9，暗示空间和基因需要在可比的量级。
    # 这里我们使用归一化后的数据直接计算，这样最稳健)
    
    # 预测数据
    pred_coords = pred_adata.obsm['spatial']
    pred_expr = pred_adata.obsm['X_pca']
    print(pred_coords.shape, pred_expr.shape)
    
    # 真实数据 (确保使用相同预处理后的归一化数据)
    gt_coords = gt_adata['coords'] # 这是预处理字典里的 numpy 数组
    gt_expr = gt_adata['expr']
    
    # 转为 JAX array
    p_c = jnp.array(pred_coords)
    p_e = jnp.array(pred_expr)
    g_c = jnp.array(gt_coords)
    g_e = jnp.array(gt_expr)
    
    # -------------------------------------------
    # 指标 A: Gene Expression Only (w=0)
    # -------------------------------------------
    dist_e = _pairwise_squared_distances(p_e, g_e)
    
    # 建立几何对象 (使用 Sinkhorn 求解 W2)
    # 注意：epsilon 越小越接近真实 Wasserstein，但计算越慢。验证时可以设小一点。
    geom_gene = geometry.Geometry(cost_matrix=dist_e, epsilon=0.01)
    
    # 假设均一分布 (mass balancing)
    # 如果是不平衡 OT，这里应该用 Unbalanced，但做 Benchmark 指标通常假设分布归一化
    a = jnp.ones(p_e.shape[0]) / p_e.shape[0]
    b = jnp.ones(g_e.shape[0]) / g_e.shape[0]
    
    prob_gene = linear_problem.LinearProblem(geom_gene, a=a, b=b)
    solver = sinkhorn.Sinkhorn()
    out_gene = solver(prob_gene)
    
    # OT Cost = <P, C>. Sinkhorn 输出的 regularized cost 包含熵项，
    # 我们通常需要 primal_cost 或者直接 sum(P * C)
    # ott 的 out.reg_ot_cost 是正则化的， out.primal_cost 是无正则的
    w2_gene = jnp.sqrt(out_gene.primal_cost)
    
    # -------------------------------------------
    # 指标 B: Combined (Gene + Spatial) (w=0.9)
    # -------------------------------------------
    dist_c = _pairwise_squared_distances(p_c, g_c)
    
    # 混合 Cost Matrix
    # 注意：论文中 w=0.9 是个强假设，前提是空间和基因的数值范围要合适。
    # 这里我们使用的是 Z-Score 归一化后的数据，数值都在 0 附近，方差为 1，直接加权是合理的。
    combined_cost = (1 - w) * dist_e + w * dist_c
    
    geom_comb = geometry.Geometry(cost_matrix=combined_cost, epsilon=0.01)
    prob_comb = linear_problem.LinearProblem(geom_comb, a=a, b=b)
    out_comb = solver(prob_comb)
    w2_combined = jnp.sqrt(out_comb.primal_cost)
    
    return float(w2_gene), float(w2_combined)

# ==========================================
# 2. 数据预处理与加载 (保持不变)
# ==========================================
def preprocess_multislice(adata_list, time_points, n_top_genes=2000, n_pca=50, use_spatial_split=False):
    """处理多个切片，统一进行 PCA 和标准化。"""
    print("Concatenating all datasets for global normalization...")
    
    for adata, t in zip(adata_list, time_points):
        adata.obs['time_point'] = t
        
    adata_concat = ad.concat(adata_list, label="batch_time", keys=[str(t) for t in time_points], index_unique="-")
    
    # 简单的 Annotations 处理
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
    
    sc.tl.pca(adata_concat, n_comps=n_pca)
    
    coords_all = adata_concat.obsm['spatial'] if 'spatial' in adata_concat.obsm else adata_concat.obsm['X_pca'][:, :2]
    expr_all = adata_concat.obsm['X_pca']
    mass_all = adata_concat.obs['mass'].values if 'mass' in adata_concat.obs else np.ones(adata_concat.n_obs)
        
    # 全局标准化 (Z-score)
    c_mean, c_std = np.mean(coords_all, axis=0), np.std(coords_all, axis=0) + 1e-6
    e_mean, e_std = np.mean(expr_all, axis=0), np.std(expr_all, axis=0) + 1e-6
    
    coords_norm = (coords_all - c_mean) / c_std
    expr_norm = (expr_all - e_mean) / e_std
    
    scalers = {'c_mean': c_mean, 'c_std': c_std, 'e_mean': e_mean, 'e_std': e_std}
    
    processed_data_dict = {} # 改成字典方便按时间存取
    start_idx = 0
    for t, adata in zip(time_points, adata_list):
        n = adata.n_obs
        processed_data_dict[t] = {
            'coords': coords_norm[start_idx : start_idx+n],
            'expr': expr_norm[start_idx : start_idx+n],
            'mass': mass_all[start_idx : start_idx+n],
            'obs_names': adata.obs_names,
            'type_onehot': one_hot_all[start_idx : start_idx+n],
            'n_cells': n
        }
        start_idx += n
        
    return processed_data_dict, scalers

# ==========================================
# 3. OT 计算辅助函数
# ==========================================
def _pairwise_squared_distances(x, y):
    x_sq = jnp.sum(jnp.square(x), axis=1, keepdims=True)
    y_sq = jnp.sum(jnp.square(y), axis=1)
    distances = x_sq + y_sq - 2.0 * jnp.matmul(x, y.T)
    return jnp.maximum(distances, 0.0)

def compute_unbalanced_ot(d0, d1, tau=0.8, epsilon=0.01, lambda_type=5.0):
    """封装 OT 计算"""
    coords_x, coords_y = jnp.array(d0['coords']), jnp.array(d1['coords'])
    expr_x, expr_y = jnp.array(d0['expr']), jnp.array(d1['expr'])
    type_x, type_y = jnp.array(d0['type_onehot']), jnp.array(d1['type_onehot'])
    
    dist_c = _pairwise_squared_distances(coords_x, coords_y)
    dist_e = _pairwise_squared_distances(expr_x, expr_y)
    
    # 类型约束 (可选，如果有Annotation)
    if d0['type_onehot'].shape[1] > 1:
        similarity = jnp.matmul(type_x, type_y.T)
        dist_type = 1.0 - similarity
        cost = (dist_c / (jnp.mean(dist_c)+1e-8)) + (dist_e / (jnp.mean(dist_e)+1e-8)) + lambda_type * dist_type
    else:
        cost = (dist_c / (jnp.mean(dist_c)+1e-8)) + (dist_e / (jnp.mean(dist_e)+1e-8))
    raw_cost = dist_c + dist_e
    scale_factor = jnp.mean(raw_cost) + 1e-8
    cost = raw_cost / scale_factor

    a = jnp.array(d0['mass']) / np.sum(d0['mass'])
    b = jnp.array(d1['mass']) / np.sum(d1['mass'])
    
    geom = geometry.Geometry(cost_matrix=cost, epsilon=epsilon)
    prob = linear_problem.LinearProblem(geom, a=a, b=b, tau_a=tau, tau_b=tau)
    solver = sinkhorn.Sinkhorn()
    out = solver(prob)
    return np.array(out.matrix)

# ==========================================
# 4. 模型与训练 (支持传入 subset list)
# ==========================================
class MassFlowMatching(nn.Module):
    def __init__(self, coord_dim=2, expression_dim=50, hidden_dim=256):
        super().__init__()
        input_dim = coord_dim + expression_dim + 1 + 1 
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.spatial_head = nn.Linear(hidden_dim, coord_dim)
        self.expression_head = nn.Linear(hidden_dim, expression_dim)
        self.mass_head = nn.Linear(hidden_dim, 1)

    def forward(self, coords, expression, mass, t):
        if t.dim() == 1: t = t.unsqueeze(-1)
        if mass.dim() == 1: mass = mass.unsqueeze(-1)
        x = torch.cat([coords, expression, mass, t], dim=-1)
        h = self.net(x)
        return self.spatial_head(h), self.expression_head(h), self.mass_head(h)

class OTSampler:
    def __init__(self, ot_matrix):
        self.ot_matrix = np.array(ot_matrix, dtype=np.float32)
        self.row_sums = self.ot_matrix.sum(axis=1) + 1e-10
        self.conditional_probs = self.ot_matrix / self.row_sums[:, None]
        zero_rows = (self.row_sums < 1e-8)
        self.conditional_probs[zero_rows] = 1.0 / self.ot_matrix.shape[1]
        self.conditional_probs_tensor = torch.tensor(self.conditional_probs)
        self.row_weights = torch.tensor(self.row_sums)
        self.n_source = self.ot_matrix.shape[0]

    def sample(self, batch_size):
        row_indices = torch.randint(0, self.n_source, (batch_size,))
        col_indices = torch.multinomial(self.conditional_probs_tensor[row_indices], num_samples=1).squeeze()
        weights = self.row_weights[row_indices]
        return row_indices.numpy(), col_indices.numpy(), weights

def train_model(train_data_list, train_times, epochs=1000, batch_size=256):
    """
    针对给定的数据列表训练模型 (Benchmark 专用)。
    包含：维度修复 + Loss_m 回归
    注意：时间会被归一化到 [0, 1] 区间，每个时间间隔映射为 [0, 1]
    """
    # 1. 计算 OT 矩阵链
    ot_matrices = []
    samplers = []
    
    # print(f"  Computing OT matrices for sequence: {train_times}")
    for i in range(len(train_data_list) - 1):
        d0 = train_data_list[i]
        d1 = train_data_list[i+1]
        ot = compute_unbalanced_ot(d0, d1)
        samplers.append(OTSampler(ot))
        ot_matrices.append(ot)
        
    model = MassFlowMatching()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    n_intervals = len(ot_matrices)
    
    model.train() # 确保在训练模式
    
    for epoch in range(epochs):
        # A. 采样
        interval_idx = np.random.randint(0, n_intervals)
        sampler = samplers[interval_idx]
        
        t_start = train_times[interval_idx]
        t_end = train_times[interval_idx+1]
        dt = t_end - t_start
        
        d0 = train_data_list[interval_idx]
        d1 = train_data_list[interval_idx+1]
        
        n_cells_0 = d0['n_cells']
        n_cells_1 = d1['n_cells']
        
        idx0, idx1, weights = sampler.sample(batch_size)
        weights_tensor = torch.tensor(weights, dtype=torch.float32).unsqueeze(1) # [B, 1]
        
        # B. 准备数据
        c0 = torch.tensor(d0['coords'][idx0], dtype=torch.float32)
        e0 = torch.tensor(d0['expr'][idx0], dtype=torch.float32)
        m0 = torch.tensor(d0['mass'][idx0], dtype=torch.float32).unsqueeze(1)
        
        c1 = torch.tensor(d1['coords'][idx1], dtype=torch.float32)
        e1 = torch.tensor(d1['expr'][idx1], dtype=torch.float32)
        # m1 = torch.tensor(d1['mass'][idx1], dtype=torch.float32).unsqueeze(1) 
        
        # C. 构造中间态
        # 关键修复：将时间归一化到 [0, 1] 区间
        # alpha 已经在 [0, 1] 范围内，直接作为归一化时间使用
        alpha = torch.rand(batch_size, 1)
        t_current = alpha  # 归一化时间：每个时间间隔内部使用 [0, 1]
        
        c_t = (1 - alpha) * c0 + alpha * c1
        e_t = (1 - alpha) * e0 + alpha * e1
        m_t = m0 # 简化处理：输入 mass 暂时用 t0 的
        
        # D. 计算 Target
        # D.1 速度 Target
        # 注意：由于时间已归一化到 [0, 1]，速度需要相应调整
        # 原始速度 = (c1 - c0) / dt，归一化后 dt_norm = 1.0
        v_s_target = (c1 - c0)  # 归一化时间下的速度
        v_e_target = (e1 - e0)
        
        # D.2 增殖率 Target (k_target)
        # 逻辑：weights 是转移概率和，反映了该细胞分裂了多少份流向 t1
        # relative_growth = 局部扩增倍数 / 全局平均扩增倍数
        global_avg_growth = n_cells_1 / n_cells_0
        raw_growth = weights_tensor * n_cells_1 
        relative_growth = raw_growth / (global_avg_growth + 1e-8)
        
        # Sharpening (可选，增强差异)
        sharpened_relative = torch.pow(relative_growth, 2.0) 
        target_growth_factor = sharpened_relative * global_avg_growth
        target_growth_factor = torch.clamp(target_growth_factor, min=1e-4, max=10.0)
        
        # k = ln(growth) / dt
        # 注意：由于时间归一化，dt_norm = 1.0，但 k 的物理意义需要保持
        # 所以 k_target 仍然除以原始 dt，以保持正确的增殖率单位
        k_target = torch.log(target_growth_factor) / dt
        
        # E. 前向传播
        # 注意这里接收 k_pred
        v_s_pred, v_e_pred, k_pred = model(c_t, e_t, m_t, t_current)
        
        # F. 计算 Loss
        # F.1 空间 Loss [Batch, 2] -> scalar
        loss_s = torch.mean(weights_tensor * (v_s_pred - v_s_target)**2)
        
        # F.2 基因 Loss [Batch, 50] -> scalar
        loss_e = torch.mean(weights_tensor * (v_e_pred - v_e_target)**2)
        
        # F.3 增殖 Loss [Batch, 1] -> scalar
        loss_m = torch.mean(weights_tensor * (k_pred - k_target)**2)
        
        # F.4 总 Loss (加权求和，防止权重总和波动)
        weight_sum = weights_tensor.mean() + 1e-8
        loss = (loss_s + loss_e + loss_m) / weight_sum
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return model
# ==========================================
# 5. 预测函数 (返回结果对象)
# ==========================================
def predict_state(model, start_data, start_time, target_time):
    """
    从 start_time 演化到 target_time (包含细胞分裂与凋亡逻辑)
    注意：时间会被归一化到 [0, 1] 区间进行预测
    """
    # 1. 初始化细胞列表
    # 将 numpy 数组转为 list of dicts，方便动态增删
    current_cells = []
    n_start = start_data['n_cells']
    for i in range(n_start):
        current_cells.append({
            'coords': start_data['coords'][i],
            'expr': start_data['expr'][i],
            'mass': start_data['mass'][i], # 初始 mass 通常归一化为 1
            'id': f"cell_{i}"
        })
    
    # 计算时间归一化参数
    total_dt = target_time - start_time
    dt_norm = 0.01  # 归一化时间步长（在 [0, 1] 区间内）
    curr_t_norm = 0.0  # 归一化时间，从 0 开始
    
    model.eval()
    
    print(f"    Simulating {start_time} -> {target_time} (normalized: 0.0 -> 1.0) ...")
    
    while curr_t_norm < 1.0:
        step_dt_norm = min(dt_norm, 1.0 - curr_t_norm)
        step_dt_actual = step_dt_norm * total_dt  # 转换为实际时间步长
        
        # A. 准备 Batch Tensor
        if len(current_cells) == 0:
            print("    [Warning] All cells died.")
            break
            
        # 将当前活着的细胞堆叠成 Tensor
        coords = torch.tensor(np.array([c['coords'] for c in current_cells]), dtype=torch.float32)
        expr = torch.tensor(np.array([c['expr'] for c in current_cells]), dtype=torch.float32)
        mass = torch.tensor(np.array([c['mass'] for c in current_cells]), dtype=torch.float32).unsqueeze(1)
        # 关键修复：使用归一化时间
        t_tensor = torch.ones(len(current_cells), 1) * curr_t_norm
        
        # B. 模型预测
        with torch.no_grad():
            v_s, v_e, k = model(coords, expr, mass, t_tensor)
        
        # C. 更新状态 (欧拉积分)
        # 注意：模型输出的速度是在归一化时间下的，需要乘以实际时间步长
        coords_new = coords + v_s * step_dt_actual
        expr_new = expr + v_e * step_dt_actual
        # Mass 更新公式: m(t+dt) = m(t) * exp(k * dt)
        # k 的单位是 1/实际时间，所以这里用实际时间步长
        mass_new = mass * torch.exp(k * step_dt_actual)
        
        # 转回 Numpy
        coords_np = coords_new.numpy()
        expr_np = expr_new.numpy()
        mass_np = mass_new.squeeze(1).numpy()
        
        # D. 分裂与凋亡处理 (Population Dynamics)
        next_cells = []
        for i, old_c in enumerate(current_cells):
            m = mass_np[i]
            m=1
            # 分裂阈值 (通常设为 2.0 左右，表示质量翻倍了)
            # 你可以微调这个 1.8 - 2.0
            if m > 1.9: 
                # === 分裂 ===
                # 变成两个子细胞，质量减半，位置微扰(可选)，基因继承
                child1 = {
                    'coords': coords_np[i],
                    'expr': expr_np[i],
                    'mass': m / 2.0, # 质量减半
                    'id': old_c['id'] + "_c1"
                }
                child2 = {
                    'coords': coords_np[i],
                    'expr': expr_np[i],
                    'mass': m / 2.0,
                    'id': old_c['id'] + "_c2"
                }
                next_cells.append(child1)
                next_cells.append(child2)
            
            # 凋亡阈值 (质量太小就死掉)
            elif m < 0.1:
                # === 凋亡 ===
                continue # 不加入 next_cells
                
            else:
                # === 正常存活 ===
                # 更新属性
                old_c['coords'] = coords_np[i]
                old_c['expr'] = expr_np[i]
                old_c['mass'] = m
                next_cells.append(old_c)
        
        # 更新列表和时间
        current_cells = next_cells
        curr_t_norm += step_dt_norm
        
        # (可选) 打印一下当前细胞数量，防止爆炸
        # if int(curr_t_norm * 100) % 10 == 0:
        #     actual_t = start_time + curr_t_norm * total_dt
        #     print(f"      t_norm={curr_t_norm:.2f} (actual={actual_t:.2f}), count={len(current_cells)}")

    # 3. 结果重组
    if len(current_cells) == 0:
        # 如果全死光了，返回一个空的或者 dummy
        return ad.AnnData(X=np.zeros((1, 50))) 

    final_expr = np.array([c['expr'] for c in current_cells])
    final_coords = np.array([c['coords'] for c in current_cells])
    final_mass = np.array([c['mass'] for c in current_cells])
    
    print(f"    Simulation finished. Final cell count: {len(current_cells)}")
    
    # 构建 AnnData
    pred_adata = ad.AnnData(X=final_expr)
    pred_adata.obsm['X_pca'] = final_expr
    pred_adata.obsm['spatial'] = final_coords
    pred_adata.obs['mass'] = final_mass
    
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
    
    # 全局归一化（非常重要，保证所有折叠的数据尺度一致）
    data_dict, scalers = preprocess_multislice(sorted_adatas, sorted_times)
    
    results = []
    
    # 2. 循环：依次将中间的时间点作为测试集
    # 我们不能去掉第一个(t_0)和最后一个(t_N)，因为那样就变成外推(Extrapolation)了
    # 这里的 Benchmark 主要是测插值(Interpolation)能力
    possible_targets = sorted_times[1:-1] 
    
    print(f"\n==============================================")
    print(f"Starting Leave-One-Out Benchmark")
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
        model = train_model(train_data_list, train_times, epochs=args.epochs)
        
        # C. 预测
        # 找到训练集中最近的“过去”时间点
        past_times = [t for t in train_times if t < target_t]
        start_t = past_times[-1]
        start_data = data_dict[start_t]
        
        print(f"2. Predicting Day {target_t} (starting from Day {start_t})...")
        pred_adata = predict_state(model, start_data, start_t, target_t)
        
        # D. 计算指标
        gt_data = data_dict[target_t] # Ground Truth (Raw normalized dict)
        w2_gene, w2_comb = calculate_benchmark_metrics(pred_adata, gt_data, scalers)
        
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
    
    args = parser.parse_args()
    run_leave_one_out_benchmark(args)