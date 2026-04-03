import bpy
import numpy as np
import mathutils

def fit_line_2d(points_2d):
    """
    拟合 2D 直线: ax + by + c = 0
    使用 SVD。
    points_2d: (N, 2) 数组
    返回 (a, b, c) 归一化向量。
    """
    center = np.mean(points_2d, axis=0)
    uu, dd, vv = np.linalg.svd(points_2d - center)
    normal = vv[1] # 变异最小的方向即为法线
    
    a, b = normal
    c = - (a * center[0] + b * center[1])
    
    return np.array([a, b, c])

def solve_svd(subset_lines):
    # subset_lines: (K, 3)
    if len(subset_lines) < 2: return None
    try:
        u, s, vh = np.linalg.svd(subset_lines)
        v = vh[-1]
        
        if abs(v[2]) < 1e-5:
            return None
        return v[:2] / v[2]
    except:
        return None

def solve_weighted_svd(lines, weights):
    """
    lines: (N, 3) 数组 (a,b,c)
    weights: (N,) 数组，表示线条重要性 (例如长度)
    返回: (u, v)
    """
    if len(lines) < 2: return None
    
    # 按 sqrt(weights) 缩放线条，使 SVD 最小化加权平方误差
    # W * (L . p) = 0
    w_sqrt = np.sqrt(weights)[:, np.newaxis]
    weighted_lines = lines * w_sqrt
    
    try:
        # 对 A (Nx3) 进行 SVD
        # 我们寻找向量 v (3x1) 使得 |A v|^2 最小化，约束条件 |v|=1
        u, s, vh = np.linalg.svd(weighted_lines, full_matrices=False)
        v = vh[-1] # 解是对应于最小奇异值的右奇异向量
        
        if abs(v[2]) < 1e-8: # 无穷远点
            return None
            
        return v[:2] / v[2]
    except:
        return None

def solve_vanishing_point_2d(lines, weights=None, image_diag=2000.0):
    """
    给定线条 (a,b,c) 和可选权重。
    使用迭代重加权最小二乘法 (IRLS) 以保持稳定性。
    """
    n = len(lines)
    if n < 2: return None
    
    if weights is None:
        weights = np.ones(n)
    else:
        weights = np.array(weights)
        # 归一化权重
        if weights.max() > 0:
            weights /= weights.max()
            
    # 当只有2条线时，使用均匀权重
    if n == 2:
        weights = np.ones(n)
    
    # 第一遍：初始加权求解
    vp = solve_weighted_svd(lines, weights)
    if vp is None: return None
    
    # 当只有2条线时，验证消失点的合理性
    if n == 2:
        # 检查消失点是否在合理范围内
        if np.linalg.norm(vp) > image_diag * 5:
            # 消失点太远，可能是平行线情况，返回None
            return None
        # 直接返回，不进行异常值抑制
        return vp
    
    # 第二遍：异常值抑制 (平滑)
    # 计算从 VP 到线条的几何距离
    # dist = |ax + by + c| / sqrt(a^2+b^2) -> 假设 a,b 已归一化
    # 线条在 operators.py 中应该已经归一化
    
    vp_h = np.array([vp[0], vp[1], 1.0])
    dists = np.abs(lines @ vp_h)
    
    # 软重加权 (类 Cauchy 分布)
    # 对较远的线条给予较小权重
    # 阈值大约 10-20 像素
    # 为稳定性/模糊性增加：15.0 -> 40.0
    # 动态阈值：图像对角线的 2%
    # 使用传入的 image_diag 参数
    const_k = image_diag * 0.02

     
    robust_weights = weights / (1.0 + (dists / const_k)**2)
    
    # 第三遍：精细求解
    vp_final = solve_weighted_svd(lines, robust_weights)
    
    return vp_final if vp_final is not None else vp
    
# ... (omitted)

def orthonormalize_matrix(R):
    U, S, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt
    if np.linalg.det(R_ortho) < 0:
       U[:, 2] *= -1
       R_ortho = U @ Vt
    return R_ortho

def calculate_focal_length(vp1, vp2):
    u1, v1 = vp1
    u2, v2 = vp2
    dot = u1*u2 + v1*v2
    if dot < 0: return np.sqrt(-dot)
    return None



def get_effective_f_pixels(f_mm, sensor_width_mm, sensor_height_mm, sensor_fit, pixel_width, pixel_height):
    """
    稳健地计算以像素为单位的焦距，正确处理 'AUTO' 传感器适配。
    """
    if sensor_fit == 'VERTICAL':
        return (f_mm / sensor_height_mm) * pixel_height
    elif sensor_fit == 'HORIZONTAL':
        return (f_mm / sensor_width_mm) * pixel_width
    else: # AUTO
        # Blender AUTO: 如果宽度比 >= 高度比，则适配水平 ?
        # 实际上更简单：它适配相对于传感器长宽比的较大维度？
        # 如果 pixel_width / pixel_height > sensor_width / sensor_height: 适配水平
        # 否则: 适配垂直
        # 通常传感器是 36x24 (3:2 = 1.5)
        # 如果图像是 1920x1080 (16:9 = 1.77) -> 1.77 > 1.5 -> 适配水平
        # 如果图像是 1080x1920 (9:16 = 0.56) -> 0.56 < 1.5 -> 适配垂直
        
        sensor_aspect = sensor_width_mm / sensor_height_mm if sensor_height_mm > 0 else 1.5
        image_aspect = pixel_width / pixel_height if pixel_height > 0 else 1.0
        
        if image_aspect >= sensor_aspect:
             return (f_mm / sensor_width_mm) * pixel_width
        else:
             return (f_mm / sensor_height_mm) * pixel_height

def calculate_camera_transform(vp_data, sensor_width_mm, sensor_height_mm, sensor_fit, pixel_width, pixel_height, current_dist, default_f_mm=50.0, axis_weights=None):
    """
    vp_data: {'X':(u,v), ...} 以中心像素为单位
    axis_weights: {'X': count, ...} 各轴的线段数量权重
    返回 f_mm, rot_matrix, shift_x, shift_y, new_location
    
    增强版：添加焦距合理性验证和置信度评估。
    """
    # 1. 像主点 (Principal Point) & 偏移
    # 默认为 0,0 (中心)，除非用户特别希望求解偏移
    # 对于"保持世界原点在中心"，偏移必须为 0。
    shift_x = 0.0
    shift_y = 0.0
    principal_point = np.array([0.0, 0.0])
    
    # 2. 偏移后的 VP
    vp_data_shifted = {k: np.array(v) - principal_point for k, v in vp_data.items()}
    
    # 3. 焦距
    def calc_f(v1, v2):
        # 过滤：如果 VP 太远（不稳定），则返回 None
        # 阈值：图像尺寸的 10 倍对于"近似平行"是安全的
        # 如果 > 阈值，点积主要由位置决定，对噪声敏感。
        limit = 10.0 * max(pixel_width, pixel_height)
        if np.linalg.norm(v1) > limit or np.linalg.norm(v2) > limit:
             return None
             
        d = np.dot(v1, v2)
        if d < 0: return np.sqrt(-d)
        return None
    
    def validate_focal_length(f_pixels, default_f_pixels, pixel_width, pixel_height):
        """
        验证焦距的合理性。
        返回 (is_valid, confidence_score)
        """
        if f_pixels is None or f_pixels <= 0:
            return False, 0.0
        
        # 检查焦距是否在合理范围内
        # 一般相机焦距在 10mm-300mm 之间，对应像素焦距在图像高度的 0.5-15 倍
        min_f = pixel_height * 0.3
        max_f = pixel_height * 20.0
        
        if f_pixels < min_f or f_pixels > max_f:
            return False, 0.0
        
        # 计算与默认焦距的差异率
        diff_ratio = abs(f_pixels - default_f_pixels) / default_f_pixels
        
        # 差异越小，置信度越高
        if diff_ratio < 0.1:
            confidence = 0.9
        elif diff_ratio < 0.3:
            confidence = 0.7
        elif diff_ratio < 0.5:
            confidence = 0.5
        elif diff_ratio < 1.0:
            confidence = 0.3
        else:
            confidence = 0.1
        
        return True, confidence
        
    # 计算默认焦距的像素值，用于并未参考
    default_f_pixels = get_effective_f_pixels(default_f_mm, sensor_width_mm, sensor_height_mm, sensor_fit, pixel_width, pixel_height)

    f_candidates_info = [] # 存储 (f_val, axes_pair, weight_score)
    
    # 设置默认权重
    if axis_weights is None:
        axis_weights = {k: 1 for k in vp_data.keys()}
    
    if 'X' in vp_data_shifted and 'Y' in vp_data_shifted:
        f = calc_f(vp_data_shifted['X'], vp_data_shifted['Y'])
        if f: 
            # 使用平方权重以拉大差异 (例如 2条=4, 3条=9)
            # 这样 3+3 (18) 会远优于 2+3 (13)
            w = axis_weights.get('X', 0)**2 + axis_weights.get('Y', 0)**2
            f_candidates_info.append((f, {'X', 'Y'}, w))
            
    if 'X' in vp_data_shifted and 'Z' in vp_data_shifted:
        f = calc_f(vp_data_shifted['X'], vp_data_shifted['Z'])
        if f: 
            w = axis_weights.get('X', 0)**2 + axis_weights.get('Z', 0)**2
            f_candidates_info.append((f, {'X', 'Z'}, w))
            
    if 'Y' in vp_data_shifted and 'Z' in vp_data_shifted:
        f = calc_f(vp_data_shifted['Y'], vp_data_shifted['Z'])
        if f: 
            w = axis_weights.get('Y', 0)**2 + axis_weights.get('Z', 0)**2
            f_candidates_info.append((f, {'Y', 'Z'}, w))
        
    f_mm_final = default_f_mm
    
    # 跟踪哪些轴是认为“可靠”的
    # 默认为全部存在
    trusted_axes = set(vp_data_shifted.keys())

    if f_candidates_info:
        # 首先按权重排序 (降序)，然后按与默认焦距的差异排序 (升序)
        # 这里的逻辑是：权重是第一优先级。
        # 如果权重有显著差异，绝对优先使用高权重的解。
        
        # 找出最大权重
        max_weight = max(x[2] for x in f_candidates_info)
        
        # 筛选出具有最大权重的候选项
        best_candidates = [x for x in f_candidates_info if x[2] == max_weight]
        
        # 如果只有一个最高权重的，直接使用
        if len(best_candidates) == 1:
            best_choice = best_candidates[0]
            f_pixels = best_choice[0]
            trusted_axes = best_choice[1]
            # print(f"DEBUG: Selected by weight {best_choice[2]}: {trusted_axes} f={f_pixels:.1f}")
        else:
            # 如果有多个相同权重的，或者大家权重都一样
            # 则使用最接近默认焦距的那个（假设用户大概知道焦距范围）
            # 或者取平均值？取平均值可能更好，如果数据一致的话。
            # 但如果数据不一致（标准差大），取最接近默认值的更安全。
            
            f_vals = [x[0] for x in best_candidates]
            f_mean = np.mean(f_vals)
            f_std = np.std(f_vals)
            
            if f_std < f_mean * 0.1: # 差异不大，取平均
                f_pixels = f_mean
                # trust 所有的组合？取并集
                trusted_axes = set()
                for x in best_candidates:
                    trusted_axes.update(x[1])
            else:
                 # 差异较大，选最接近默认的
                 best_diff = float('inf')
                 best_sub_choice = None
                 for cand in best_candidates:
                     diff = abs(cand[0] - default_f_pixels)
                     if diff < best_diff:
                         best_diff = diff
                         best_sub_choice = cand
                 
                 if best_sub_choice:
                     f_pixels = best_sub_choice[0]
                     trusted_axes = best_sub_choice[1]

        # 验证焦距合理性
        is_valid, confidence = validate_focal_length(f_pixels, default_f_pixels, pixel_width, pixel_height)
        
        if not is_valid:
            # 如果焦距不合理，使用默认焦距
            f_pixels = default_f_pixels
            # 保持所有轴为可信
            trusted_axes = set(vp_data_shifted.keys())
        
        # 根据置信度调整焦距
        if confidence < 0.5:
            # 低置信度时，向默认焦距靠拢
            blend_factor = 1.0 - confidence  # 置信度越低，混合比例越高
            f_pixels = f_pixels * (1.0 - blend_factor) + default_f_pixels * blend_factor
        
        # 使用 Robust Helper 基于传感器适配将 f_pixels 转换为 f_mm
        # 逻辑：f_mm = f_pixels / effective_pixel_size * sensor_size
        # 或者简单地反转 get_effective_f_pixels 逻辑？
        # 等等，get_effective_f_pixels: f_mm -> f_pixels
        # 这里我们有 f_pixels -> f_mm。
        
        # 确定适配模式，与 helper 相同
        sensor_aspect = sensor_width_mm / sensor_height_mm if sensor_height_mm > 0 else 1.5
        image_aspect = pixel_width / pixel_height if pixel_height > 0 else 1.0
        
        fit_mode = sensor_fit
        if fit_mode == 'AUTO':
            if image_aspect >= sensor_aspect: fit_mode = 'HORIZONTAL'
            else: fit_mode = 'VERTICAL'
            
        if fit_mode == 'VERTICAL':
             val_mm = (f_pixels / pixel_height) * sensor_height_mm
        else:
             val_mm = (f_pixels / pixel_width) * sensor_width_mm

        # 更严格的焦距范围检查
        if 10.0 < val_mm < 2000.0:
            f_mm_final = val_mm
        else:
            # 焦距超出范围，使用默认值
            f_mm_final = default_f_mm
            
    # 重新计算 f_pixels (旋转向量需要)
    # 必须反相使用相同的逻辑以获得一致的像素值
    f_pixels = get_effective_f_pixels(f_mm_final, sensor_width_mm, sensor_height_mm, sensor_fit, pixel_width, pixel_height)
    
    # 4. 旋转
    def get_cam_vec(vp):
         # 如果 VP 很远 (无穷大)，归一化 (u,v,0)
         if np.linalg.norm(vp) > 1e7:
             v = np.array([vp[0], vp[1], 0.0])
         else:
             v = np.array([vp[0], vp[1], -f_pixels])
         return v / np.linalg.norm(v)
        
    vx = get_cam_vec(vp_data_shifted['X']) if 'X' in list(vp_data_shifted.keys()) and 'X' in trusted_axes else None
    vy = get_cam_vec(vp_data_shifted['Y']) if 'Y' in list(vp_data_shifted.keys()) and 'Y' in trusted_axes else None
    vz = get_cam_vec(vp_data_shifted['Z']) if 'Z' in list(vp_data_shifted.keys()) and 'Z' in trusted_axes else None
    
    current_cols = {}
    if vx is not None: current_cols['X'] = vx
    if vy is not None: current_cols['Y'] = vy
    if vz is not None: current_cols['Z'] = vz
    
    if 'X' in current_cols and 'Y' in current_cols and 'Z' not in current_cols:
        current_cols['Z'] = np.cross(current_cols['X'], current_cols['Y'])
    elif 'X' in current_cols and 'Z' in current_cols and 'Y' not in current_cols:
        current_cols['Y'] = np.cross(current_cols['Z'], current_cols['X'])
    elif 'Y' in current_cols and 'Z' in current_cols and 'X' not in current_cols:
         current_cols['X'] = np.cross(current_cols['Y'], current_cols['Z'])
    
    if 'X' not in current_cols or 'Y' not in current_cols or 'Z' not in current_cols:
        return None, None, 0, 0, None
        
    rx, ry, rz = current_cols['X'], current_cols['Y'], current_cols['Z']
    
    # 强制 Z 轴朝上检查
    if rz[1] < 0:
        ry = -ry
        rz = -rz
        current_cols['Y'] = ry
        current_cols['Z'] = rz
        
    # 初始正交化
    R_raw = np.column_stack((rx, ry, rz))
    R_ortho = orthonormalize_matrix(R_raw)
    rot_matrix = mathutils.Matrix(R_ortho.T)
    
    # 5. 位置 (轨道)
    ray_cam = np.array([
        0.0 - principal_point[0],
        0.0 - principal_point[1],
        -f_pixels
    ])
    ray_cam = ray_cam / np.linalg.norm(ray_cam) 
    
    # 世界空间的相机原始位置应该沿着这条射线距离 'dist'
    # P_org_in_cam = dist * ray_cam
    p_org_cam = ray_cam * current_dist
    
    # 世界空间中的相机位置
    # P_org_world = R_cw @ P_org_cam + C_world
    # 0 = R_cw @ P_org_cam + C_world
    # C_world = - R_cw @ P_org_cam
    
    # rot_matrix 是 R_cw (相机 -> 世界)
    vec_org_cam = mathutils.Vector(p_org_cam)
    
    loc_orbit = -(rot_matrix @ vec_org_cam)
    
    return f_mm_final, rot_matrix, shift_x, shift_y, loc_orbit

def solve_camera_rotation_constrained(lines_data, f_pixels, cx, cy, current_rot_matrix):
    """
    使用单线（平面）和固定焦距求解旋转。
    lines_data: {'X': [[a,b,c,len],...], 'Y':...}
    f_pixels: 当前像素焦距
    current_rot_matrix: 3x3 mathutils 矩阵 (世界到相机? 不，相机方向)
                        Blender 相机矩阵: Col 0=右, Col 1=上, Col 2=后.
    返回: rot_matrix (3x3)
    """
    
    # 1. 计算每个轴的平面法线
    # 相机空间中的平面法线: (a, b, -c/f)
    
    normals = {}
    for axis in ['X', 'Y', 'Z']:
        if axis not in lines_data or not lines_data[axis]:
            continue
            
        # 收集该轴的所有法线并求平均？
        # 或者对线条使用 SVD 寻找最佳法线？
        # 让我们只是为了稳健性平均法线
        axis_normals = []
        for line in lines_data[axis]:
            a, b, c, length = line
            # 法线: (a, b, -c/f_pixels)
            n = np.array([a, b, -c / f_pixels])
            n = n / np.linalg.norm(n)
            # 按长度加权
            axis_normals.append(n * length)
            
        if axis_normals:
            sum_n = np.sum(axis_normals, axis=0)
            normals[axis] = sum_n / np.linalg.norm(sum_n)
            
    if len(normals) < 2:
        return None # 需要至少 2 个轴
        
    # 2. 优化
    # 我们希望 R = [rx, ry, rz] 使得 rx 垂直 Nx, ry 垂直 Ny, rz 垂直 Nz
    # 从当前旋转开始
    R_curr = np.array(current_rot_matrix).T # 转置以获得 [rx, ry, rz] 作为列？
    # Blender 矩阵:
    # Col 0: 右 (世界 X 在相机空间？不。相机 X 在世界空间。)
    # 等等，我们想要相机方向矩阵 R_cam_to_world。
    # 但在这里我们在相机空间工作。
    # 世界 X 轴在相机空间是 R_world_to_cam * [1,0,0]^T = R_world_to_cam 的第 0 列。
    # R_world_to_cam = R_cam_to_world.T
    # 所以我们正在寻找 R_world_to_cam 的列。
    # 令 R = R_world_to_cam。列是 u, v, w (在相机中看到的世界 X, Y, Z)。
    # u 垂直 Nx, v 垂直 Ny, w 垂直 Nz。
    
    # 使用当前的 R_world_to_cam 初始化
    R = np.array(current_rot_matrix).T # current_rot_matrix 是 Cam->World。转置 -> World->Cam。
    # 确保正交仅防万一
    R = orthonormalize_matrix(R) 
    
    # 迭代投影
    for i in range(20):
        # 1. 投影列到平面
        u, v, w = R[:, 0], R[:, 1], R[:, 2]
        
        if 'X' in normals:
            n = normals['X']
            u = u - np.dot(u, n) * n
            if np.linalg.norm(u) > 1e-6: u /= np.linalg.norm(u)
            
        if 'Y' in normals:
            n = normals['Y']
            v = v - np.dot(v, n) * n
            if np.linalg.norm(v) > 1e-6: v /= np.linalg.norm(v)
            
        if 'Z' in normals:
            n = normals['Z']
            w = w - np.dot(w, n) * n
            if np.linalg.norm(w) > 1e-6: w /= np.linalg.norm(w)
            
        # 2. 正交化 (SVD)
        R_new = np.column_stack((u, v, w))
        R = orthonormalize_matrix(R_new)
        
    # 3. 检查 Z 轴朝上 (如果需要则翻转)
    # R 第 2 列是相机空间中的世界 Z。
    # 相机 Y 是上。所以 R[1, 2] 应该是正的？
    # 实际上，通过水平观察，世界 Z 在图像中应该是“向上”的。
    # 如果 R[1, 2] < 0，Z 是指向下的。
    if R[1, 2] < 0:
        # 翻转世界的 Y 和 Z 轴
        # 交换第 1 和 第 2 列？不，那会改变手性。
        # 绕 X 旋转 180？
        # 翻转 Y 和 Z 列 -> 改变手性。
        # 翻转 Y 和 Z 列并取反一个？
        # 让我们直接取反 Y 和 Z 列。
        R[:, 1] = -R[:, 1]
        R[:, 2] = -R[:, 2]
        
    # 作为 Blender 矩阵返回 (Cam->World)
    return mathutils.Matrix(R.T)
