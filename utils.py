import bpy
import math
import numpy as np
import mathutils
from bpy_extras import view3d_utils


def is_camera_view(context):
    area = getattr(context, "area", None)
    space_data = getattr(context, "space_data", None)
    rv3d = getattr(space_data, "region_3d", None)
    if not area or area.type != 'VIEW_3D' or rv3d is None:
        return False
    return rv3d.view_perspective == 'CAMERA'


def get_camera_view_region_data(context):
    area = getattr(context, "area", None)
    space_data = getattr(context, "space_data", None)
    if not area or area.type != 'VIEW_3D' or space_data is None:
        return None

    rv3d = getattr(context, "region_data", None) or getattr(space_data, "region_3d", None)
    if rv3d is None or rv3d.view_perspective != 'CAMERA':
        return None

    scene = getattr(context, "scene", None)
    if scene is None or scene.camera is None:
        return None

    space_camera = getattr(space_data, "camera", None)
    if space_camera is not None and space_camera != scene.camera:
        return None

    return rv3d


def iter_camera_view_regions(context):
    scene = getattr(context, "scene", None)
    screen = getattr(context, "screen", None)

    if scene is None or scene.camera is None:
        return

    if screen is not None:
        for area in screen.areas:
            if area.type != 'VIEW_3D':
                continue

            space = getattr(area.spaces, "active", None)
            if space is None or space.type != 'VIEW_3D':
                continue

            rv3d = getattr(space, "region_3d", None)
            if rv3d is None or rv3d.view_perspective != 'CAMERA':
                continue

            space_camera = getattr(space, "camera", None)
            if space_camera is not None and space_camera != scene.camera:
                continue

            yield area, rv3d
        return

    area = getattr(context, "area", None)
    rv3d = get_camera_view_region_data(context)
    if area is not None and rv3d is not None:
        yield area, rv3d


def capture_camera_view_state(context):
    states = []
    for area, rv3d in iter_camera_view_regions(context):
        states.append({
            "area": area,
            "region_3d": rv3d,
            "offset": tuple(rv3d.view_camera_offset),
            "zoom": rv3d.view_camera_zoom,
        })
    return states


def restore_camera_view_state(state):
    if not state:
        return

    for entry in state:
        area = entry.get("area")
        rv3d = entry.get("region_3d")
        if area is None or rv3d is None:
            continue

        try:
            if getattr(rv3d, "view_perspective", None) != 'CAMERA':
                continue
            rv3d.view_camera_offset = entry["offset"]
            rv3d.view_camera_zoom = entry["zoom"]
            area.tag_redraw()
        except Exception:
            continue


def get_effective_render_size(render):
    scale = render.resolution_percentage / 100.0
    res_x = render.resolution_x * scale * render.pixel_aspect_x
    res_y = render.resolution_y * scale * render.pixel_aspect_y
    return res_x, res_y


def uv_to_centered_px(uv, pixel_res_x, pixel_res_y, principal_u=0.5, principal_v=0.5):
    u, v = uv
    return np.array([
        (float(u) - principal_u) * float(pixel_res_x),
        (float(v) - principal_v) * float(pixel_res_y),
    ], dtype=float)


def centered_px_to_uv(point_px, pixel_res_x, pixel_res_y, principal_u=0.5, principal_v=0.5):
    x, y = point_px
    px = float(pixel_res_x) if pixel_res_x else 1.0
    py = float(pixel_res_y) if pixel_res_y else 1.0
    return (
        float(x) / px + principal_u,
        float(y) / py + principal_v,
    )


def render_centered_px_to_region_xy(point_px, render_res_x, render_res_y, region_w, region_h):
    u, v = centered_px_to_uv(point_px, render_res_x, render_res_y)
    return np.array([
        float(u) * float(region_w),
        float(v) * float(region_h),
    ], dtype=float)


def region_xy_to_render_centered_px(point_xy, render_res_x, render_res_y, region_w, region_h):
    rw = float(region_w) if region_w else 1.0
    rh = float(region_h) if region_h else 1.0
    u = float(point_xy[0]) / rw
    v = float(point_xy[1]) / rh
    return uv_to_centered_px((u, v), render_res_x, render_res_y)


def camera_frame_uv_to_world(context, u, v):
    scene = getattr(context, "scene", None)
    cam = getattr(scene, "camera", None) if scene is not None else None
    if cam is None:
        return None

    TR, TL, BL, BR = get_ordered_frame_points(context)
    if not TR:
        return None

    top = TL.lerp(TR, float(u))
    bot = BL.lerp(BR, float(u))
    p_loc = bot.lerp(top, float(v))
    return cam.matrix_world @ p_loc


def render_centered_px_to_camera_region_xy(context, point_px, render_res_x, render_res_y):
    region = getattr(context, "region", None)
    rv3d = get_camera_view_region_data(context)
    if region is None or rv3d is None:
        return None

    uv = centered_px_to_uv(point_px, render_res_x, render_res_y)
    p_world = camera_frame_uv_to_world(context, uv[0], uv[1])
    if p_world is None:
        return None

    p_region = view3d_utils.location_3d_to_region_2d(region, rv3d, p_world)
    if p_region is None:
        return None

    return np.array([float(p_region[0]), float(p_region[1])], dtype=float)


def build_axis_line_data(lines, pixel_res_x, pixel_res_y, principal_u=0.5, principal_v=0.5):
    cx = pixel_res_x * principal_u
    cy = pixel_res_y * principal_v
    lines_data = {'X': [], 'Y': [], 'Z': []}

    for line in lines:
        u1, v1 = line.start
        u2, v2 = line.end

        px1 = u1 * pixel_res_x - cx
        py1 = v1 * pixel_res_y - cy
        px2 = u2 * pixel_res_x - cx
        py2 = v2 * pixel_res_y - cy

        dx = px2 - px1
        dy = py2 - py1
        length = np.hypot(dx, dy)
        if length < 10:
            continue

        a = -dy / length
        b = dx / length
        c = -(a * px1 + b * py1)

        if line.axis in lines_data:
            lines_data[line.axis].append([a, b, c, length])

    return lines_data


def clone_lines_data(lines_data):
    return {
        axis: [list(item) for item in lines_data.get(axis, [])]
        for axis in ('X', 'Y', 'Z')
    }


def build_perspective_mode_constraints(lines_data, pixel_res_x, pixel_res_y, finite_vp_axes=None):
    counts = {axis: len(lines_data.get(axis, [])) for axis in ('X', 'Y', 'Z')}
    active_axes = [axis for axis, count in counts.items() if count >= 1]
    vp_capable_axes = [axis for axis, count in counts.items() if count >= 2]

    axis_order = {'X': 0, 'Y': 1, 'Z': 2}
    finite_vp_axes = [axis for axis in (finite_vp_axes or []) if axis in axis_order]
    finite_vp_axes = sorted(set(finite_vp_axes), key=lambda axis: axis_order[axis])

    guided_lines_data = clone_lines_data(lines_data)
    guided_axes = {}
    primary_axis = None
    base_axes = []
    missing_axis = None
    mode = 'INSUFFICIENT'

    image_diag = max(float(np.hypot(pixel_res_x, pixel_res_y)), 1.0)
    guide_length = image_diag * 0.35

    def set_guided_axis(axis, orientation):
        if orientation == 'VERTICAL':
            guided_lines_data[axis] = [[1.0, 0.0, 0.0, guide_length]]
        else:
            guided_lines_data[axis] = [[0.0, 1.0, 0.0, guide_length]]
        guided_axes[axis] = orientation

    if len(active_axes) >= 3:
        mode = 'THREE_POINT'
        base_axes = ['X', 'Y', 'Z']

    elif len(active_axes) == 2:
        mode = 'TWO_POINT'
        if len(finite_vp_axes) >= 2:
            base_axes = sorted(
                finite_vp_axes,
                key=lambda axis: (-counts.get(axis, 0), axis_order[axis]),
            )[:2]
        else:
            base_axes = sorted(active_axes, key=lambda axis: axis_order[axis])

    elif len(active_axes) == 1:
        mode = 'ONE_POINT'
        if finite_vp_axes:
            primary_axis = max(finite_vp_axes, key=lambda axis: (counts.get(axis, 0), -axis_order[axis]))
        else:
            primary_axis = active_axes[0]
        base_axes = [primary_axis]

    if mode == 'TWO_POINT':
        missing_axis = next((axis for axis in ('X', 'Y', 'Z') if axis not in base_axes), None)
        missing_orientation_map = {
            'X': 'HORIZONTAL',
            'Y': 'VERTICAL',
            'Z': 'VERTICAL',
        }
        if missing_axis is not None:
            set_guided_axis(missing_axis, missing_orientation_map.get(missing_axis, 'VERTICAL'))

    elif mode == 'ONE_POINT':
        if primary_axis is None:
            primary_axis = max(('X', 'Y', 'Z'), key=lambda axis: counts.get(axis, 0))

        one_point_guides = {
            'X': {'Y': 'VERTICAL', 'Z': 'HORIZONTAL'},
            'Y': {'X': 'HORIZONTAL', 'Z': 'VERTICAL'},
            'Z': {'X': 'HORIZONTAL', 'Y': 'VERTICAL'},
        }
        for axis in ('X', 'Y', 'Z'):
            if axis == primary_axis:
                continue
            orientation = one_point_guides.get(primary_axis, {}).get(axis, 'VERTICAL')
            set_guided_axis(axis, orientation)

    return {
        'mode': mode,
        'active_axes': active_axes,
        'base_axes': base_axes,
        'vp_capable_axes': vp_capable_axes,
        'finite_vp_axes': finite_vp_axes,
        'primary_axis': primary_axis,
        'missing_axis': missing_axis,
        'guided_axes': guided_axes,
        'guided_lines_data': guided_lines_data,
        'counts': counts,
    }



def solve_vanishing_points(lines_data, pixel_res_x, pixel_res_y):
    vp_data = {}
    axis_weights = {}
    image_diag = np.hypot(pixel_res_x, pixel_res_y)

    for axis in ['X', 'Y', 'Z']:
        data = lines_data[axis]
        count = len(data)
        axis_weights[axis] = count

        if count < 2:
            continue

        arr = np.array(data)
        lines_abc = arr[:, :3]
        weights = arr[:, 3]

        if count == 2:
            weights = np.ones(count)

        vp = solve_vanishing_point_2d(lines_abc, weights, image_diag=image_diag)
        if vp is not None:
            vp_data[axis] = vp

    return vp_data, axis_weights


def rotate_vector_2d(vec, angle_rad):
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([vec[0] * c - vec[1] * s, vec[0] * s + vec[1] * c], dtype=float)


def compute_adjusted_horizon(vp_data, offset_px=0.0):
    has_x = 'X' in vp_data
    has_y = 'Y' in vp_data

    if not has_x and not has_y:
        return None

    if has_x and has_y:
        vx = np.array(vp_data['X'], dtype=float)
        vy = np.array(vp_data['Y'], dtype=float)
        point = (vx + vy) * 0.5
        direction = vy - vx
        if np.linalg.norm(direction) <= 1e-8:
            direction = np.array([1.0, 0.0], dtype=float)
    elif has_x:
        vx = np.array(vp_data['X'], dtype=float)
        point = np.array([0.0, float(vx[1])], dtype=float)
        direction = np.array([1.0, 0.0], dtype=float)
    else:
        vy = np.array(vp_data['Y'], dtype=float)
        point = np.array([0.0, float(vy[1])], dtype=float)
        direction = np.array([1.0, 0.0], dtype=float)

    dir_norm = np.linalg.norm(direction)
    if dir_norm <= 1e-8:
        direction = np.array([1.0, 0.0], dtype=float)
    else:
        direction = direction / dir_norm


    normal = np.array([-direction[1], direction[0]], dtype=float)
    normal_norm = np.linalg.norm(normal)
    if normal_norm <= 1e-8:
        normal = np.array([0.0, 1.0], dtype=float)
    else:
        normal = normal / normal_norm

    point = point + normal * float(offset_px)

    a, b = normal
    c = -(a * point[0] + b * point[1])

    return {
        'point': point,
        'direction': direction,
        'normal': normal,
        'line': np.array([a, b, c], dtype=float),
    }


def project_point_to_line_2d(point, line_point, line_direction):
    p = np.array(point, dtype=float)
    p0 = np.array(line_point, dtype=float)
    d = np.array(line_direction, dtype=float)
    denom = np.dot(d, d)
    if denom < 1e-12:
        return p
    t = np.dot(p - p0, d) / denom
    return p0 + d * t


def signed_distance_to_line_2d(point, line):
    p = np.array(point, dtype=float)
    ln = np.array(line, dtype=float)
    nrm = np.hypot(ln[0], ln[1])
    if nrm < 1e-12:
        return 0.0
    return float((ln[0] * p[0] + ln[1] * p[1] + ln[2]) / nrm)


def solve_horizon_data(lines, pixel_res_x, pixel_res_y, horizon_enabled, horizon_offset_px, principal_u=0.5, principal_v=0.5):
    lines_data = build_axis_line_data(lines, pixel_res_x, pixel_res_y, principal_u, principal_v)
    vp_raw, axis_weights = solve_vanishing_points(lines_data, pixel_res_x, pixel_res_y)
    vp_adj, horizon_data = apply_horizon_constraint_to_vps(
        vp_raw,
        enabled=horizon_enabled,
        offset_px=horizon_offset_px,
    )
    return lines_data, vp_raw, vp_adj, axis_weights, horizon_data


def compute_horizon_overlay_geometry(lines, cmp_data, pixel_res_x, pixel_res_y, region_width, region_height, context=None):
    _lines_data, vp_raw, _vp_adj, _axis_weights, horizon_data = solve_horizon_data(
        lines,
        pixel_res_x,
        pixel_res_y,
        True,
        cmp_data.horizon_offset_px,
    )
    if horizon_data is None:
        return None

    # 计算相机视口中心 (0, 0) 在地平线上的投影作为平均手柄中心点
    p0 = np.array(horizon_data['point'], dtype=float)
    dir_vec = np.array(horizon_data['direction'], dtype=float)
    proj_t = -np.dot(p0, dir_vec)
    handle_render = p0 + proj_t * dir_vec

    # 计算地平线与图像边界 (-w/2, -h/2) 到 (w/2, h/2) 的交点
    hw, hh = float(pixel_res_x) * 0.5, float(pixel_res_y) * 0.5
    valid_ts = []
    if abs(dir_vec[0]) > 1e-8:
        t1 = (-hw - p0[0]) / dir_vec[0]
        y1 = p0[1] + t1 * dir_vec[1]
        if -hh - 1e-4 <= y1 <= hh + 1e-4: valid_ts.append(t1)
        
        t2 = (hw - p0[0]) / dir_vec[0]
        y2 = p0[1] + t2 * dir_vec[1]
        if -hh - 1e-4 <= y2 <= hh + 1e-4: valid_ts.append(t2)
        
    if abs(dir_vec[1]) > 1e-8:
        t3 = (-hh - p0[1]) / dir_vec[1]
        x3 = p0[0] + t3 * dir_vec[0]
        if -hw - 1e-4 <= x3 <= hw + 1e-4: valid_ts.append(t3)
        
        t4 = (hh - p0[1]) / dir_vec[1]
        x4 = p0[0] + t4 * dir_vec[0]
        if -hw - 1e-4 <= x4 <= hw + 1e-4: valid_ts.append(t4)
        
    if len(valid_ts) >= 2:
        valid_ts.sort()
        line_a_render = p0 + valid_ts[0] * dir_vec
        line_b_render = p0 + valid_ts[-1] * dir_vec
        draw_line = True
    else:
        # 如果地平线完全在画面外，就不画线，但保留手柄
        line_a_render = handle_render
        line_b_render = handle_render
        draw_line = False

    if context is not None:
        center_region = render_centered_px_to_camera_region_xy(
            context,
            horizon_data['point'],
            pixel_res_x,
            pixel_res_y,
        )

        offset_handle_region = render_centered_px_to_camera_region_xy(
            context,
            handle_render,
            pixel_res_x,
            pixel_res_y,
        )

        line_a_region = render_centered_px_to_camera_region_xy(
            context,
            line_a_render,
            pixel_res_x,
            pixel_res_y,
        )
        line_b_region = render_centered_px_to_camera_region_xy(
            context,
            line_b_render,
            pixel_res_x,
            pixel_res_y,
        )

        if (
            center_region is not None
            and offset_handle_region is not None
            and line_a_region is not None
            and line_b_region is not None
        ):
            dir_region = line_b_region - line_a_region
            dir_region_norm = np.linalg.norm(dir_region)
            if dir_region_norm > 1e-8:
                dir_region = dir_region / dir_region_norm
                nrm_region = np.array([-dir_region[1], dir_region[0]], dtype=float)
            else:
                dir_region = np.array([1.0, 0.0], dtype=float)
                nrm_region = np.array([0.0, 1.0], dtype=float)

            viewport_center_region = np.array([
                float(region_width) * 0.5,
                float(region_height) * 0.5,
            ], dtype=float)

            return {
                'horizon': horizon_data,
                'center_region': center_region,
                'viewport_center_region': viewport_center_region,
                'direction_region': dir_region,
                'normal_region': nrm_region,
                'line_region_a': line_a_region,
                'line_region_b': line_b_region,
                'offset_handle_region': offset_handle_region,
                'draw_line': draw_line,
            }

    center_region = render_centered_px_to_region_xy(
        horizon_data['point'],
        pixel_res_x,
        pixel_res_y,
        region_width,
        region_height,
    )

    offset_handle_region = render_centered_px_to_region_xy(
        handle_render,
        pixel_res_x,
        pixel_res_y,
        region_width,
        region_height,
    )

    viewport_center_region = np.array([
        float(region_width) * 0.5,
        float(region_height) * 0.5,
    ], dtype=float)

    line_region_a = render_centered_px_to_region_xy(
        line_a_render,
        pixel_res_x,
        pixel_res_y,
        region_width,
        region_height,
    )
    line_region_b = render_centered_px_to_region_xy(
        line_b_render,
        pixel_res_x,
        pixel_res_y,
        region_width,
        region_height,
    )
    dir_region = line_region_b - line_region_a
    dir_region_norm = np.linalg.norm(dir_region)
    if dir_region_norm <= 1e-8:
        dir_region = np.array([1.0, 0.0], dtype=float)
        nrm_region = np.array([0.0, 1.0], dtype=float)
    else:
        dir_region = dir_region / dir_region_norm
        nrm_region = np.array([-dir_region[1], dir_region[0]], dtype=float)

    return {
        'horizon': horizon_data,
        'center_region': center_region,
        'viewport_center_region': viewport_center_region,
        'direction_region': dir_region,
        'normal_region': nrm_region,
        'line_region_a': line_region_a,
        'line_region_b': line_region_b,
        'offset_handle_region': offset_handle_region,
        'draw_line': draw_line,
    }



def apply_horizon_constraint_to_vps(vp_data, enabled=False, offset_px=0.0):
    if not enabled:
        return vp_data, None

    horizon = compute_adjusted_horizon(vp_data, offset_px=offset_px)
    if horizon is None:
        return vp_data, None

    adjusted = {k: np.array(v, dtype=float) for k, v in vp_data.items()}
    for axis in ('X', 'Y'):
        if axis not in adjusted:
            continue
        p = adjusted[axis]
        adjusted[axis] = project_point_to_line_2d(p, horizon['point'], horizon['direction'])

    return adjusted, horizon


def distance_point_to_segment_2d(point, start, end):
    point = np.array(point, dtype=float)
    start = np.array(start, dtype=float)
    end = np.array(end, dtype=float)
    segment = end - start
    segment_length_sq = np.dot(segment, segment)
    if segment_length_sq <= 1e-12:
        return np.linalg.norm(point - start)
    t = np.dot(point - start, segment) / segment_length_sq
    t = np.clip(t, 0.0, 1.0)
    closest = start + t * segment
    return np.linalg.norm(point - closest)


def rotate_matrix_around_point(matrix_world, rotation_matrix, pivot):
    return (
        mathutils.Matrix.Translation(pivot)
        @ rotation_matrix
        @ mathutils.Matrix.Translation(-pivot)
        @ matrix_world
    )


def register_class_safe(cls):
    try:
        bpy.utils.register_class(cls)
    except ValueError:
        bpy.utils.unregister_class(cls)
        bpy.utils.register_class(cls)


def unregister_class_safe(cls):
    try:
        bpy.utils.unregister_class(cls)
    except Exception:
        pass


def get_ordered_frame_points(context):
    cam = context.scene.camera
    if not cam:
        return None, None, None, None

    try:
        frame = cam.data.view_frame(scene=context.scene)
    except Exception:
        return None, None, None, None

    TR, TL, BL, BR = None, None, None, None
    for v in frame:
        if v.x > 0 and v.y > 0:
            TR = v
        elif v.x < 0 and v.y > 0:
            TL = v
        elif v.x < 0 and v.y < 0:
            BL = v
        elif v.x > 0 and v.y < 0:
            BR = v

    if not all([TR, TL, BL, BR]):
        return frame[0], frame[1], frame[3], frame[2]
    return TR, TL, BL, BR




def solve_weighted_svd(lines, weights):
    """
    lines: (N, 3) 数组 (a,b,c)
    weights: (N,) 数组，表示线条重要性 (例如长度)
    返回: (u, v)
    """
    if len(lines) < 2:
        return None

    lines = np.asarray(lines, dtype=float)
    weights = np.asarray(weights, dtype=float)

    # 2x3 矩阵的薄型 SVD 不包含齐次零空间，直接用叉乘求两条线的交点。
    if len(lines) == 2:
        point_h = np.cross(lines[0], lines[1])
        if not np.all(np.isfinite(point_h)) or abs(point_h[2]) < 1e-8:
            return None
        return point_h[:2] / point_h[2]
    
    # 按 sqrt(weights) 缩放线条，使 SVD 最小化加权平方误差
    # W * (L . p) = 0
    w_sqrt = np.sqrt(weights)[:, np.newaxis]
    weighted_lines = lines * w_sqrt
    
    try:
        # 对 A (Nx3) 进行 SVD
        # 我们寻找向量 v (3x1) 使得 |A v|^2 最小化，约束条件 |v|=1
        _u, _s, vh = np.linalg.svd(weighted_lines, full_matrices=False)
        v = vh[-1] # 解是对应于最小奇异值的右奇异向量
        
        if abs(v[2]) < 1e-8: # 无穷远点
            return None
            
        return v[:2] / v[2]
    except Exception:
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

        if pixel_width >= pixel_height:
             return (f_mm / sensor_width_mm) * pixel_width
        else:
             return (f_mm / sensor_height_mm) * pixel_height


def get_effective_f_mm_from_pixels(f_pixels, sensor_width_mm, sensor_height_mm, sensor_fit, pixel_width, pixel_height):
    if not np.isfinite(f_pixels) or f_pixels <= 1e-8:
        return None

    fit_mode = sensor_fit
    if fit_mode == 'AUTO':
        sensor_aspect = sensor_width_mm / sensor_height_mm if sensor_height_mm > 0 else 1.5
        image_aspect = pixel_width / pixel_height if pixel_height > 0 else 1.0
        fit_mode = 'HORIZONTAL' if pixel_width >= pixel_height else 'VERTICAL'

    if fit_mode == 'VERTICAL':
        if pixel_height <= 1e-8 or sensor_height_mm <= 1e-8:
            return None
        return float((f_pixels / pixel_height) * sensor_height_mm)

    if pixel_width <= 1e-8 or sensor_width_mm <= 1e-8:
        return None
    return float((f_pixels / pixel_width) * sensor_width_mm)



def calculate_camera_transform(vp_data, sensor_width_mm, sensor_height_mm, sensor_fit, pixel_width, pixel_height, current_dist, default_f_mm=50.0, axis_weights=None, anchor_location=None, anchor_screen_offset=None):
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
            if pixel_width >= pixel_height: fit_mode = 'HORIZONTAL'
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
    target_px = 0.0 - principal_point[0]
    target_py = 0.0 - principal_point[1]
    if anchor_screen_offset is not None:
        target_px, target_py = anchor_screen_offset

    ray_cam = np.array([
        target_px,
        target_py,
        -f_pixels
    ])
    ray_cam = ray_cam / np.linalg.norm(ray_cam) 
    
    # 世界空间的相机原始位置应该沿着这条射线距离 'dist'
    # P_org_in_cam = dist * ray_cam
    p_org_cam = ray_cam * current_dist
    
    # 世界空间中的相机位置
    # P_org_world = R_cw @ P_org_cam + C_world
    # anchor = R_cw @ P_org_cam + C_world
    # C_world = anchor - R_cw @ P_org_cam
    anchor = mathutils.Vector(anchor_location) if anchor_location is not None else mathutils.Vector((0.0, 0.0, 0.0))
    vec_org_cam = mathutils.Vector(p_org_cam)

    loc_orbit = anchor - (rot_matrix @ vec_org_cam)

    return f_mm_final, rot_matrix, shift_x, shift_y, loc_orbit

def solve_camera_rotation_constrained(lines_data, f_pixels, current_rot_matrix):
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
        reference_normal = None
        for line in lines_data[axis]:
            a, b, c, length = line
            # 法线: (a, b, -c/f_pixels)
            n = np.array([a, b, -c / f_pixels], dtype=float)
            n_length = np.linalg.norm(n)
            if not np.isfinite(n_length) or n_length <= 1e-8:
                continue
            n = n / n_length

            # 交换线段起终点会让整条直线方程反号，但几何约束不应改变。
            if reference_normal is None:
                reference_normal = n.copy()
            elif np.dot(n, reference_normal) < 0.0:
                n = -n

            # 按长度加权
            axis_normals.append(n * length)
            
        if axis_normals:
            sum_n = np.sum(axis_normals, axis=0)
            sum_length = np.linalg.norm(sum_n)
            if np.isfinite(sum_length) and sum_length > 1e-8:
                normals[axis] = sum_n / sum_length
            
    if len(normals) < 2:
        return None # 需要至少 2 个轴
        
    # 2. 优化
    # 我们希望 R = [rx, ry, rz] 使得 rx 垂直 Nx, ry 垂直 Ny, rz 垂直 Nz
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



def solve_strict_mode_constrained(
    lines_data,
    current_f_mm,
    sensor_width_mm,
    sensor_height_mm,
    sensor_fit,
    pixel_width,
    pixel_height,
    current_rot_matrix,
    allow_focal_refine=True,
):
    f_seed = float(max(current_f_mm, 1e-6))

    if allow_focal_refine:
        refinement = refine_focal_length_for_constrained_rotation(
            lines_data,
            f_seed,
            sensor_width_mm,
            sensor_height_mm,
            sensor_fit,
            pixel_width,
            pixel_height,
            current_rot_matrix,
        )

        if refinement.get('reliable', False):
            f_mm = float(refinement.get('f_mm', f_seed))
            focal_state = 'refined'
        else:
            f_mm = f_seed
            focal_state = 'locked'
    else:
        refinement = {
            'f_mm': float(f_seed),
            'reliable': False,
            'baseline_residual': float('inf'),
            'best_residual': float('inf'),
        }
        f_mm = f_seed
        focal_state = 'fixed'

    f_pixels = get_effective_f_pixels(
        f_mm,
        sensor_width_mm,
        sensor_height_mm,
        sensor_fit,
        pixel_width,
        pixel_height,
    )
    if not np.isfinite(f_pixels) or f_pixels <= 1e-8:
        return {
            'ok': False,
            'f_mm': f_seed,
            'focal_state': focal_state,
            'refinement': refinement,
            'rot_matrix': None,
            'residual': float('inf'),
        }

    rot_matrix = solve_camera_rotation_constrained(lines_data, f_pixels, current_rot_matrix)
    if rot_matrix is None:
        return {
            'ok': False,
            'f_mm': f_mm,
            'focal_state': focal_state,
            'refinement': refinement,
            'rot_matrix': None,
            'residual': float('inf'),
        }

    residual = compute_rotation_constraint_residual(lines_data, rot_matrix, f_pixels)

    return {
        'ok': True,
        'f_mm': float(f_mm),
        'focal_state': focal_state,
        'refinement': refinement,
        'rot_matrix': rot_matrix,
        'residual': float(residual),
    }


def compute_rotation_constraint_residual(lines_data, rot_matrix, f_pixels):
    if rot_matrix is None or f_pixels is None or not np.isfinite(f_pixels) or f_pixels <= 1e-8:
        return float('inf')

    try:
        R_world_to_cam = np.array(rot_matrix).T
    except Exception:
        return float('inf')

    axis_index = {'X': 0, 'Y': 1, 'Z': 2}
    total_error = 0.0
    total_weight = 0.0

    for axis, col_idx in axis_index.items():
        axis_lines = lines_data.get(axis, [])
        if not axis_lines:
            continue

        axis_vec = np.array(R_world_to_cam[:, col_idx], dtype=float)
        axis_norm = np.linalg.norm(axis_vec)
        if axis_norm <= 1e-8:
            continue
        axis_vec /= axis_norm

        for line in axis_lines:
            a, b, c, length = line
            n = np.array([a, b, -c / float(f_pixels)], dtype=float)
            n_norm = np.linalg.norm(n)
            if n_norm <= 1e-8:
                continue
            n /= n_norm

            weight = max(float(length), 1e-6)
            err = abs(float(np.dot(axis_vec, n)))
            total_error += err * weight
            total_weight += weight

    if total_weight <= 1e-8:
        return float('inf')

    return total_error / total_weight


def refine_focal_length_for_constrained_rotation(
    lines_data,
    current_f_mm,
    sensor_width_mm,
    sensor_height_mm,
    sensor_fit,
    pixel_width,
    pixel_height,
    current_rot_matrix,
):
    active_axes = [axis for axis in ('X', 'Y', 'Z') if len(lines_data.get(axis, [])) >= 1]
    if len(active_axes) < 2:
        return {
            'f_mm': float(current_f_mm),
            'reliable': False,
            'baseline_residual': float('inf'),
            'best_residual': float('inf'),
        }

    current_f = max(float(current_f_mm), 1e-6)
    coarse_factors = np.linspace(0.35, 2.3, 24)
    fine_factors = np.linspace(0.80, 1.25, 19)
    factors = sorted({float(v) for v in np.concatenate((coarse_factors, fine_factors, np.array([1.0])))})

    candidates = []
    for factor in factors:
        f_candidate = float(np.clip(current_f * factor, 8.0, 2000.0))
        if not any(abs(f_candidate - prev) < 1e-6 for prev in candidates):
            candidates.append(f_candidate)

    if all(abs(candidate - current_f) > 1e-6 for candidate in candidates):
        candidates.append(current_f)

    scored = []
    for f_candidate in candidates:
        f_pixels = get_effective_f_pixels(
            f_candidate,
            sensor_width_mm,
            sensor_height_mm,
            sensor_fit,
            pixel_width,
            pixel_height,
        )
        if not np.isfinite(f_pixels) or f_pixels <= 1e-8:
            continue

        rot_candidate = solve_camera_rotation_constrained(lines_data, f_pixels, current_rot_matrix)
        if rot_candidate is None:
            continue

        residual = compute_rotation_constraint_residual(lines_data, rot_candidate, f_pixels)
        if not np.isfinite(residual):
            continue

        proximity_penalty = 0.008 * abs(math.log(max(f_candidate, 1e-6) / current_f))
        score = residual + proximity_penalty
        scored.append((score, residual, f_candidate, rot_candidate))

    if not scored:
        return {
            'f_mm': float(current_f_mm),
            'reliable': False,
            'baseline_residual': float('inf'),
            'best_residual': float('inf'),
        }

    scored.sort(key=lambda item: item[0])
    _best_score, best_residual, best_f_mm, _best_rot = scored[0]

    baseline_items = [item for item in scored if abs(item[2] - current_f) < 1e-6]
    if baseline_items:
        baseline_residual = baseline_items[0][1]
    else:
        baseline_residual = best_residual

    improvement = baseline_residual - best_residual
    improvement_ratio = improvement / max(abs(baseline_residual), 1e-6)
    changed = abs(best_f_mm - current_f) > max(0.1, current_f * 0.01)

    reliable = (
        changed
        and np.isfinite(baseline_residual)
        and np.isfinite(best_residual)
        and (
            best_residual <= baseline_residual * 0.97
            or improvement >= 0.003
            or improvement_ratio >= 0.02
        )
    )

    return {
        'f_mm': float(best_f_mm if reliable else current_f_mm),
        'reliable': bool(reliable),
        'baseline_residual': float(baseline_residual),
        'best_residual': float(best_residual),
    }
