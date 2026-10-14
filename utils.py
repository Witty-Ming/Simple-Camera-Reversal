import bpy
import math
import numpy as np
import mathutils
import bpy_extras
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

        # 单轴**不向 guided_lines_data 注入合成引导线**（只填 guided_axes 供显示层参考）。
        #
        # 合成引导线假设"另外两轴一水平一垂直"，该假设只在相机水平时成立；俯视/
        # 仰视时完全错误。实测真值姿态 rotX=64.9°、只画 X 轴两条线时，注入的
        # "Y 垂直 / Z 水平"会把解算拽到远离当前姿态的等价解上：相机被拉成水平
        # （up_z 从 0.9055 掉到 -0.0162），画面整体变样，用户看到的就是"单轴匹配不上"。
        #
        # 改为只约束用户真正画的那一个轴，其余两个自由度由"离当前姿态最近的解"
        # 决定——solve_camera_rotation_constrained 对没有法线的轴不做投影，SVD
        # 正交化会尽量保留种子姿态，因此结果稳定地贴近用户画线前的视角。
        one_point_guides = {
            'X': {'Y': 'VERTICAL', 'Z': 'HORIZONTAL'},
            'Y': {'X': 'HORIZONTAL', 'Z': 'VERTICAL'},
            'Z': {'X': 'HORIZONTAL', 'Y': 'VERTICAL'},
        }
        for axis in ('X', 'Y', 'Z'):
            if axis == primary_axis:
                continue
            guided_axes[axis] = one_point_guides.get(primary_axis, {}).get(axis, 'VERTICAL')

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


def compute_axis_consistency(lines_data, pixel_res_x, pixel_res_y, lines=None,
                             principal_u=0.5, principal_v=0.5):
    """检查每个轴的线段是否真能构成一组空间平行线（判断透视是否合乎物理）。

    主判据是**留一法夹角**：用除某条线段之外的其他线段求消失点，再看这条线段
    "指向该消失点"偏了多少度（= asin(垂距 / 中点到消失点的距离)）。理想情况下
    所有线段都精确指向同一个消失点，夹角为 0；手工绘制误差带来几度；而画错的
    线（例如本该沿 Y 轴却画成了沿 X 轴）会给出几十度。

    **不要用"线段到消失点的垂距"当判据**（max_dist 字段仅为向后兼容保留）。
    实测（1920x1080、三条 Y 轴长线、端点独立高斯噪声）：当两条线段在图像上
    接近平行时，它们确定的消失点落在极远处，垂距对端点噪声的敏感度按
    1/sin(夹角) 放大 —— ±8px 噪声下"好线"的垂距 p50 就有 98px、最大 1578px，
    而"第三条线实际沿 X 轴"这种明显错误也才 989px，两者完全重叠、无法区分。
    夹角判据把垂距除以"中点到消失点的距离"，正好抵消这个放大，同样条件下：
      好线 ±8px：p50 4.2 度 / p90 9.7 度 / 最大 20.7 度
      第三条偏离 20 度：76.8 度；偏离 45 度：47.9 度；偏离 90 度：42.7 度
    因此取 30 度作阈值：±8px 这种极端手抖也不会误报，而明显画错的线一定触发。

    留一法要求该轴至少 3 条线段（2 条线永远"精确"交于一点，无从判断一致性），
    不足 3 条时 max_angle 为 None，调用方应视为"无法判断"而不是"有问题"。

    返回 {axis: {'max_dist','mean_dist','max_angle','vp'} 或 None}。距离单位为
    "以主点为原点的像素"，max_angle 单位为度。None 表示该轴线段不足 2 条或
    消失点落在无穷远（线段彼此平行）。
    """
    report = {}
    image_diag = max(float(np.hypot(pixel_res_x, pixel_res_y)), 1.0)
    cx = pixel_res_x * principal_u
    cy = pixel_res_y * principal_v

    # 收集每条"会被 build_axis_line_data 接受"的线段中点（同样的 length >= 10
    # 门槛），保证与 lines_data[axis] 里的条目逐条对应 —— 否则短线段被丢弃后
    # 中点会整体错位，留一法就张冠李戴了。
    mids_by_axis = {'X': [], 'Y': [], 'Z': []}
    if lines is not None:
        for line in lines:
            axis = getattr(line, 'axis', None)
            if axis not in mids_by_axis:
                continue
            try:
                px1 = float(line.start[0]) * pixel_res_x - cx
                py1 = float(line.start[1]) * pixel_res_y - cy
                px2 = float(line.end[0]) * pixel_res_x - cx
                py2 = float(line.end[1]) * pixel_res_y - cy
            except Exception:
                continue
            if float(np.hypot(px2 - px1, py2 - py1)) < 10.0:
                continue
            mids_by_axis[axis].append(((px1 + px2) * 0.5, (py1 + py2) * 0.5))

    for axis in ('X', 'Y', 'Z'):
        data = lines_data.get(axis, [])
        if len(data) < 2:
            report[axis] = None
            continue
        try:
            arr = np.array(data, dtype=float)
        except Exception:
            report[axis] = None
            continue
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 3:
            report[axis] = None
            continue
        lines_abc = arr[:, :3]
        weights = arr[:, 3] if arr.shape[1] >= 4 else np.ones(arr.shape[0])
        if arr.shape[0] == 2:
            weights = np.ones(2)
        vp = solve_vanishing_point_2d(lines_abc, weights, image_diag=image_diag)
        if vp is None:
            report[axis] = None
            continue
        vp_h = np.array([float(vp[0]), float(vp[1]), 1.0])
        # (a, b) 在 build_axis_line_data 里已归一化，因此 |a*x + b*y + c| 就是垂距
        dists = np.abs(lines_abc @ vp_h)
        if not np.all(np.isfinite(dists)):
            report[axis] = None
            continue

        # 留一法夹角：逐条线段用"其余线段"求消失点，再量这条线段偏离该点的角度
        max_angle = None
        mids = mids_by_axis[axis]
        if lines is not None and len(lines_abc) >= 3 and len(mids) == len(lines_abc):
            worst_angle = 0.0
            measured = False
            for i in range(len(lines_abc)):
                others = np.delete(lines_abc, i, axis=0)
                vp_i = solve_vanishing_point_2d(
                    others, np.ones(len(others)), image_diag=image_diag)
                if vp_i is None:
                    continue
                vx, vy = float(vp_i[0]), float(vp_i[1])
                d_i = abs(float(lines_abc[i][0]) * vx
                          + float(lines_abc[i][1]) * vy
                          + float(lines_abc[i][2]))
                dist_i = float(np.hypot(mids[i][0] - vx, mids[i][1] - vy))
                if not np.isfinite(dist_i) or dist_i < 1e-6:
                    continue
                ratio = min(1.0, max(0.0, d_i / dist_i))
                worst_angle = max(worst_angle, float(np.degrees(np.arcsin(ratio))))
                measured = True
            if measured:
                max_angle = float(worst_angle)

        report[axis] = {
            'max_dist': float(np.max(dists)),
            'mean_dist': float(np.mean(dists)),
            'max_angle': max_angle,
            'vp': [float(vp[0]), float(vp[1])],
        }
    return report


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
    # 必须与解算层使用同一个像主点，否则 shift 非零时叠加层会与真正参与解算的
    # 地平线错开（实测 shift_y=0.05 时偏差 96px）。
    scene = getattr(context, "scene", None) if context is not None else None
    cam = getattr(scene, "camera", None) if scene is not None else None
    principal_u, principal_v = compute_principal_point_uv(scene, cam)

    _lines_data, vp_raw, _vp_adj, _axis_weights, horizon_data = solve_horizon_data(
        lines,
        pixel_res_x,
        pixel_res_y,
        True,
        cmp_data.horizon_offset_px,
        principal_u,
        principal_v,
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

        # 有 context 但三维投影失败（点落在相机背后/坐标非有限）：直接放弃这一帧，
        # 不再回退到"假设图像铺满 region"的纯 UV 映射——那条路径既不考虑
        # view_camera_offset/zoom，也会把 NaN 坐标直接送进顶点缓冲。
        return None

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


def rotate_matrix_around_point(matrix_world, rotation_matrix, pivot):
    return (
        mathutils.Matrix.Translation(pivot)
        @ rotation_matrix
        @ mathutils.Matrix.Translation(-pivot)
        @ matrix_world
    )


def compute_principal_point_uv(scene, cam):
    """
    计算真实像主点在归一化图像坐标中的位置（把 shift_x / shift_y 考虑进去）。

    做法是沿光轴向前探一个点做 world_to_camera_view —— 光轴方向的消失点就是
    主点，因此返回的 UV 即主点位置。

    这是"像素坐标 <-> 相机空间方向"换算的基准：**解算层与绘制层必须用同一个
    主点**，否则地平线叠加层会与真正参与解算的地平线错开（shift_y=0.05 时实测
    偏差可达 96px）。
    """
    if scene is None or cam is None or getattr(cam, "data", None) is None:
        return 0.5, 0.5

    try:
        cursor = scene.cursor.location
        dist = (cam.matrix_world.translation - cursor).length
        probe_dist = max(float(dist), 1.0)
        probe_world = cam.matrix_world.translation + (
            cam.matrix_world.to_quaternion() @ mathutils.Vector((0.0, 0.0, -probe_dist)))
        view = bpy_extras.object_utils.world_to_camera_view(scene, cam, probe_world)
        if np.isfinite(view.x) and np.isfinite(view.y):
            return float(view.x), float(view.y)
    except Exception:
        pass

    return 0.5, 0.5


OFFSCREEN_MARGIN_UV = 0.05


def line_is_offscreen(line, margin=OFFSCREEN_MARGIN_UV):
    """线段是否完全落在画面之外（允许 margin 的越界宽容度）。"""
    for uv in (line.start, line.end):
        u, v = float(uv[0]), float(uv[1])
        if not (np.isfinite(u) and np.isfinite(v)):
            return True
        if -margin <= u <= 1.0 + margin and -margin <= v <= 1.0 + margin:
            return False
    return True


def count_offscreen_lines(lines, margin=OFFSCREEN_MARGIN_UV):
    """统计完全落在画面外的线段数量。"""
    return sum(1 for line in lines if line_is_offscreen(line, margin))


def compensate_shift_for_target_uv(
    scene,
    cam,
    world_point,
    target_uv,
    update_view_layer=None,
    iterations=3,
    max_shift_step=0.02,
    tolerance_px=0.25,
):
    """
    用数值雅可比微调相机的 shift_x / shift_y，让 world_point 精确落在
    target_uv（归一化图像坐标）上。

    这里替代了原先散落在 properties.py 与 operators.py 里的两份重复实现：
    两者算法相同（有限差分求 2x2 雅可比 -> lstsq 解增量 -> 0.7 阻尼 + 限幅，
    只接受误差下降的步），仅迭代轮数不同，因此统一为一个可参数化的 helper。

    :param update_view_layer: 无参回调，用于在探测前后刷新 view_layer；
                              省略时退化为 scene.view_layers[0].update()。
    :return: 是否把误差压到了 tolerance_px 以内
    """
    if scene is None or cam is None or world_point is None or target_uv is None:
        return False

    render = scene.render
    pixel_res_x, pixel_res_y = get_effective_render_size(render)
    if pixel_res_x <= 1e-8 or pixel_res_y <= 1e-8:
        return False

    if update_view_layer is None:
        def update_view_layer():
            try:
                if len(scene.view_layers) > 0:
                    scene.view_layers[0].update()
            except Exception:
                pass

    shift_eps = 1e-4
    converged = False

    for _ in range(max(1, int(iterations))):
        try:
            cur_view = bpy_extras.object_utils.world_to_camera_view(scene, cam, world_point)
        except Exception:
            break
        if not (np.isfinite(cur_view.x) and np.isfinite(cur_view.y)):
            break

        err_u = float(target_uv[0]) - float(cur_view.x)
        err_v = float(target_uv[1]) - float(cur_view.y)
        err_px = float(np.hypot(err_u * pixel_res_x, err_v * pixel_res_y))
        if not np.isfinite(err_px):
            break
        if err_px < tolerance_px:
            converged = True
            break

        base_shift_x = float(cam.data.shift_x)
        base_shift_y = float(cam.data.shift_y)

        try:
            cam.data.shift_x = base_shift_x + shift_eps
            update_view_layer()
            view_sx = bpy_extras.object_utils.world_to_camera_view(scene, cam, world_point)

            cam.data.shift_x = base_shift_x
            cam.data.shift_y = base_shift_y + shift_eps
            update_view_layer()
            view_sy = bpy_extras.object_utils.world_to_camera_view(scene, cam, world_point)

            cam.data.shift_y = base_shift_y
            update_view_layer()
        except Exception:
            cam.data.shift_x = base_shift_x
            cam.data.shift_y = base_shift_y
            break

        if not (
            np.isfinite(view_sx.x) and np.isfinite(view_sx.y)
            and np.isfinite(view_sy.x) and np.isfinite(view_sy.y)
        ):
            break

        jac = np.array([
            [(float(view_sx.x) - float(cur_view.x)) / shift_eps, (float(view_sy.x) - float(cur_view.x)) / shift_eps],
            [(float(view_sx.y) - float(cur_view.y)) / shift_eps, (float(view_sy.y) - float(cur_view.y)) / shift_eps],
        ])

        if not np.all(np.isfinite(jac)):
            break

        try:
            if np.linalg.cond(jac) > 1e4:
                break
        except Exception:
            break

        delta, *_ = np.linalg.lstsq(jac, np.array([err_u, err_v]), rcond=None)
        if not np.all(np.isfinite(delta)):
            break

        dsx = float(np.clip(delta[0] * 0.7, -max_shift_step, max_shift_step))
        dsy = float(np.clip(delta[1] * 0.7, -max_shift_step, max_shift_step))

        cam.data.shift_x = base_shift_x + dsx
        cam.data.shift_y = base_shift_y + dsy
        update_view_layer()

        try:
            new_view = bpy_extras.object_utils.world_to_camera_view(scene, cam, world_point)
        except Exception:
            cam.data.shift_x = base_shift_x
            cam.data.shift_y = base_shift_y
            update_view_layer()
            break

        if not (np.isfinite(new_view.x) and np.isfinite(new_view.y)):
            cam.data.shift_x = base_shift_x
            cam.data.shift_y = base_shift_y
            update_view_layer()
            break

        new_err_u = float(target_uv[0]) - float(new_view.x)
        new_err_v = float(target_uv[1]) - float(new_view.y)
        new_err_px = float(np.hypot(new_err_u * pixel_res_x, new_err_v * pixel_res_y))
        if (not np.isfinite(new_err_px)) or new_err_px >= err_px:
            cam.data.shift_x = base_shift_x
            cam.data.shift_y = base_shift_y
            update_view_layer()
            break

    return converged


def apply_hitchcock_zoom(scene, cam, new_lens_mm):
    """
    希区柯克变焦（dolly zoom / 滑动变焦）：
    改变焦距的同时让相机沿自身光轴移动，使 3D 游标所在深度平面的成像
    大小与位置完全不变 —— 主体构图纹丝不动，只有背景透视被压缩或扩张。

    推导：相机空间里深度 d 处的点满足 u = 0.5 + (x/d)·(f_px/W)。
    只要让焦距与深度按同一比例缩放（d' = d · f'/f），所有位于深度 d 的
    平面上的点其 UV 都不变。注意这里用的是"游标深度(-z_cam)"而不是
    相机到游标的欧氏距离：游标偏离光轴时，只有按深度缩放才能让 UV 严格
    保持不变（欧氏距离会让游标在画面上漂移）。

    :return: (ok, reason) —— reason ∈ {'ok','unchanged','degenerate','invalid'}
    """
    if scene is None or cam is None or getattr(cam, "data", None) is None:
        return False, 'invalid'

    f_old = float(cam.data.lens)
    try:
        f_new = float(new_lens_mm)
    except (TypeError, ValueError):
        return False, 'invalid'

    if not np.isfinite(f_new) or not np.isfinite(f_old) or f_old <= 1e-8:
        return False, 'invalid'

    f_new = float(min(max(f_new, 1.0), 10000.0))
    if abs(f_new - f_old) < 1e-9:
        return True, 'unchanged'

    mw = cam.matrix_world.copy()
    rot = mw.to_3x3()
    origin = mw.translation.copy()
    cursor = scene.cursor.location.copy()

    # 游标在相机空间的位置（Blender 相机看向自身 -Z）
    p_cam = rot.transposed() @ (cursor - origin)
    depth_old = -float(p_cam.z)

    if not np.isfinite(depth_old) or depth_old <= 1e-6:
        # 游标在相机后方/与相机重合：无法做 dolly，退化为单纯改焦距
        cam.data.lens = f_new
        return True, 'degenerate'

    scale = f_new / f_old
    delta = depth_old * (scale - 1.0)          # 沿光轴（相机背后为正）的位移
    offset_world = rot @ mathutils.Vector((0.0, 0.0, delta))

    cam.matrix_world = mathutils.Matrix.Translation(offset_world) @ mw
    cam.data.lens = f_new
    return True, 'ok'


def register_class_safe(cls):
    try:
        bpy.utils.register_class(cls)
    except ValueError:
        # 已经注册过：先注销再注册，确保拿到的是最新实现。
        # 这一步的失败必须显式打印，否则紧接着的 register_class 会抛出
        # 让人看不懂的 "already registered"，或者更糟 —— 旧类残留，
        # UI 里出现重复面板、旧操作符覆盖新实现。
        try:
            bpy.utils.unregister_class(cls)
        except Exception as e:
            print(f"[CameraMatch] 重新注册前注销类 {getattr(cls, '__name__', cls)} 失败: {e}")
        bpy.utils.register_class(cls)


def unregister_class_safe(cls):
    try:
        bpy.utils.unregister_class(cls)
    except Exception as e:
        # 静默吞掉会让"类没注销成功"变成无法诊断的问题。
        print(f"[CameraMatch] 注销类 {getattr(cls, '__name__', cls)} 失败: {e}")


def get_ordered_frame_points(context):
    """
    返回相机视图边框的 (TR, TL, BL, BR) 四个角点（相机局部空间）。

    主路径按 x/y 符号分类，与 view_frame() 的返回顺序无关。当某个角的 x 或 y
    恰好为 0 时符号分类会失败（shift_x == ±0.5 时可精确出现），此时改用极值
    重建四角 —— 原先的 `frame[0], frame[1], frame[3], frame[2]` 隐含假设
    view_frame() 返回 [TR, TL, BR, BL]，而 Blender 官方实现
    （bpy_extras.world_to_camera_view 用 frame[1].x 作 max_x、frame[1].y 作
    min_y）可证 frame[1] 必然是右下角，该假设是错的，会把四角角色整体错位。
    """
    cam = context.scene.camera
    if not cam:
        return None, None, None, None

    try:
        frame = cam.data.view_frame(scene=context.scene)
    except Exception:
        return None, None, None, None

    if not frame:
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

    if all([TR, TL, BL, BR]):
        return TR, TL, BL, BR

    # 退化路径：不依赖 view_frame() 的返回顺序，直接用极值重建
    try:
        xs = [float(v.x) for v in frame]
        ys = [float(v.y) for v in frame]
        z = float(frame[0].z)
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        make = lambda x, y: mathutils.Vector((x, y, z))
        return (make(max_x, max_y), make(min_x, max_y),
                make(min_x, min_y), make(max_x, min_y))
    except Exception:
        return None, None, None, None




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
    if vp_final is None:
        vp_final = vp

    # 与上面 n == 2 的分支保持一致：消失点远到 image_diag * 5 以外，说明这些线
    # 其实近似平行，IRLS 只是把端点噪声"外推"成了一个极远的假交点。实测三条
    # 近似平行的竖线能给出 1e5 像素以外的交点，而它随后会以"合法消失点"的身份
    # 参与焦距解算，把焦距推向荒谬值。所以 n >= 3 也必须做同样的距离检查 ——
    # 此前只有 n == 2 分支有，导致"三条近似平行线"反而比"两条"处置更宽松。
    if np.linalg.norm(vp_final) > image_diag * 5:
        return None

    return vp_final
    
# ... (omitted)

def orthonormalize_matrix(R):
    """
    把 3x3 矩阵正交化（保持右手系 det=+1）。

    这里保留 SVD 实现：实测（Blender 5.2.2 / numpy）3x3 的 np.linalg.svd 是
    LAPACK 单次调用（约 25 μs），而手写 Gram-Schmidt 需要 8~10 次小型 numpy
    调用（约 59 μs），反而慢 2.4 倍，所以不做替换。
    """
    U, S, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt
    if np.linalg.det(R_ortho) < 0:
       U[:, 2] *= -1
       R_ortho = U @ Vt
    return R_ortho


def select_axis_signs(vx, vy, vz, current_rot_matrix=None):
    """
    消除"世界轴在相机坐标系下的方向向量"的符号歧义。

    消失点反演得到的轴向 (vx, vy, vz) 各自存在 ± 二义性（平行线本身无法
    区分轴线正反方向，尤其是轴线指向相机后方时反演结果必然取反）。这里
    枚举所有 det=+1 的符号组合，再借助当前相机旋转选择与现有机位最接近的
    组合，从而避免"世界 Z 朝上"一刀切启发式把俯视/偏航相机错误扭转。

    返回: 世界->相机 旋转矩阵 (numpy 3x3)
    """
    base = np.column_stack((vx, vy, vz))
    d0 = float(np.linalg.det(base))
    if not np.isfinite(d0) or abs(d0) < 1e-12:
        # 退化帧，直接正交化
        return orthonormalize_matrix(base)
    sign = 1.0 if d0 > 0 else -1.0

    combos = []
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            for sz in (1.0, -1.0):
                if sx * sy * sz != sign:
                    continue
                combos.append(orthonormalize_matrix(
                    np.column_stack((sx * vx, sy * vy, sz * vz))))

    if current_rot_matrix is not None:
        R_seed = np.array(current_rot_matrix).T  # Cam->World -> World->Cam
        scored = [(float(np.sum(M * R_seed)), M) for M in combos]
        best_score = max(s for s, _ in scored)
        # 得分接近（例如对称机位）时优先选择世界 Z 朝上的组合
        zup = [M for s, M in scored if abs(s - best_score) < 1e-6 and M[1, 2] > 0]
        if zup:
            return zup[0]
        return max(scored, key=lambda t: t[0])[1]

    zup = [M for M in combos if M[1, 2] > 0]
    return zup[0] if zup else combos[0]


def get_effective_f_pixels(f_mm, sensor_width_mm, sensor_height_mm, sensor_fit, pixel_width, pixel_height):
    """
    稳健地计算以像素为单位的焦距，正确处理 'AUTO' 传感器适配。

    AUTO 规则（实测 Blender 5.2.2，用 world_to_camera_view 反推真实像素焦距）：
        **始终用 sensor_width 去适配图像较长的那条边**，与 sensor_height、
        与传感器的宽高比都无关。判定"长边"用的是已经含 pixel_aspect 的
        有效像素尺寸（本函数收到的 pixel_width/pixel_height 正是
        get_effective_render_size 的输出，已含 pixel_aspect 与分辨率百分比）。

        36x24 传感器 / 50mm 实测：
            1920x1080 -> f_px = 50/36*1920 = 2666.67   (sensor_width 配宽)
            1080x1920 -> f_px = 50/36*1920 = 2666.67   (sensor_width 配高)
            1000x1000 -> f_px = 50/36*1000 = 1388.89   (sensor_width 配宽)

    修正前竖构图会误用 sensor_height，导致 f_px 偏差 sensor_width/sensor_height
    倍（典型 1.5 倍），竖幅照片解算出的焦距与位置会整体错位。
    """
    if sensor_fit == 'VERTICAL':
        if sensor_height_mm <= 1e-8:
            return float('nan')
        return (f_mm / sensor_height_mm) * pixel_height

    if sensor_fit == 'HORIZONTAL':
        if sensor_width_mm <= 1e-8:
            return float('nan')
        return (f_mm / sensor_width_mm) * pixel_width

    # AUTO: sensor_width 适配图像较长边
    if sensor_width_mm <= 1e-8:
        return float('nan')
    if pixel_width >= pixel_height:
        return (f_mm / sensor_width_mm) * pixel_width
    return (f_mm / sensor_width_mm) * pixel_height


def get_effective_f_mm_from_pixels(f_pixels, sensor_width_mm, sensor_height_mm, sensor_fit, pixel_width, pixel_height):
    """get_effective_f_pixels 的逆运算（AUTO 规则见上）。"""
    if f_pixels is None or not np.isfinite(f_pixels) or f_pixels <= 1e-8:
        return None

    if sensor_fit == 'VERTICAL':
        if pixel_height <= 1e-8 or sensor_height_mm <= 1e-8:
            return None
        return float((f_pixels / pixel_height) * sensor_height_mm)

    if sensor_fit == 'HORIZONTAL':
        if pixel_width <= 1e-8 or sensor_width_mm <= 1e-8:
            return None
        return float((f_pixels / pixel_width) * sensor_width_mm)

    # AUTO
    if sensor_width_mm <= 1e-8:
        return None
    if pixel_width >= pixel_height:
        if pixel_width <= 1e-8:
            return None
        return float((f_pixels / pixel_width) * sensor_width_mm)
    if pixel_height <= 1e-8:
        return None
    return float((f_pixels / pixel_height) * sensor_width_mm)



def calculate_camera_transform(vp_data, sensor_width_mm, sensor_height_mm, sensor_fit, pixel_width, pixel_height, current_dist, default_f_mm=50.0, axis_weights=None, anchor_location=None, anchor_screen_offset=None, current_rot_matrix=None):
    """
    vp_data: {'X':(u,v), ...} 以中心像素为单位
    axis_weights: {'X': count, ...} 各轴的线段数量权重
    current_rot_matrix: 当前相机旋转 (Cam->World, 3x3)，用于消除轴线方向符号歧义
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
        if d < 0:
            f_px = float(np.sqrt(-d))
            # 两个消失点几乎正交时 d ≈ 0⁻，sqrt(-d) 会给出接近 0 的像素焦距，
            # 而它随后会作为除数出现在射线构造里（ray = [px, py, -f_px] 归一化、
            # 位置 = ray * dist 等），直接把位置炸成 inf/nan。这个值在物理上
            # 也不可能（f_px → 0 对应无穷视场角），判为"这对消失点给不出焦距"。
            if not np.isfinite(f_px) or f_px < 1e-6:
                return None
            return f_px
        return None
    
    def validate_focal_length(f_pixels, pixel_width, pixel_height):
        """
        只做"物理上是否可能"的范围校验。
        返回 bool。

        注意：这里**不再**依据"解算值与当前焦距的差距"给置信度 —— 那个做法
        会让解算结果被强行拉回用户猜测的初值，而相机反求的目的恰恰是求出
        未知的焦距。可信度改由"各消失点对焦距估计的一致程度"决定（见下）。
        """
        if f_pixels is None or not np.isfinite(f_pixels) or f_pixels <= 0:
            return False

        # 范围必须按"较长边"定标，而不是固定用 pixel_height —— AUTO 传感器适配
        # 就是用较长边把毫米焦距换算成像素焦距的（见 get_effective_f_pixels），
        # 所以同一个镜头在横竖构图下 pixel_height 会差一倍，拿它当尺子会让横构图
        # 的合法区间整体缩水一半（1920x1080 下 0.3*pixel_height 相当于 6.1mm，
        # 而竖构图 1080x1920 下同样的镜头得到的是 10.8mm 的下限）。
        #
        # 系数含义：f_px / longer = f_mm / sensor_width，取 sensor_width = 36mm，
        # 于是 0.2 ~ 12 对应约 7mm ~ 430mm 的镜头，覆盖从超广角到长焦的常用区间，
        # 又足以拦住 sqrt(-d) 在消失点近正交时冒出来的荒谬小值。
        longer = max(float(pixel_width), float(pixel_height))
        min_f = longer * 0.2
        max_f = longer * 12.0
        return min_f <= f_pixels <= max_f

    def focal_consistency_confidence(f_std, f_mean):
        """
        由"多组正交消失点给出的焦距估计有多一致"决定可信度。

        一致性高说明线条质量好、解算自洽，此时应当完全相信解算结果；
        一致性差才需要向默认焦距做正则化。这样即使解算值与初始值相差
        数倍（例如真实 35mm、初始 135mm），也不会被错误地拉回初值。
        """
        if f_mean is None or not np.isfinite(f_mean) or f_mean <= 1e-9:
            return 0.2
        ratio = float(f_std) / float(f_mean)
        if not np.isfinite(ratio):
            return 0.2
        if ratio <= 0.03:
            return 1.0
        if ratio <= 0.08:
            return 0.8
        if ratio <= 0.15:
            return 0.6
        if ratio <= 0.30:
            return 0.4
        return 0.2

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
        # 权重是第一优先级：线段越多的轴对越可信。
        max_weight = max(x[2] for x in f_candidates_info)

        # 筛选出具有最大权重的候选项
        best_candidates = [x for x in f_candidates_info if x[2] == max_weight]

        f_vals = [x[0] for x in best_candidates]
        f_mean = float(np.mean(f_vals))
        f_std = float(np.std(f_vals))

        if len(best_candidates) == 1:
            # 只有一组正交对可用
            f_pixels = best_candidates[0][0]
            trusted_axes = best_candidates[0][1]
        elif f_std < f_mean * 0.1:
            # 各组高度一致 —— 取平均并信任所有参与组合
            f_pixels = f_mean
            trusted_axes = set()
            for x in best_candidates:
                trusted_axes.update(x[1])
        else:
            # 各组分歧较大：选最接近默认焦距的那个（用户通常大致知道焦段）
            best_sub_choice = min(best_candidates, key=lambda c: abs(c[0] - default_f_pixels))
            f_pixels = best_sub_choice[0]
            trusted_axes = best_sub_choice[1]

        # 物理范围校验（与"和初始值差多少"无关）
        if not validate_focal_length(f_pixels, pixel_width, pixel_height):
            f_pixels = default_f_pixels
            trusted_axes = set(vp_data_shifted.keys())

        # 只有在"各组消失点给出的焦距互相矛盾"时才做正则化，
        # 且强度受限（最多 40%），避免把解算结果拉回用户猜测的初值。
        confidence = focal_consistency_confidence(f_std, f_mean)
        if confidence < 0.6:
            blend_factor = min((0.6 - confidence) * 0.5, 0.4)
            f_pixels = f_pixels * (1.0 - blend_factor) + default_f_pixels * blend_factor
        
        # f_pixels -> f_mm：统一走 get_effective_f_mm_from_pixels，
        # 避免此处再维护一份（曾经写错的）AUTO 适配分支。
        val_mm = get_effective_f_mm_from_pixels(
            f_pixels,
            sensor_width_mm,
            sensor_height_mm,
            sensor_fit,
            pixel_width,
            pixel_height,
        )

        # 更严格的焦距范围检查
        if val_mm is not None and 10.0 < val_mm < 2000.0:
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
    # 符号消歧：枚举 det=+1 的符号组合，借助当前相机旋转选择最接近的组合，
    # 替代原来"世界 Z 朝上则翻转"的错误启发式（俯视相机会被错误扭转 2×pitch）。
    R_w2c = select_axis_signs(rx, ry, rz, current_rot_matrix=current_rot_matrix)
    rot_matrix = mathutils.Matrix(R_w2c.T)
    
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

def solve_camera_rotation_constrained(lines_data, f_pixels, current_rot_matrix, iterations=20):
    """
    使用单线（平面）和固定焦距求解旋转。
    lines_data: {'X': [[a,b,c,len],...], 'Y':...}
    f_pixels: 当前像素焦距
    current_rot_matrix: 3x3 mathutils 矩阵 (世界到相机? 不，相机方向)
                        Blender 相机矩阵: Col 0=右, Col 1=上, Col 2=后.
    iterations: 迭代投影的迭代次数
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
            
    # 至少 1 个轴即可解算。单轴（单点透视）时只有该轴被投影约束，其余两列保持
    # 种子方向、仅由末尾的 SVD 正交化做最小修正，因此解稳定地落在"离当前姿态
    # 最近"的那个解上，不会漂到画面完全变样的等价解（见 build_perspective_mode_constraints
    # 里 ONE_POINT 分支的说明）。
    if len(normals) < 1:
        return None # 一个轴都没有时无从约束
        
    # 2. 优化
    # 我们希望 R = [rx, ry, rz] 使得 rx 垂直 Nx, ry 垂直 Ny, rz 垂直 Nz
    # 使用当前的 R_world_to_cam 初始化
    R = np.array(current_rot_matrix).T # current_rot_matrix 是 Cam->World。转置 -> World->Cam。
    R_seed = R.copy()
    # 确保正交仅防万一
    R = orthonormalize_matrix(R) 

    # 单轴（单点透视）走解析解，不做下面的交替投影迭代。
    #
    # 为什么：迭代体每轮只把被约束的那一列投影到约束平面，其余两列原样带入
    # orthonormalize_matrix 做 SVD 正交化；而 SVD 找的是"与 [u_proj, v0, w0] 的
    # Frobenius 距离最小的旋转矩阵"，为了让三列重新正交，它会顺带把未被约束的
    # v、w 也转走一点 —— 每轮转一点，20 轮累积下来姿态可以偏离种子很远。
    # 实测用户工程（I:\空净\T90\首图2.blend，4096x4096）只画 3 条 X 轴线段时：
    # 相机 up_z 从 0.96945 被转到 0.24294（约 70 度），而正确解应当停在 0.9686
    # 附近。这个被转歪的姿态又会成为下一次解算的种子，于是用户每画一条线画面就
    # 歪一次，后续即使补上 Y 轴线段也从坏种子出发收敛到别的局部极小 ——
    # 正是用户报告的"单轴、多轴都会出现匹配不上"的来源。
    #
    # 最小旋转的做法：把种子里被约束的那一列投影到约束平面，再用"把该列转到
    # 投影方向"的那一次最小旋转作用于整个种子姿态。这样约束被严格满足
    # （R[:, idx] 恰为投影方向），另外两列只跟着这一次必要的旋转走，不引入任何
    # 多余转动。投影退化为零向量（视线与法线平行、该轴没有可见的消失点方向）时
    # 保持种子不变。
    if len(normals) == 1:
        axis_name = next(iter(normals))
        n_single = normals[axis_name]
        idx = {'X': 0, 'Y': 1, 'Z': 2}[axis_name]
        base_col = R_seed[:, idx].copy()
        proj = base_col - np.dot(base_col, n_single) * n_single
        proj_norm = float(np.linalg.norm(proj))
        if np.isfinite(proj_norm) and proj_norm > 1e-6:
            proj = proj / proj_norm
            cross = np.cross(base_col, proj)
            sin_a = float(np.linalg.norm(cross))
            cos_a = float(np.dot(base_col, proj))
            if sin_a > 1e-9:
                kx = np.array([[0.0, -cross[2], cross[1]],
                               [cross[2], 0.0, -cross[0]],
                               [-cross[1], cross[0], 0.0]])
                rot_min = np.eye(3) + kx + kx @ kx * ((1.0 - cos_a) / (sin_a * sin_a))
                R = orthonormalize_matrix(rot_min @ R_seed)
        else:
            R = R_seed.copy()
    else:
        # 迭代投影（两个及以上轴时约束足够，交替投影收敛到唯一解）
        for i in range(iterations):
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
        
    # 3. 离散二义性消除。
    #    迭代投影的残差对 R 与 R_flip（世界 Y/Z 两列同时取反，等价于绕视轴转 180°）
    #    完全相同 —— 翻转不改变任何轴的约束方向，所以残差根本无法区分二者，
    #    符号只能由种子（迭代起点）决定。
    #    旧实现只在两者中选"与种子更接近"的那个；于是当种子本身是倒的
    #    （用户手动把相机转倒、上一次解算遗留、相机初始姿态很怪）时，
    #    解出来的相机就是倒的 —— 这正是用户报告的"绘制后 Z 轴反了、产品整个倒过来"。
    #    实测（合成场景 5 个视角，th = 23/38/84/101/125 度）：真值 up_z 全部为正
    #    （0.3907 / 0.6157 / 0.9945 / 0.9816 / 0.8192）；而用 Matrix.Identity(3)
    #    当种子时 top_down 给出 up_z = -0.0114、bottom_up 给出 0.033、
    #    steep 给出 0.1021 —— 坏种子确实会把相机翻倒。
    #    因此改为：只要二者符号不同，一律取 up_z >= 0 的那个（相机不会倒置拍摄，
    #    这是稳定的物理先验；俯拍 th=23 度与仰拍 th=125 度时 up_z 依然为正，不受影响）；
    #    只有二者同号（含 up_z ≈ 0 的水平视线情形）才回退到"与种子更接近"。
    #    up_z = R[1, 2]：R 是 World->Cam，其转置的第 1 列即相机 up 在世界中的方向，
    #    该向量的世界 Z 分量就是转置矩阵的 [2, 1] 元素，也就是 R 的 [1, 2] 元素。
    R_flip = R.copy()
    R_flip[:, 1] = -R_flip[:, 1]
    R_flip[:, 2] = -R_flip[:, 2]
    up_z = float(R[1, 2])
    flip_up_z = float(R_flip[1, 2])
    if up_z < 0.0 <= flip_up_z:
        R = R_flip
    elif up_z >= 0.0 > flip_up_z:
        pass
    elif float(np.sum(R_flip * R_seed)) > float(np.sum(R * R_seed)):
        R = R_flip

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
    fast=False,
    smooth_ref_mm=None,
    extra_seeds=None,
    multistart_random=0,
):
    """
    :param smooth_ref_mm: 焦距正则化的参考值（跨帧时间平滑基准）。实时解算时
        传入相机**当前**焦距，让"向参考值正则化"真正起到平滑作用；为 None 时
        退化为以扫描中心 current_f_mm 为基准。扫描中心始终是 current_f_mm，
        因此参考值偏离真值也不会让扫描错过解。
    :param extra_seeds: 除 current_rot_matrix 之外的额外旋转起点（例如消失点法
        给出的姿态）。交替投影只做局部收敛，起点越多越不容易卡在错误的局部极小。
    :param multistart_random: 额外生成多少个随机姿态起点。0 表示关闭（只精化
        给定的起点），非零时由调用方按 fast/exact 决定数量以控制开销。
    """
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
            fast=fast,
            smooth_ref_mm=smooth_ref_mm,
            extra_seeds=extra_seeds,
            multistart_random=(max(2, int(multistart_random) // 4) if multistart_random else 0),
        )

        if refinement.get('reliable', False):
            f_mm = float(refinement.get('f_mm', f_seed))
            focal_state = 'refined'
        else:
            # 焦距锁定：采用 refine 给出的保守值（相机当前焦距），而不是消失点给的
            # f_seed。f_seed 在噪声下可能离谱到 +131%，"锁定"却锁在错误值上 ——
            # 实测 bottom_up/YZ/n8/s2 因此把焦距误差从 42.9%（保持相机初值）恶化到
            # 82.9%。refine 在不可靠时会返回 smooth_ref（调用方未提供时才是 f_seed）。
            f_mm = float(refinement.get('f_mm', f_seed))
            # 再细分：refine 报不可靠，但残差比较显示它采用的正是"消失点给出的
            # 焦距"（state='consistent'）时，焦距其实已经正确、无需改变，不应报
            # "约束不足/已锁定"。只有真正退回相机当前焦距时才报 locked。
            focal_state = 'consistent' if refinement.get('state') == 'consistent' else 'locked'
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

    # 多起点求解：交替投影只做局部收敛 —— 实测从"相机当前姿态"出发会卡在残差
    # 0.1441 的局部极小（把相机解成仰视 29.6 度、整体歪掉），而多起点能找到
    # 0.0238 的解（俯视 12.4 度、up_z=0.977）。用户看到的"画面是歪的""Z 轴
    # 不指向空间方向"都源于姿态卡在错误的局部极小上，因此这里必须多起点。
    # 但**单轴时必须关掉多起点**：只有一个轴向有线段时，约束只剩"把该轴列投影到
    # 法线平面"，精化后任何起点的残差都恰好是 0，而 solve_rotation_multistart 里
    # 用 `residual < best_residual` 严格比较，于是取粗筛排第一的那个。粗筛排第一的
    # 是"种子自身残差最小"的**随机**姿态，它另外两个自由度完全随机 —— 实测用户
    # 工程只画 1 条 X 线时 up_z 从 0.96945 掉到 0.24294（相机被凭空转了 60 多度）、
    # 画到 3 条 X 线时是 0.23827；且 cold（每步重置相机）与 warm 结果完全相同，
    # 说明与状态累积无关，就是多起点选错了起点。单轴唯一合理解是"离当前姿态最近
    # 的那个"（最小旋转解析解），所以只精化当前姿态、不生成随机起点。
    active_axis_count = sum(1 for axis in ('X', 'Y', 'Z') if lines_data.get(axis))
    final_seeds = [np.array(current_rot_matrix, dtype=float)]
    if active_axis_count <= 1:
        random_seed_count = 0
        keep_seeds = 1
    else:
        for seed in (extra_seeds or []):
            if seed is None:
                continue
            try:
                final_seeds.append(np.array(seed, dtype=float))
            except Exception:
                continue
        random_seed_count = int(multistart_random)
        keep_seeds = 2 if fast else 4

    rot_matrix, _final_residual = solve_rotation_multistart(
        lines_data,
        f_pixels,
        final_seeds,
        iterations=20,
        extra_random=random_seed_count,
        keep=keep_seeds,
    )
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


# 非锚点起点（消失点法姿态、随机姿态）的解被采纳所需的"相对锚点残差改善"门槛。
#
# 多起点是"盲搜"：候选起点先用自身残差粗筛，再迭代精化，最后取残差最小者。
# 在端点噪声下这会过拟合 —— 别的起点可能恰好让"消失点落在各自线段上"这个残差
# 略小，方向却完全错误。实测俯视 65 度、每轴 3 条线、±8px 端点噪声：
#   · 只精化锚点（相机当前姿态）：残差 0.0664、整体旋转误差 6.1 度（正确）
#   · 加入消失点法姿态后：残差 0.0523（仅改善 21%）、旋转误差 82.2 度（错误）
#   · 加入 24 个随机起点后：残差 0.0724、旋转误差 98.2 度（错误）
# 反过来，真正卡在错误局部极小的场景改善极其显著：用户工程
# I:\空净\T90\首图2.blend 从锚点出发残差 0.1441，多起点找到 0.0238（改善 83%）。
# 取 0.4 作为门槛可把"改善 20% 的假解"与"改善 80% 的真解"干净分开。
#
# 注意：门槛只对**非锚点**起点生效。锚点是调用方给的"相机当前姿态"，它本身
# 永远是合法候选（用户手动摆好的视角不该被盲搜随手推翻）。
MULTISTART_ADOPT_MARGIN = 0.4


def solve_rotation_multistart(
    lines_data,
    f_pixels,
    seeds=None,
    iterations=20,
    extra_random=0,
    keep=4,
    rng=None,
):
    """多起点求解相机旋转，返回约束残差最小的解。

    solve_camera_rotation_constrained 是交替投影迭代，**只做局部收敛**：同一组
    线条从不同起点出发会落到不同的不动点，残差能差好几倍。实测用户的实际工程
    （I:\\空净\\T90\\首图2.blend，4096x4096，X/Y/Z 各 3 条线）：从"相机当前姿态"
    出发残差 0.1441（等效 8.3 度角度误差，相机被解成**仰视 29.6 度**），而从其他
    起点出发能找到 0.0238（1.37 度，**俯视 12.4 度**）—— 相差 6 倍。姿态错了整个
    画面的方向就都错了（用户看到的"Z 轴不指向空间方向""画面是歪的"），所以必须
    做多起点，不能只从当前姿态出发。

    做法：候选起点先用"种子自身的残差"粗筛（不迭代，非常便宜），只对最优的
    keep 个做迭代精化，最后取残差最小的解。这样既覆盖了残差的多峰结构，又不会
    让开销随起点数线性增长。

    :param seeds: 候选起点（Cam->World 3x3，mathutils.Matrix 或 numpy 数组均可）
    :param extra_random: 额外生成多少个随机姿态起点（用固定 rng 保证可复现）
    :param keep: 粗筛后真正做迭代精化的起点个数
    :param rng: 可选的 numpy Generator；为 None 时用固定种子新建，保证可复现
    """
    candidates = []
    for seed in (seeds or []):
        if seed is None:
            continue
        try:
            m = np.array(seed, dtype=float)
        except Exception:
            continue
        if m.shape == (3, 3) and np.all(np.isfinite(m)):
            candidates.append(m)

    # 显式种子（调用方给的：通常是"相机当前姿态"，可能还有消失点法姿态）与
    # 随机种子要分开记账 —— 前者可信，后者只是盲搜，采纳门槛不同。
    explicit_count = len(candidates)

    if extra_random > 0:
        rng_local = rng if rng is not None else np.random.default_rng(20260214)
        for _ in range(int(extra_random)):
            q = rng_local.normal(size=4)
            q_norm = float(np.linalg.norm(q))
            if not np.isfinite(q_norm) or q_norm <= 1e-9:
                continue
            qw, qx, qy, qz = (q / q_norm).tolist()
            candidates.append(np.array([
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
            ]))

    if not candidates:
        return None, float('inf')

    scored = []
    for index, candidate in enumerate(candidates):
        try:
            seed_residual = compute_rotation_constraint_residual(
                lines_data, mathutils.Matrix(candidate), f_pixels)
        except Exception:
            continue
        if np.isfinite(seed_residual):
            scored.append((float(seed_residual), index, candidate))

    if not scored:
        return None, float('inf')

    scored.sort(key=lambda item: item[0])
    keep_n = max(1, min(int(keep), len(scored)))
    chosen = list(scored[:keep_n])

    # 锚点（第一个显式种子）必须参与精化：它是盲搜解的"参照基准"，如果粗筛把它
    # 挤掉，下面的采纳门槛就失效了（anchor_residual 停在 inf）。
    if explicit_count > 0 and not any(item[1] == 0 for item in chosen):
        anchor_entry = next((item for item in scored if item[1] == 0), None)
        if anchor_entry is not None:
            chosen.append(anchor_entry)

    # 锚点（第一个显式种子，即调用方给的"相机当前姿态"）必须**先**精化：它的结果
    # 是下面所有非锚点起点的采纳基准。若把它和其他起点混在同一个循环里，处理到
    # 非锚点起点时 anchor_residual 可能还是 inf，门槛就形同虚设 —— 实测过这个坑：
    # 粗筛把锚点挤出前 keep 名、锚点被追加到 chosen 末尾，于是 hybrid 姿态（残差
    # 0.0523、旋转误差 82.2 度）在基准就绪前就被采纳了。
    anchor_rot = None
    anchor_residual = float('inf')
    anchor_entry = next((item for item in scored if item[1] == 0), None)
    if anchor_entry is not None:
        rot = solve_camera_rotation_constrained(
            lines_data, f_pixels, mathutils.Matrix(anchor_entry[2]), iterations=iterations)
        if rot is not None:
            residual = compute_rotation_constraint_residual(lines_data, rot, f_pixels)
            if np.isfinite(residual):
                anchor_rot = rot
                anchor_residual = float(residual)

    best_rot = anchor_rot
    best_residual = anchor_residual
    for _seed_residual, index, candidate in chosen:
        if index == 0:
            continue
        rot = solve_camera_rotation_constrained(
            lines_data, f_pixels, mathutils.Matrix(candidate), iterations=iterations)
        if rot is None:
            continue
        residual = compute_rotation_constraint_residual(lines_data, rot, f_pixels)
        if not np.isfinite(residual):
            continue
        # 非锚点起点的解，只有相对锚点改善超过 MULTISTART_ADOPT_MARGIN 才采纳，
        # 见该常量的注释。
        if np.isfinite(anchor_residual) and residual > anchor_residual * (1.0 - MULTISTART_ADOPT_MARGIN):
            continue
        if residual < best_residual:
            best_residual = float(residual)
            best_rot = rot

    return best_rot, best_residual


def refine_focal_length_for_constrained_rotation(
    lines_data,
    current_f_mm,
    sensor_width_mm,
    sensor_height_mm,
    sensor_fit,
    pixel_width,
    pixel_height,
    current_rot_matrix,
    fast=False,
    smooth_ref_mm=None,
    extra_seeds=None,
    multistart_random=0,
):
    active_axes = [axis for axis in ('X', 'Y', 'Z') if len(lines_data.get(axis, [])) >= 1]
    if len(active_axes) < 2:
        return {
            'f_mm': float(current_f_mm),
            'reliable': False,
            'baseline_residual': float('inf'),
            'best_residual': float('inf'),
        }

    # 焦距可辨识性门槛：每个消失点至少要 2 条同轴线段才能定义，而焦距由两个
    # 正交轴的消失点共同决定（f² = -V1·V2）。所以"能出消失点的轴"少于 2 个时，
    # 焦距在数学上完全不可辨识，此时**必须完全不碰焦距**。
    #
    # 实测（用户工程 I:\空净\IN19\4.blend：X 1 条 + Y 3 条 + Z 1 条，只有 Y 轴
    # 能出消失点）：固定姿态下扫描焦距 10/20/30/40/50/60/80/100/150/200/257/
    # 350/500/800/1500mm，残差为 0.1802/0.1112/0.0829/0.0678/0.0586/0.0523/
    # 0.0444/0.0397/0.0333/0.0301/0.0279/0.0260/0.0245/0.0231/0.0220 —— 单调
    # 下降、**没有谷底**，焦距越大残差越小（每换一次姿态重解也一样单调）。
    # 任何"残差有改善"的判据在这种曲线上都必然把焦距推向扫描上限；而扫描范围
    # 是 current_f × geomspace(0.15, 3.0)，上限随 current_f 一起涨，逐帧累积就是
    # 用户看到的"焦距无限变大"（实时档 6%/帧限幅下 28Hz 一秒能涨 5 倍）。一旦
    # 最优落回扫描边界，at_edge 又把它判成不可靠而锁死 —— 于是出现"涨一阵、
    # 卡一阵"，也就是用户说的"数值很大时锁死，要调很久才变化"。
    #
    # 正确行为：只有 1 个轴可定义消失点时，焦距交给用户/相机原值，插件只用约束
    # 法解姿态（约束法对焦距误差几乎免疫：实测传 13.71mm / 109.99mm 这种完全
    # 错误的焦距，姿态误差也只到 2.47°）。
    vp_capable_axes = [axis for axis in ('X', 'Y', 'Z') if len(lines_data.get(axis, [])) >= 2]
    if len(vp_capable_axes) < 2:
        return {
            'f_mm': float(current_f_mm),
            'reliable': False,
            'baseline_residual': float('inf'),
            'best_residual': float('inf'),
            'identifiable': False,
        }

    current_f = max(float(current_f_mm), 1e-6)

    # 多起点种子池：在函数入口生成一次，所有焦距候选复用。
    #
    # 种子是"姿态"，与焦距无关（只有粗筛排序会随 f_pixels 变化），所以池子可以
    # 共享。反过来，若每个 score_candidate 各自临时生成随机种子，就会陷入两难：
    # 生成多了总开销乘以几十次调用、生成少了又找不到真解 —— 实测只给 6 个随机
    # 种子时，真值焦距处的残差都降不到 0，于是 strong_evidence 判据
    # （最优残差 < 基线一半）不触发、"最优焦距"落回基线，焦距完全不动
    # （无噪声、真值 35mm、初值 50mm 时 5/5 例误差 42.86%）。
    #
    # 池子大小按 fast/exact 分档，直接乘进总开销（score_candidate 会被调用约
    # 40 次），因此精确档 24 个、实时档 8 个。
    seed_pool = []
    try:
        seed_pool.append(np.array(current_rot_matrix, dtype=float))
    except Exception:
        pass
    for extra_seed in (extra_seeds or []):
        if extra_seed is None:
            continue
        try:
            seed_pool.append(np.array(extra_seed, dtype=float))
        except Exception:
            continue
    pool_rng = np.random.default_rng(20260214)
    for _ in range(3 if fast else 4):
        q = pool_rng.normal(size=4)
        q_norm = float(np.linalg.norm(q))
        if not np.isfinite(q_norm) or q_norm <= 1e-9:
            continue
        qw, qx, qy, qz = (q / q_norm).tolist()
        seed_pool.append(np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]))

    # 固定入选集合：在函数入口用**扫描中心** current_f 对种子池粗筛一次，之后所有
    # 焦距候选共用同一批起点（当前相机姿态永远入选）。
    #
    # 若让每个 score_candidate 各自对池子排序，入选集合会随 f_pixels 变化 —— 焦距
    # 微变就可能换掉一个起点，残差曲线随之跳变，表现为焦距抖动（实测旧实现下焦距
    # 对初值的离散度 1.696mm，是 1.2.0 的 2.9 倍）。固定集合还省掉了每轮粗筛的
    # 开销（旧实现每个候选都要对池子里所有种子各算一次残差）。
    fixed_seeds = []
    try:
        f_pixels_seed = get_effective_f_pixels(
            current_f, sensor_width_mm, sensor_height_mm, sensor_fit, pixel_width, pixel_height)
        if np.isfinite(f_pixels_seed) and f_pixels_seed > 1e-8:
            seed_scored = []
            for _idx, _seed in enumerate(seed_pool):
                _res = compute_rotation_constraint_residual(lines_data, _seed, f_pixels_seed)
                if np.isfinite(_res):
                    seed_scored.append((float(_res), _idx))
            seed_scored.sort()
            keep_seeds = 2 if fast else 3
            picked = [_idx for _res, _idx in seed_scored]
            if 0 in picked:
                picked.remove(0)
            picked = [0] + picked[: max(0, keep_seeds - 1)]
            fixed_seeds = [seed_pool[_idx] for _idx in picked]
    except Exception:
        fixed_seeds = []
    if not fixed_seeds:
        fixed_seeds = list(seed_pool)

    # 扫描阶段的迭代投影次数（与 1.2.0 一致取 12）。手工绘制的线段不可能精确交于
    # 同一消失点，迭代不足会让旋转解算的残差带上噪声，而焦距正是靠"比较不同
    # 候选的残差"选出来的，噪声会直接表现为焦距抖动。开销改用粗扫点数与局部
    # 精化步数去省，而不是牺牲迭代精度。
    scan_iterations = 12
    coarse_count = 7 if fast else 12
    fine_count = 5 if fast else 9
    # 局部精化只比较相对优劣，迭代次数可以再降一档
    local_iterations = max(4, scan_iterations - 4)
    # 实时模式只对最优的 1 个候选做 2 步三分搜索：三分搜索在残差多峰时会随
    # 线条抖动跳到不同局部极小，是拖拽中"画面忽大忽小"的来源之一；但完全跳过
    # 又会让精度停在离散细扫的格点间隔上（实时档约 4.8%，肉眼可见）。
    refine_candidates = 1
    refine_steps = 2 if fast else 5

    # "向扫描中心正则化"的强度，用来压住手工绘制误差造成的抖动。
    #
    # 参照系是**扫描中心 current_f**（消失点法给出的焦距），与 1.2.0 一致。
    # 曾经把它改成"相机当前焦距 smooth_ref"，本意是做时间平滑，结果适得其反：
    # 惩罚项在"相机焦距已接近真值"时恒为 0，等于把候选钉死在现状，用户看到的
    # 就是"焦距不会变化了"（实测用户工程 50mm→45mm 的残差改善 0.0045 被 0.05
    # 的惩罚 0.0052 吃掉，总分排序把 50mm 选回来）。
    #
    # 权重也回到 1.2.0 的 0.008。实测（用户工程 I:\空净\T90\首图2.blend，
    # 4096x4096、6 条线 X/Y 各 3）：权重 0.012/0.025 时焦距对初值的敏感度
    # （初值 20/35/50/85/135 的离散度）是 1.696mm、单次约 105ms；1.2.0
    # （权重 0.008、单起点、12 次迭代）离散度 0.587mm、约 21ms。
    proximity_weight = 0.008
    smooth_ref = float(smooth_ref_mm) if smooth_ref_mm is not None else current_f
    if not np.isfinite(smooth_ref) or smooth_ref <= 1e-6:
        smooth_ref = current_f

    def score_candidate(f_candidate, iterations=None, seed=None):
        """对单个焦距候选打分：线长加权残差 + 轻微偏离扫描中心惩罚。

        seed 非空时用它当**唯一起点**（warm start）：相邻焦距的最优姿态相差不到
        0.5°，从上一个候选的姿态出发比"每个候选都从头多起点搜"又快又准。
        """
        iters = scan_iterations if iterations is None else iterations
        f_candidate = float(np.clip(f_candidate, 8.0, 2000.0))
        f_pixels = get_effective_f_pixels(
            f_candidate,
            sensor_width_mm,
            sensor_height_mm,
            sensor_fit,
            pixel_width,
            pixel_height,
        )
        if not np.isfinite(f_pixels) or f_pixels <= 1e-8:
            return None

        # 粗扫必须多起点：焦距是"比较不同候选的残差"选出来的，若某个候选的旋转
        # 卡在局部极小，它的残差会被系统性抬高、谷底位置随之偏掉。实测用户的
        # 工程里，单起点扫描给出的是单调下降的曲线（没有谷底、根本无法定焦距），
        # 多起点扫描才有明确谷底（45mm 处 79.5，两侧 16mm→458、180mm→241）。
        # 起点集合在函数入口已固定（fixed_seeds），不随候选焦距重排序。
        if seed is not None:
            rot_candidate = solve_camera_rotation_constrained(
                lines_data, f_pixels, seed, iterations=iters)
        else:
            rot_candidate, _cand_residual = solve_rotation_multistart(
                lines_data,
                f_pixels,
                fixed_seeds,
                iterations=iters,
                extra_random=0,
                keep=len(fixed_seeds),
            )
        if rot_candidate is None:
            return None

        residual = compute_rotation_constraint_residual(lines_data, rot_candidate, f_pixels)
        if not np.isfinite(residual):
            return None

        proximity_penalty = proximity_weight * abs(math.log(max(f_candidate, 1e-6) / current_f))
        return (residual + proximity_penalty, residual, f_candidate, rot_candidate)

    def scan_factors(factors, iterations=None, seed=None):
        scored = []
        for factor in factors:
            item = score_candidate(current_f * factor, iterations=iterations, seed=seed)
            if item is None:
                continue
            if any(abs(item[2] - prev) < 1e-6 for prev, *_ in scored):
                continue
            scored.append(item)
        return scored

    # 阶段一: 粗扫定位最优焦距的大致区间。
    # 用对数均匀分布而不是线性分布：焦距是乘性量，线性网格在"初始焦距与真值
    # 相差 2 倍以上"时会留下过大的空档（实测从 85mm 起步、真值 35mm 时，
    # 线性 12 点里最接近的是 29.75 与 44.8，扫描彻底错过真值并收敛到 37.2）。
    # 范围取 0.15x~3.0x，可覆盖初始值与真值相差约 6 倍的情形；粗网格的精度
    # 由后面的细扫与三分搜索补回来。
    # 粗扫只需要把最优区间定位到"细扫的 ±12% 窗口内"，用局部精化的迭代次数就够；
    # 实测粗扫用 8 次与 4 次迭代选出的是同一个区间，但前者在 12 个粗扫点上要多花
    # 一倍时间（每次 score_candidate 都要对种子池里 10 个起点各跑一遍迭代）。
    coarse_factors = np.geomspace(0.15, 3.0, coarse_count)
    scored = scan_factors(coarse_factors, iterations=local_iterations)

    # 始终把"扫描中心 current_f"与"正则参考 smooth_ref"都纳入候选。
    # current_f 是消失点法给出的焦距，可能离谱（实测能偏 +131%）；smooth_ref
    # 是相机当前焦距，是独立于消失点估计的先验。两者都进候选，下面的基线比较
    # 才有一个"保守答案"可比 —— 若扫描结果连相机初值都赢不了，就不该改焦距。
    #
    # 注意：加入时必须和 score_candidate 用同一套打分（含 proximity penalty）。
    # 只放纯残差会让 current_f 永远占便宜 —— 实测 level_side 视角 ±8px 噪声下，
    # current_f=80.7083mm 的纯残差 0.022425 小于扫描最优点 47.25mm 的总分
    # 0.02401，于是"最优"永远是 current_f 自己，离谱的消失点焦距被原样保留
    # （实测 130.6% 的焦距误差正是这么来的）。
    for f_base in (current_f, smooth_ref):
        if any(abs(item[2] - f_base) < 1e-6 for item in scored):
            continue
        f_pixels0 = get_effective_f_pixels(
            f_base, sensor_width_mm, sensor_height_mm, sensor_fit, pixel_width, pixel_height)
        if not np.isfinite(f_pixels0) or f_pixels0 <= 1e-8:
            continue
        rot0 = solve_camera_rotation_constrained(
            lines_data, f_pixels0, current_rot_matrix, iterations=scan_iterations)
        if rot0 is None:
            continue
        residual0 = compute_rotation_constraint_residual(lines_data, rot0, f_pixels0)
        if not np.isfinite(residual0):
            continue
        penalty0 = proximity_weight * abs(math.log(max(f_base, 1e-6) / current_f))
        scored.append((residual0 + penalty0, residual0, f_base, rot0))

    if not scored:
        return {
            'f_mm': float(current_f_mm),
            'reliable': False,
            'baseline_residual': float('inf'),
            'best_residual': float('inf'),
        }

    scored.sort(key=lambda item: item[0])
    best = scored[0]

    # 阶段二: 在粗扫最优附近细扫
    f_best = float(best[2])
    lo = max(8.0, f_best * 0.88)
    hi = min(2000.0, f_best * 1.12)
    if hi > lo:
        fine_factors = np.linspace(lo / current_f, hi / current_f, fine_count)
        # warm start：从粗扫最优的姿态出发（相邻焦距的最优姿态差异极小）
        scored.extend(scan_factors(fine_factors, seed=best[3]))
        scored.sort(key=lambda item: item[0])

    # 阶段三: 局部连续精化。
    # 前两阶段是离散网格（细扫相邻候选间隔约 1mm），会在最优值附近留下约 1%
    # 的系统偏差：实测理想数据下真值 35.000 只能收敛到 34.63~35.36，残差停在
    # ~5e-4，而真值处残差是 ~8e-8（差 4 个数量级）。这里对细扫前几名分别做
    # 三分搜索连续逼近 —— 只对第 1 名做会在残差多峰时陷入局部极小。
    for _score, _res, f_ref, _rot in scored[:refine_candidates]:
        lo_f = max(8.0, f_ref * 0.94)
        hi_f = min(2000.0, f_ref * 1.06)
        for _ in range(refine_steps):
            if hi_f - lo_f <= max(0.005, f_ref * 1e-4):
                break
            m1 = lo_f + (hi_f - lo_f) / 3.0
            m2 = hi_f - (hi_f - lo_f) / 3.0
            s1 = score_candidate(m1, iterations=local_iterations, seed=_rot)
            s2 = score_candidate(m2, iterations=local_iterations, seed=_rot)
            if s1 is not None:
                scored.append(s1)
            if s2 is not None:
                scored.append(s2)
            if s1 is None and s2 is None:
                break
            if s1 is None:
                lo_f = m1
                continue
            if s2 is None:
                hi_f = m2
                continue
            if s1[0] <= s2[0]:
                hi_f = m2
            else:
                lo_f = m1

    scored.sort(key=lambda item: item[0])

    # 基线取消失点法给出的扫描中心 current_f，与 1.2.0 一致。
    # 判据问的是"扫描结果比消失点法的估计更好吗"（几何一致性）；曾改成问
    # "比相机现状更好吗"（保守性），与上面的惩罚项形成双重保守，结果焦距
    # 几乎永远不动（实测用户工程五种初值离散度 1.696mm，1.2.0 只有 0.587mm）。
    # current_f 本身离谱时，由 strong_evidence（残差数量级优势）与
    # reliable=False 时返回 smooth_ref 两处兜底。
    baseline_items = [item for item in scored if abs(item[2] - current_f) < 1e-6]
    if not baseline_items:
        baseline_items = [item for item in scored if abs(item[2] - smooth_ref) < 1e-6]
    if baseline_items:
        baseline_score = baseline_items[0][0]
        baseline_residual = baseline_items[0][1]
    else:
        baseline_score = scored[0][0]
        baseline_residual = scored[0][1]

    # 强证据：真实解在残差上有数量级优势时，改用"纯残差"重新择优，跳过 proximity
    # penalty。实测（无噪声、真值 35mm、相机初值 135mm）：消失点给出的 current_f=35mm
    # 残差是 **0**，但它的 penalty = 0.008 × |ln(35/135)| = 0.0108；而 smooth_ref=135mm
    # 的残差只有 0.0096 且 penalty 为 0 —— 按总分排序会把正确答案罚输，焦距原样停在
    # 135mm（误差 285.7%，实测 10/15 例）。penalty 的本意是"抑制噪声引起的随机漂移"，
    # 不该压过"残差接近 0"这种强证据。噪声下最优残差实测 0.0125~0.0316（±8px），
    # 与基线之比 0.65~0.88，远达不到 0.5 的门槛，所以这个分支不会在噪声下误触发。
    by_residual = sorted(scored, key=lambda item: item[1])
    strong_evidence = (
        np.isfinite(by_residual[0][1])
        and np.isfinite(baseline_residual)
        and baseline_residual > 1e-9
        and by_residual[0][1] < baseline_residual * 0.5
    )
    if strong_evidence:
        best_score, best_residual, best_f_mm, _best_rot = by_residual[0]
    else:
        best_score, best_residual, best_f_mm, _best_rot = scored[0]

    # 残差曲线是否真的存在内部极小？
    # 实测（T1 全矩阵：level_side 视角 = 相机几乎水平、X/Y 轴接近平行于像面）：
    # ±8px 端点噪声下残差随焦距**单调递减**（18mm→0.0337、35mm→0.0245、
    # 80.7mm→0.0224、140mm→0.0217），根本没有谷底 —— 因为约束残差 |axis·n|
    # 里的法线 n=(a,b,-c/f) 在 f 增大时位置分量 c/f 趋于 0，约束退化成"只看
    # 线段方向"，系统变松、残差自然变小，并不是大焦距更对。此时"最小残差"
    # 会落在扫描区间的端点上，说明焦距不可辨识，必须保持初值，否则会解出
    # 80.7mm 这种偏差 +131% 的结果（实测 130.6% 的焦距误差正是这么来的）。
    f_lo = min(item[2] for item in scored)
    f_hi = max(item[2] for item in scored)
    span = f_hi - f_lo
    at_edge = False
    if span > 1e-6:
        edge_band = span * 0.05
        at_edge = best_f_mm <= f_lo + edge_band or best_f_mm >= f_hi - edge_band

    improvement = baseline_residual - best_residual
    improvement_ratio = improvement / max(abs(baseline_residual), 1e-6)
    # "焦距是否改变了"必须相对基线（相机当前焦距）判断，而不是相对 current_f。
    # current_f 是消失点给的扫描中心，它本身可能就是离谱值；若拿它当参照，
    # "扫描最优恰好等于 current_f"会被判成"没改变"而否掉，于是又退回 current_f，
    # 逻辑上自相矛盾。实测 bottom_up/YZ/n8/s2：current_f = 64.02mm（真值 35mm），
    # 扫描最优与基线的残差只差 0.00011，旧判据 reliable=False 后原样返回 64.02mm，
    # 焦距误差 82.9%；改用基线作参照后正确保持相机初值 50mm（误差 42.9%）。
    # 死区统一为 1%（与 1.2.0 一致）。实时档曾用 5%，但用户工程实测焦距对
    # 初值的离散度已达 1.696mm（约 4%），5% 死区会把真实的变化一起吸收，
    # 表现为"焦距不动"。抖动改由 reliable 判据（相对残差改善 + 谷底抬升）把关。
    changed = abs(best_f_mm - current_f) > max(0.1, current_f * 0.01)

    # 曾经有过一条"消失点法与残差扫描分歧超过 50% 时无条件相信扫描结果"的判据，
    # 现已移除。它当初是为了救"current_f 是噪声下偏 +131% 的离谱值、却被原样保留"
    # 这个场景，但那个场景现在由两处更可靠的机制覆盖：① reliable=False 时返回的是
    # smooth_ref（相机当前焦距）而不是 current_f；② strong_evidence 分支在残差有
    # 数量级优势时直接采用扫描结果。反过来，这条判据在噪声下极危险 —— 消失点漂到
    # 4304px 时 current_f 本身就离谱，扫描结果同样离谱，两者的"分歧"反而是常态。
    # 实测（±8px、初值=真值 35mm、30 种子 × 5 视角 = 150 例）：带这条判据时 6 例
    # 被拉到 48.4% 的焦距误差，移除后归零。

    # "最优残差相对基线有多大改善"是判断谷底是否真实的信号。
    #
    # 曾经用**绝对阈值**：best_residual × pixel_height <= 2.0（按合成 ±8px 端点
    # 噪声标定，对应约 ±1px 噪声）。但真实用户是**手绘**线条，误差是系统性的
    # （线段不严格交于同一消失点），它把整条残差曲线整体抬高 —— 实测用户工程
    # （I:\空净\T90\首图2.blend，4096x4096）该值落在 47~590 之间，恒大于 2.0，
    # 判据等于失效，焦距更新全押在 valley_ok 一条上，用户看到的就是"焦距不动"。
    # 改成相对基线（消失点法的估计）的改善比例后与手绘精度无关；仍乘 pixel_height
    # 只是为了量纲统一，比值本身已与分辨率无关。
    residual_norm = best_residual * float(pixel_height) if np.isfinite(best_residual) else float('inf')
    baseline_norm = baseline_residual * float(pixel_height) if np.isfinite(baseline_residual) else float('inf')

    # 谷底"两侧抬升"：真谷底的两端残差都会明显高于谷底，而端点噪声造成的假谷底
    # 至少有一侧是平的。
    #
    # 为什么不用"谷底相对两端的深度比值"（曾用 valley_sharpness = best/edge_ref）：
    # 那个比值随扫描范围变化。实测合成 ±8px、XYZ、1920x1080 的第 3 号种子，在
    # 20~70mm 范围内比值是 0.98（看起来"很平"，像噪声），但把范围放宽到 8~80mm
    # 后比值降到 0.549 —— 于是"深度 <= 0.7"反而把它误判成"真谷底"而放行，焦距从
    # 35mm 漂到 10.8mm。同一批数据换个扫描范围就换个结论，说明它不是稳定判据。
    #
    # 改用"两侧各自的抬升倍数"：真谷底要求两端残差都至少比谷底高 50%。
    # 实测用户工程（I:\空净\T90\首图2.blend，4096x4096）：谷底 40mm 处 167.2，
    # 低端 16mm 处 483.8（+189%）、高端 135mm 处 302.7（+81%），两侧都过；
    # 合成 ±8px 第 3 号种子：谷底 12mm 处 0.0281，低端 8mm 处 0.0344（仅 +22%），
    # 被挡住。谷底落在扫描区间边缘时抬升为 0，判为不可靠（保守，符合预期）。
    valley_lift = 0.0
    if len(scored) >= 3 and np.isfinite(best_residual) and best_residual > 1e-9:
        by_focal = sorted(scored, key=lambda item: item[2])
        best_index = min(range(len(by_focal)), key=lambda i: by_focal[i][1])
        lifts = []
        if best_index > 0:
            lifts.append((float(by_focal[0][1]) - float(best_residual)) / float(best_residual))
        if best_index < len(by_focal) - 1:
            lifts.append((float(by_focal[-1][1]) - float(best_residual)) / float(best_residual))
        if lifts:
            valley_lift = float(min(lifts))
    valley_ok = bool(np.isfinite(valley_lift) and valley_lift >= 0.5)

    if fast:
        # 实时模式：判据要求"残差有足够大的绝对改善"，而不是相对改善 —— 相对判据
        # 挡不住噪声（0.003 相对 0.006 已经算 50% 的"改善"了）。真正把噪声挡在
        # 外面的是下面的 residual_norm <= 2.0，所以这里的 improvement 阈值可以放得
        # 很低：实测无噪声、初值 50mm、真值 35mm 时，各视角 50mm 处的残差只有
        # 0.003~0.008，沿用 0.008 会让实时档 50 例里只有 1 例跟随焦距 —— 用户拖拽
        # 线条时焦距纹丝不动。改成 0.002 后无噪声与亚像素噪声下正常精化，±1px 以上
        # 由 residual_norm 拦住（实测 ±1px 时 residual_norm 中位数 2.4 > 2.0）。
        reliable = (
            changed
            and np.isfinite(best_residual)
            and np.isfinite(baseline_residual)
            and improvement >= 0.002
        )
        reliable = reliable and not at_edge and (residual_norm <= baseline_norm * 0.6 or valley_ok)
    else:
        # 精确档同样要求"残差有足够大的绝对改善"，理由与实时档一致，只是阈值可以
        # 略松一点。实测（±8px、初值=真值 35mm、30 种子 × 5 视角 = 150 例）：
        # 噪声下 best_residual/baseline_residual 的中位数是 0.966，improvement 最大
        # 0.0092 —— 旧的 0.97 / 0.003 / 0.02 三条判据全部落在噪声波动范围内，结果
        # 9/150 例被拉到 48.4% 的焦距误差。收紧到 0.8 / 0.008 / 0.2 后噪声波动
        # 完全够不着；而"初值远离真值"时残差有数量级优势（无噪声初值 135mm 时
        # best_residual=0、baseline≈0.0096），由 strong_evidence 分支接管，不受影响。
        # 相对改善阈值从 0.8 / 0.008 / 0.2 放宽到 0.95 / 0.002 / 0.05。
        #
        # 旧阈值是按"合成 ±8px 端点噪声"标定的：那时 best/baseline 的中位数是
        # 0.966，所以 0.97/0.003/0.02 全部落在噪声波动内、必须收紧到 0.8 才挡得住
        # 9/150 例 48.4% 的焦距误差。但真实用户是**手绘**线条，误差是系统性的
        # （线段不严格交于同一消失点），它把整条残差曲线整体抬高、却让谷底位置
        # 非常稳定 —— 实测用户工程 best/baseline = 0.875、谷底锐度 0.35，旧阈值
        # 0.8 差一点就把它挡在外面，用户看到的就是"焦距永远不动"。
        # 真正区分"手绘系统误差"与"端点噪声"的是 valley_ok（谷底两侧抬升，实测
        # 用户工程两侧 +189%/+81%、合成 ±8px 只有 +22%），所以这里可以放宽，由
        # (residual_norm <= 2.0 or valley_ok) 统一把关。
        reliable = (
            changed
            and np.isfinite(baseline_residual)
            and np.isfinite(best_residual)
            and (
                best_residual <= baseline_residual * 0.95
                or improvement >= 0.002
                or improvement_ratio >= 0.05
            )
        )
        reliable = reliable and not at_edge and (residual_norm <= baseline_norm * 0.6 or valley_ok)

    # 焦距判为不可靠时（"这次没解出可信的焦距，先不动"），在"相机当前焦距"
    # (smooth_ref) 与"消失点法给出的扫描中心"(current_f) 之间**选残差更小的那个**，
    # 而不是无条件退回 smooth_ref。
    #
    # 为什么不能无条件退回 smooth_ref：smooth_ref 是 cam.data.lens，也就是用户
    # 随手填的初值，它可能比消失点法的估计差得多。实测用户工程
    # （I:\空净\T90\首图2.blend，4096x4096、真值约 44.25mm）：把相机 lens 设成
    # 20mm 再点解算，消失点法给出扫描中心 42.987mm（残差 0.012013），扫描最优
    # 44.25mm（残差 0.011533）—— 改善只有 4%，exact 判据（0.95 / 0.002 / 0.05）
    # 全部够不着，于是 reliable=False，退回 smooth_ref = 20mm，残差从 0.0936
    # 恶化到 0.0955。用户看到的就是"点了匹配相机，焦距纹丝不动"，而且用的还是
    # 自己随手填的错误初值 —— 这正是用户报告的核心症状。
    #
    # 两个候选本来就都已经在 scored 里（见上面把它们纳入候选的说明），且用的是
    # 同一套解算流程，残差可直接比较，不需要额外计算。反过来，相机当前焦距确实
    # 比扫描中心更好时（实测同一工程：相机 44.0901mm 残差 0.011599 < 扫描中心
    # 42.987mm 的 0.012013），仍然保持相机现状，不会无故改动。
    #
    # 调用方没提供 smooth_ref_mm 时 smooth_ref 就等于 current_f，候选只剩一个，
    # 行为与旧版一致。
    locked_candidates = [
        item for item in scored
        if abs(item[2] - current_f) < 1e-6
        or (smooth_ref_mm is not None and abs(item[2] - smooth_ref) < 1e-6)
    ]
    if locked_candidates:
        locked_f = float(min(locked_candidates, key=lambda item: item[1])[2])
    else:
        locked_f = float(smooth_ref if smooth_ref_mm is not None else current_f_mm)

    # 细分"不可靠"的两种含义，供调用方给出准确提示。
    #
    # 曾经只有 refined / locked 两种状态，于是"焦距其实已经正确"也被报成
    # "约束不足，焦距已锁定" —— 实测 XYZ 各 3 条线、无噪声、相机初值 50mm、
    # 真值 35mm：消失点法给出 current_f = 35mm（残差 0），扫描最优同样是 35mm，
    # 于是 changed=False → reliable=False，而焦距结果完全正确，用户却看到
    # "焦距已锁定"，与面板上 50→35 的实际变化自相矛盾。
    #
    # consistent：不可靠，但残差比较显示"消失点给出的焦距"就是最优 —— 焦距
    #             无需改变（不是保守退回），调用方不必提示。
    # fallback ：不可靠，且退回的是"相机当前焦距"（smooth_ref）—— 这才是真正
    #             需要提示用户"这次没解出可信焦距"的情况。
    if reliable:
        chosen_state = 'refined'
    elif abs(locked_f - current_f) <= 1e-6:
        chosen_state = 'consistent'
    else:
        chosen_state = 'fallback'

    return {
        'f_mm': float(best_f_mm if reliable else locked_f),
        'reliable': bool(reliable),
        'state': chosen_state,
        'baseline_residual': float(baseline_residual),
        'best_residual': float(best_residual),
    }
