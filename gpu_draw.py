import bpy
import gpu
from gpu_extras.batch import batch_for_shader
import math
import time
from bpy_extras import view3d_utils

from . import utils

_handle = None
_shader_2d_color = None


def get_shader_2d_color():
    global _shader_2d_color
    if _shader_2d_color is not None:
        return _shader_2d_color

    _shader_2d_color = gpu.shader.from_builtin('UNIFORM_COLOR')
    return _shader_2d_color


def to_shader_positions(points):
    return [(float(point[0]), float(point[1]), 0.0) for point in points]


# 生成虚线顶点阵列
def build_dashed_line(points, dash_length=12, gap_length=8):
    if len(points) < 2:
        return []
    speed = 50
    offset = (time.time() * speed) % (dash_length + gap_length)

    verts = []
    max_segments = 2048
    max_screen_length = 20000.0

    for i in range(len(points) - 1):
        a = points[i]
        b = points[i + 1]
        vec = (b[0] - a[0], b[1] - a[1])
        length = math.hypot(vec[0], vec[1])
        if length == 0:
            continue

        # 防止极端坐标导致虚线细分循环过大造成卡死
        if length > max_screen_length:
            verts.extend((a, b))
            continue

        dir = (vec[0] / length, vec[1] / length)

        current_pos = -(dash_length + gap_length) + offset
        seg_count = 0
        while current_pos < length and seg_count < max_segments:
            start_d = max(0, current_pos)
            end_d = min(length, current_pos + dash_length)

            if end_d > start_d:
                start_pt = (a[0] + dir[0] * start_d, a[1] + dir[1] * start_d)
                end_pt = (a[0] + dir[0] * end_d, a[1] + dir[1] * end_d)
                verts.extend((start_pt, end_pt))

            current_pos += dash_length + gap_length
            seg_count += 1

    return verts


# 生成空心圆环顶点阵列 (类型为 LINES)
def build_circle_lines(center, radius, seg=24):
    verts = []
    prev_pt = None
    for i in range(seg + 1):
        ang = 2 * math.pi * i / seg
        x = center[0] + math.cos(ang) * radius
        y = center[1] + math.sin(ang) * radius
        pt = (x, y)
        if prev_pt is not None:
            verts.extend((prev_pt, pt))
        prev_pt = pt
    return verts


# 生成实心圆顶点阵列 (类型为 TRIS)
def build_filled_circle_tris(center, radius, seg=24):
    verts = []
    first_pt = (center[0] + radius, center[1])
    prev_pt = first_pt
    for i in range(1, seg + 1):
        ang = 2 * math.pi * i / seg
        x = center[0] + math.cos(ang) * radius
        y = center[1] + math.sin(ang) * radius
        pt = (x, y)
        verts.extend((center, prev_pt, pt))
        prev_pt = pt
    return verts


# 生成菱形实心三角顶点阵列 (类型为 TRIS)
def build_filled_diamond_tris(center, radius):
    cx, cy = center
    top = (cx, cy + radius)
    right = (cx + radius, cy)
    bottom = (cx, cy - radius)
    left = (cx - radius, cy)
    return [
        center, top, right,
        center, right, bottom,
        center, bottom, left,
        center, left, top,
    ]


# 生成菱形线框顶点阵列 (类型为 LINES)
def build_diamond_lines(center, radius):
    cx, cy = center
    top = (cx, cy + radius)
    right = (cx + radius, cy)
    bottom = (cx, cy - radius)
    left = (cx - radius, cy)
    return [
        top, right,
        right, bottom,
        bottom, left,
        left, top,
    ]


def draw_callback():
    try:
        context = bpy.context
        if not context or not getattr(context, "scene", None):
            return
        if not utils.is_camera_view(context):
            return
        if not getattr(context.scene, "cmp_data", None) or not context.scene.cmp_data.lines:
            return
        cam = context.scene.camera
        if not cam:
            return

        region = context.region
        rv3d = context.space_data.region_3d

        mw = cam.matrix_world
        TR, TL, BL, BR = utils.get_ordered_frame_points(context)
        if not TR:
            return

        def get_world(u, v):
            top = TL.lerp(TR, u)
            bot = BL.lerp(BR, u)
            p_loc = bot.lerp(top, v)
            return mw @ p_loc

        lines = context.scene.cmp_data.lines
        active_idx = context.scene.cmp_data.active_index
        is_drawing_mode = context.scene.cmp_data.is_drawing_mode
        is_creating_line = context.scene.cmp_data.is_creating_line

        alpha = 0.8
        cols = {
            'X': (1.0, 0.3, 0.3, alpha),
            'Y': (0.3, 1.0, 0.3, alpha),
            'Z': (0.3, 0.5, 1.0, alpha)
        }
        white = (1.0, 1.0, 1.0, 1.0)
        highlight_color = (1.0, 1.0, 0.0, 1.0)
        horizon_color = (0.0, 1.0, 1.0, 0.9)
        horizon_center_color = (0.0, 1.0, 1.0, 0.35)

        # 收集渲染数据 (按颜色分类)
        # 结构: batches[color] = {'LINES': [], 'TRIS': []}
        batches = {}

        def get_batch(c):
            if c not in batches:
                batches[c] = {'LINES': [], 'TRIS': []}
            return batches[c]

        for i, line in enumerate(lines):
            p1_3d = get_world(line.start[0], line.start[1])
            p2_3d = get_world(line.end[0], line.end[1])

            p1_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, p1_3d)
            p2_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, p2_3d)

            if p1_2d is None or p2_2d is None:
                continue

            color = cols.get(line.axis, white)
            if i == active_idx and not is_creating_line:
                color = highlight_color

            # 1. 虚线 (添加至 LINES 批次)
            get_batch(color)['LINES'].extend(build_dashed_line([p1_2d, p2_2d], 12, 8))

            # 只有在进入绘制/编辑模式时才显示控制点
            if is_drawing_mode:
                # 2. 中点实心和外圈
                mid_2d = ((p1_2d[0] + p2_2d[0]) / 2, (p1_2d[1] + p2_2d[1]) / 2)
                get_batch(color)['TRIS'].extend(build_filled_circle_tris(mid_2d, 5))
                get_batch(white)['LINES'].extend(build_circle_lines(mid_2d, 7))

                # 3. 选下端点
                if i == active_idx:
                    for pt_2d in [p1_2d, p2_2d]:
                        get_batch(color)['TRIS'].extend(build_filled_circle_tris(pt_2d, 6))
                        get_batch(white)['LINES'].extend(build_circle_lines(pt_2d, 8))

        cmp_data = context.scene.cmp_data
        if cmp_data.horizon_enabled:
            render = context.scene.render
            pixel_res_x, pixel_res_y = utils.get_effective_render_size(render)
            geo = utils.compute_horizon_overlay_geometry(
                lines,
                cmp_data,
                pixel_res_x,
                pixel_res_y,
                region.width,
                region.height,
                context=context,
            )

            if geo is not None:
                line_a = tuple(geo['line_region_a'])
                line_b = tuple(geo['line_region_b'])
                center = tuple(geo['center_region'])
                offset_handle = tuple(geo['offset_handle_region'])
                draw_line_flag = geo.get('draw_line', True)

                if draw_line_flag:
                    get_batch(horizon_color)['LINES'].extend(build_dashed_line([line_a, line_b], 20, 10))

                # 保留中心点参考（非手柄）
                get_batch(horizon_center_color)['TRIS'].extend(build_filled_circle_tris(center, 4))

                # 仅保留偏移手柄（菱形）
                get_batch(horizon_center_color)['TRIS'].extend(build_filled_diamond_tris(offset_handle, 7))
                get_batch(horizon_color)['LINES'].extend(build_diamond_lines(offset_handle, 11))

        # 统一执行极少次数的绘制调用
        shader = get_shader_2d_color()
        shader.bind()

        try:
            gpu.state.blend_set('ALPHA')

            for color, data in batches.items():
                shader.uniform_float("color", color)

                if data['TRIS']:
                    batch = batch_for_shader(
                        shader,
                        'TRIS',
                        {"pos": to_shader_positions(data['TRIS'])},
                    )
                    batch.draw(shader)

                if data['LINES']:
                    gpu.state.line_width_set(2)
                    try:
                        batch = batch_for_shader(
                            shader,
                            'LINES',
                            {"pos": to_shader_positions(data['LINES'])},
                        )
                        batch.draw(shader)
                    finally:
                        gpu.state.line_width_set(1)
        finally:
            # 绘制回调共享 Blender 的 GPU 状态，任何异常都必须恢复现场。
            try:
                gpu.state.line_width_set(1)
            finally:
                gpu.state.blend_set('NONE')

    except Exception as e:
        print(f"CMP 2D Draw Error: {e}")


def redraw_timer():
    try:
        context = bpy.context
        screen = getattr(context, "screen", None)
        if context.scene and getattr(context.scene, "cmp_data", None) and context.scene.cmp_data.lines and screen:
            for area in screen.areas:
                if area.type != 'VIEW_3D':
                    continue
                for space in area.spaces:
                    if space.type == 'VIEW_3D' and getattr(space, "region_3d", None) and space.region_3d.view_perspective == 'CAMERA':
                        area.tag_redraw()
                        break
    except Exception:
        pass
    return 0.04  # 限制为大概 25fps


def register():
    global _handle
    if _handle is None:
        _handle = bpy.types.SpaceView3D.draw_handler_add(draw_callback, (), 'WINDOW', 'POST_PIXEL')

    if not bpy.app.timers.is_registered(redraw_timer):
        bpy.app.timers.register(redraw_timer)


def unregister():
    global _handle
    global _shader_2d_color
    if _handle:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_handle, 'WINDOW')
        except Exception as e:
            print(f"[CameraMatch] 移除绘制回调失败: {e}")
        finally:
            _handle = None

    try:
        if bpy.app.timers.is_registered(redraw_timer):
            bpy.app.timers.unregister(redraw_timer)
    except Exception as e:
        print(f"[CameraMatch] 注销重绘定时器失败: {e}")

    _shader_2d_color = None
