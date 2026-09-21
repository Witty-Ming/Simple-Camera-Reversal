import bpy
import gpu
from gpu_extras.batch import batch_for_shader
import math
import time
from bpy_extras import view3d_utils

from . import utils

_handle = None
_shader_2d_color = None
_ref_count = 0
_last_error = None

# 每帧跨线段累计的虚线顶点上限（每条 LINES 顶点占 2 个位置，这里按顶点数计）
MAX_TOTAL_LINE_VERTS = 120000


def get_shader_2d_color():
    global _shader_2d_color
    if _shader_2d_color is not None:
        return _shader_2d_color

    _shader_2d_color = gpu.shader.from_builtin('UNIFORM_COLOR')
    return _shader_2d_color


def to_shader_positions(points):
    # 过滤非有限值：退化消失点产生的 NaN 一旦进入顶点缓冲，画面上会出现
    # 无法解释的杂线，而 GPU 侧不会报错。
    out = []
    for point in points:
        try:
            x = float(point[0])
            y = float(point[1])
        except (TypeError, ValueError, IndexError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            out.append((x, y, 0.0))
    return out


# 生成虚线顶点阵列
def build_dashed_line(points, dash_length=12, gap_length=8, budget=None):
    """
    :param budget: 可选的剩余顶点预算（list，budget[0] 为剩余数量）。
                   线段条数没有上限，1000 条长线会产生约 200 万顶点/帧，
                   因此需要一个跨线段累计的总量保护，而不是只看单条线段。
    """
    if len(points) < 2:
        return []
    speed = 50
    offset = (time.time() * speed) % (dash_length + gap_length)

    verts = []
    max_segments = 2048
    max_screen_length = 20000.0

    def remaining():
        if budget is None:
            return None
        return budget[0] - len(verts)

    for i in range(len(points) - 1):
        a = points[i]
        b = points[i + 1]
        vec = (b[0] - a[0], b[1] - a[1])
        length = math.hypot(vec[0], vec[1])
        if length == 0 or not math.isfinite(length):
            continue

        left = remaining()
        if left is not None and left <= 2:
            break

        # 防止极端坐标导致虚线细分循环过大造成卡死
        if length > max_screen_length:
            verts.extend((a, b))
            continue

        dir = (vec[0] / length, vec[1] / length)

        current_pos = -(dash_length + gap_length) + offset
        seg_count = 0
        while current_pos < length and seg_count < max_segments:
            left = remaining()
            if left is not None and left <= 2:
                break

            start_d = max(0, current_pos)
            end_d = min(length, current_pos + dash_length)

            if end_d > start_d:
                start_pt = (a[0] + dir[0] * start_d, a[1] + dir[1] * start_d)
                end_pt = (a[0] + dir[0] * end_d, a[1] + dir[1] * end_d)
                verts.extend((start_pt, end_pt))

            current_pos += dash_length + gap_length
            seg_count += 1

    if budget is not None:
        budget[0] = max(0, budget[0] - len(verts))

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
    global _last_error
    try:
        context = bpy.context
        if not context or not getattr(context, "scene", None):
            return
        # 用 get_camera_view_region_data 而不是 is_camera_view：后者只检查
        # view_perspective == 'CAMERA'，不校验这个视图锁定的是哪台相机。
        # 视图锁定到别的相机时，下面所有几何都按 scene.camera 计算，叠加层
        # 会整体错位。
        rv3d = utils.get_camera_view_region_data(context)
        if rv3d is None:
            return
        if not getattr(context.scene, "cmp_data", None) or not context.scene.cmp_data.lines:
            return
        cam = context.scene.camera
        if not cam:
            return

        region = getattr(context, "region", None)
        if region is None:
            return

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

        # 跨线段累计的顶点预算：线段条数本身没有上限，只靠单条线段的
        # max_segments 保护不够。
        line_budget = [MAX_TOTAL_LINE_VERTS]

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
            get_batch(color)['LINES'].extend(
                build_dashed_line([p1_2d, p2_2d], 12, 8, budget=line_budget))

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

        # 记录进入前的 GPU 状态并原样恢复 —— 绘制回调共享 Blender 的 GPU
        # 状态，硬编码恢复成 NONE/1 会覆盖掉其他 draw handler 的设置。
        try:
            prev_blend = gpu.state.blend_get()
        except Exception:
            prev_blend = None
        try:
            prev_line_width = gpu.state.line_width_get()
        except Exception:
            prev_line_width = None

        shader.bind()
        try:
            gpu.state.blend_set('ALPHA')

            for color, data in batches.items():
                shader.uniform_float("color", color)

                if data['TRIS']:
                    tris = to_shader_positions(data['TRIS'])
                    if tris:
                        batch = batch_for_shader(shader, 'TRIS', {"pos": tris})
                        batch.draw(shader)

                if data['LINES']:
                    lines_pos = to_shader_positions(data['LINES'])
                    if lines_pos:
                        gpu.state.line_width_set(2)
                        try:
                            batch = batch_for_shader(shader, 'LINES', {"pos": lines_pos})
                            batch.draw(shader)
                        finally:
                            gpu.state.line_width_set(prev_line_width if prev_line_width else 1)
        finally:
            try:
                gpu.state.line_width_set(prev_line_width if prev_line_width else 1)
            finally:
                try:
                    gpu.state.blend_set(prev_blend if prev_blend else 'NONE')
                finally:
                    try:
                        gpu.shader.unbind()
                    except Exception:
                        pass

        _last_error = None

    except Exception as e:
        # POST_PIXEL 每帧都跑：持续异常会以 25fps 刷控制台并明显拖慢主线程，
        # 因此只在错误内容变化时打印一次。
        message = f"CMP 2D Draw Error: {e}"
        if message != _last_error:
            _last_error = message
            print(message)


def redraw_timer():
    try:
        context = bpy.context
        screen = getattr(context, "screen", None)
        scene = getattr(context, "scene", None)
        cmp_data = getattr(scene, "cmp_data", None) if scene is not None else None
        if cmp_data is not None and cmp_data.lines and screen:
            scene_camera = getattr(scene, "camera", None)
            for area in screen.areas:
                if area.type != 'VIEW_3D':
                    continue
                space = getattr(area.spaces, "active", None)
                if space is None or space.type != 'VIEW_3D':
                    continue
                rv3d = getattr(space, "region_3d", None)
                if rv3d is None or rv3d.view_perspective != 'CAMERA':
                    continue
                # 与绘制回调保持一致：只重绘真正显示 scene.camera 的视图
                space_camera = getattr(space, "camera", None)
                if space_camera is not None and space_camera != scene_camera:
                    continue
                area.tag_redraw()
        else:
            # 没有参考线时不需要流动虚线动画，降频以省下无谓的主线程唤醒
            return 0.25
    except Exception:
        pass
    return 0.04  # 限制为大概 25fps


def reset_runtime_state():
    """
    清理"跨文件残留"的运行时状态。

    加载新 .blend 时 Blender 会丢弃 modal 操作符与非持久定时器，但不会调用
    Python 的 quit()/cancel()，于是 is_drawing_mode 会残留为 True、虚线动画
    定时器消失。这里做统一兜底。
    """
    try:
        for scene in bpy.data.scenes:
            cmp_data = getattr(scene, "cmp_data", None)
            if cmp_data is None:
                continue
            cmp_data.is_drawing_mode = False
            cmp_data.is_creating_line = False
            cmp_data.active_index = -1
    except Exception:
        pass

    try:
        from . import properties
        properties.reset_horizon_update_state()
    except Exception:
        pass


def _on_load_post(_dummy):
    reset_runtime_state()
    # 加载新文件后绘制回调必须彻底清掉（引用计数已无意义）
    unregister(force=True)


def register():
    global _handle, _ref_count
    _ref_count += 1
    if _handle is None:
        _handle = bpy.types.SpaceView3D.draw_handler_add(draw_callback, (), 'WINDOW', 'POST_PIXEL')

    if not bpy.app.timers.is_registered(redraw_timer):
        bpy.app.timers.register(redraw_timer, persistent=True)

    if _on_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_load_post)


def unregister(force=False):
    """
    引用计数式注销：多个 3D 视图各有一个绘制模态实例时，只有最后一个退出
    才真正移除回调；否则先退出的那个会把另一个仍在用的叠加层一起关掉。

    :return: 是否真正完成了注销（引用计数归零或 force=True）。调用方据此
             决定要不要关闭 Scene 级的 is_drawing_mode。
    """
    global _handle, _shader_2d_color, _ref_count

    if not force:
        _ref_count = max(0, _ref_count - 1)
        if _ref_count > 0:
            return False

    if _handle:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_handle, 'WINDOW')
        except Exception as e:
            # 只有确认移除成功才清空句柄：否则这个仍然注册着的 handler
            # 会永久泄漏，而后续 register() 还会再加一个。
            print(f"[CameraMatch] 移除绘制回调失败（保留句柄以便重试）: {e}")
        else:
            _handle = None

    try:
        if bpy.app.timers.is_registered(redraw_timer):
            bpy.app.timers.unregister(redraw_timer)
    except Exception as e:
        print(f"[CameraMatch] 注销重绘定时器失败: {e}")

    try:
        if _on_load_post in bpy.app.handlers.load_post:
            bpy.app.handlers.load_post.remove(_on_load_post)
    except Exception:
        pass

    _ref_count = 0
    _shader_2d_color = None
    return True
