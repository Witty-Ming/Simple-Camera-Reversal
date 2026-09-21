import bpy
import numpy as np
import mathutils
import bpy_extras
from . import utils, properties


HORIZON_LOCK_THRESHOLD_PX = 8.0


def solve_camera_core(context, fast=False, _retry=0):
    """
    核心解算函数

    :param fast: True 表示绘制/拖拽过程中的实时解算，降低焦距扫描密度以
                 保住交互帧率；松手与点击 Match Camera 按钮时用 False
                 （完整精度）。两种模式的解算路径与判据完全一致。
    :param _retry: 内部使用。shift 补偿改变了主点基准时会用新 shift 重解一次，
                   保证"解算输入的主点"与"解算输出的 shift"自洽。
    :return: (success, message)
    """
    scene = context.scene
    cam = scene.camera
    if not cam:
        return False, "No Active Camera"

    iface_ = bpy.app.translations.pgettext_iface

    if getattr(cam.data, "type", None) != 'PERSP':
        return False, iface_("Only perspective cameras are supported")

    cmp_data = scene.cmp_data

    # 参考线是"属于某台相机"的。相机切换后的自动清理只发生在绘制模式里，
    # 一旦 DrawLine 没在运行（Esc 退出、重开文件、直接点面板按钮），这里就是
    # 唯一的防线 —— 否则会把 A 相机的线解算到 B 相机上（实测会把 80mm 的
    # 相机直接搬成 35mm）。
    lines_cam = getattr(cmp_data, "lines_camera", None)
    if lines_cam is not None and lines_cam != cam:
        return False, iface_("Guide lines belong to another camera")
    if lines_cam is None:
        # 未绑定（例如快照恢复失败）：认领给当前相机，避免功能卡死
        cmp_data.lines_camera = cam

    lines = cmp_data.lines
    # 单点透视（ONE_POINT）只需要一个轴向的线段即可解算：另外两个轴向会由
    # build_perspective_mode_constraints 注入"画面水平/垂直"合成约束补足。
    if len(lines) < 1:
        return False, "Not enough lines"

    # 完全落在画面外的参考线没有意义（用户看不到背景图就没法对齐），
    # 全部越界时直接拒绝，避免解出 f=11mm 这种荒谬结果还报"成功"。
    if utils.count_offscreen_lines(lines) >= len(lines):
        return False, iface_("All guide lines are outside the camera frame")

    render = scene.render
    pixel_res_x, pixel_res_y = utils.get_effective_render_size(render)

    # 1. 计算真实的像主点 (考虑 shift_x/y)
    cursor_location = scene.cursor.location.copy()
    current_dist = (cam.location - cursor_location).length
    if current_dist < 0.1: current_dist = 10.0

    principal_u, principal_v = utils.compute_principal_point_uv(scene, cam)

    # 2. 准备数据
    lines_data, vp_data_raw, vp_data, axis_weights, horizon_data = utils.solve_horizon_data(
        lines,
        pixel_res_x,
        pixel_res_y,
        scene.cmp_data.horizon_enabled,
        scene.cmp_data.horizon_offset_px,
        principal_u,
        principal_v,
    )

    active_axes = [axis for axis in ('X', 'Y', 'Z') if len(lines_data.get(axis, [])) >= 1]
    if len(active_axes) < 1:
        iface_ = bpy.app.translations.pgettext_iface
        return False, iface_("Requires at least one axis (min 1 line per axis)")

    # 某个轴画了 >=2 条线却求不出消失点 => 这些线互相平行。此时该轴对旋转
    # 几乎没有约束力，解算结果可能明显偏离（实测 2 条平行线会给出 78° 的
    # 旋转误差），但过去仍然报"成功"。这里把它作为提示带回给用户。
    parallel_axes = [
        axis for axis in ('X', 'Y', 'Z')
        if len(lines_data.get(axis, [])) >= 2 and axis not in vp_data_raw
    ]

    perspective_constraints = utils.build_perspective_mode_constraints(
        lines_data,
        pixel_res_x,
        pixel_res_y,
        finite_vp_axes=list(vp_data_raw.keys()),
    )

    # 保留当前的旋转微调状态，并在新解算结果上重新应用
    current_world_rotation = scene.cmp_data.world_rotation
    current_flip_z = scene.cmp_data.flip_z_axis
    current_shift_x = cam.data.shift_x
    current_shift_y = cam.data.shift_y
    cursor_location = scene.cursor.location.copy()
    current_dist = (cam.location - cursor_location).length
    if current_dist < 0.1: current_dist = 10.0

    anchor_screen_offset = None
    target_cursor_uv = None
    horizon_target_uv = None
    horizon_lock_active = False
    try:
        cursor_view = bpy_extras.object_utils.world_to_camera_view(scene, cam, cursor_location)

        # principal_u 和 principal_v 已经在开头计算好了

        if np.isfinite(cursor_view.x) and np.isfinite(cursor_view.y):
            target_cursor_uv = (float(cursor_view.x), float(cursor_view.y))
            anchor_screen_offset = (
                float((cursor_view.x - principal_u) * pixel_res_x),
                float((cursor_view.y - principal_v) * pixel_res_y),
            )

            if scene.cmp_data.horizon_enabled and horizon_data is not None:
                cursor_px = utils.uv_to_centered_px(target_cursor_uv, pixel_res_x, pixel_res_y, principal_u, principal_v)
                dist_to_horizon = abs(utils.signed_distance_to_line_2d(cursor_px, horizon_data['line']))
                if dist_to_horizon <= HORIZON_LOCK_THRESHOLD_PX:
                    proj_px = utils.project_point_to_line_2d(cursor_px, horizon_data['point'], horizon_data['direction'])
                    horizon_target_uv = utils.centered_px_to_uv(proj_px, pixel_res_x, pixel_res_y, principal_u, principal_v)

                    proj_offset = (
                        float((horizon_target_uv[0] - principal_u) * pixel_res_x),
                        float((horizon_target_uv[1] - principal_v) * pixel_res_y),
                    )
                    anchor_screen_offset = proj_offset
                    horizon_lock_active = True
    except Exception:
        anchor_screen_offset = None

    f_mm = None
    rot_matrix = None
    shift_x = 0.0
    shift_y = 0.0
    loc_orbit = None
    solve_mode_hint_key = None

    def solve_strict_mode(
        lines_for_solve,
        f_seed_mm,
        hint_refined_key,
        hint_locked_key,
        allow_focal_refine=True,
        hint_fixed_key=None,
        rot_seed=None,
    ):
        try:
            rot_init = rot_seed if rot_seed is not None else cam.matrix_world.to_3x3()
            strict_result = utils.solve_strict_mode_constrained(
                lines_for_solve,
                f_seed_mm,
                cam.data.sensor_width,
                cam.data.sensor_height,
                cam.data.sensor_fit,
                pixel_res_x,
                pixel_res_y,
                rot_init,
                allow_focal_refine=allow_focal_refine,
                fast=fast,
            )
            if not strict_result.get('ok', False):
                return None

            f_val = float(strict_result.get('f_mm', f_seed_mm))
            rot_val = strict_result.get('rot_matrix')
            if rot_val is None:
                return None

            f_pixels_val = utils.get_effective_f_pixels(
                f_val,
                cam.data.sensor_width,
                cam.data.sensor_height,
                cam.data.sensor_fit,
                pixel_res_x,
                pixel_res_y,
            )
            if not np.isfinite(f_pixels_val) or f_pixels_val <= 1e-8:
                return None

            target_px, target_py = (0.0, 0.0)
            if anchor_screen_offset is not None:
                target_px, target_py = anchor_screen_offset

            ray_cam = np.array([target_px, target_py, -f_pixels_val])
            ray_cam /= np.linalg.norm(ray_cam)
            p_org_cam = ray_cam * current_dist
            loc_val = cursor_location - (rot_val @ mathutils.Vector(p_org_cam))

            focal_state = strict_result.get('focal_state', 'locked')
            if focal_state == 'refined':
                hint_key = hint_refined_key
            elif focal_state == 'fixed' and hint_fixed_key is not None:
                hint_key = hint_fixed_key
            else:
                hint_key = hint_locked_key

            return {
                'f_mm': f_val,
                'rot_matrix': rot_val,
                'shift_x': 0.0,
                'shift_y': 0.0,
                'loc_orbit': loc_val,
                'hint_key': hint_key,
                'strict_result': strict_result,
            }
        except Exception as e:
            print(f"[CameraMatch] Strict solve failed: {e}")
            return None

    iface_ = bpy.app.translations.pgettext_iface

    solve_mode = perspective_constraints.get('mode', 'INSUFFICIENT')
    guided_lines_data = perspective_constraints.get('guided_lines_data', lines_data)

    if solve_mode == 'ONE_POINT':
        strict_solution = solve_strict_mode(
            guided_lines_data,
            cam.data.lens,
            "Constrained solve: focal refined",
            "Constrained solve: focal locked (insufficient constraints)",
            allow_focal_refine=False,
            hint_fixed_key="One-point strict: focal fixed",
        )
        if strict_solution is None:
            return False, iface_("Solving failed. Check line placement.")

        f_mm = strict_solution['f_mm']
        rot_matrix = strict_solution['rot_matrix']
        shift_x = strict_solution['shift_x']
        shift_y = strict_solution['shift_y']
        loc_orbit = strict_solution['loc_orbit']
        solve_mode_hint_key = strict_solution['hint_key']

    else:
        vp_for_full_solve = dict(vp_data)
        if len(vp_for_full_solve) >= 2:
            try:
                solve_axis_weights = {axis: axis_weights.get(axis, 0) for axis in vp_for_full_solve.keys()}
                f_mm, rot_matrix, shift_x, shift_y, loc_orbit = utils.calculate_camera_transform(
                    vp_for_full_solve,
                    cam.data.sensor_width,
                    cam.data.sensor_height,
                    cam.data.sensor_fit,
                    pixel_res_x, pixel_res_y, current_dist,
                    default_f_mm=cam.data.lens,
                    axis_weights=solve_axis_weights,
                    anchor_location=cursor_location,
                    anchor_screen_offset=anchor_screen_offset,
                    current_rot_matrix=cam.matrix_world.to_3x3(),
                )
            except Exception as e:
                print(f"[CameraMatch] Hybrid full solve failed: {e}")
                f_mm = None

        if f_mm is not None and rot_matrix is not None:
            strict_post = solve_strict_mode(
                lines_data,
                f_mm,
                "Constrained solve: focal refined",
                "Constrained solve: focal locked (insufficient constraints)",
                allow_focal_refine=True,
                rot_seed=rot_matrix,
            )
            if strict_post is not None:
                strict_result = strict_post.get('strict_result', {})
                strict_residual = float(strict_result.get('residual', float('inf')))

                hybrid_f_pixels = utils.get_effective_f_pixels(
                    f_mm,
                    cam.data.sensor_width,
                    cam.data.sensor_height,
                    cam.data.sensor_fit,
                    pixel_res_x,
                    pixel_res_y,
                )
                hybrid_residual = utils.compute_rotation_constraint_residual(lines_data, rot_matrix, hybrid_f_pixels)
                improvement = hybrid_residual - strict_residual

                # 用户手动拖拽地平线产生的偏移属于"强制俯仰覆盖"，必须优先于
                # 基于线条的 strict 解，否则松手后俯仰会被瞬间回退（strict 解
                # 以原始线条为尊，会把偏移带来的俯仰还原掉）。
                horizon_override_active = abs(scene.cmp_data.horizon_offset_px) > 1e-9

                improved_enough = (
                    not horizon_override_active
                    and np.isfinite(hybrid_residual)
                    and np.isfinite(strict_residual)
                    and (
                        strict_residual <= hybrid_residual * 0.985
                        or improvement >= 0.0015
                        or (
                            strict_result.get('focal_state') == 'refined'
                            and strict_residual <= hybrid_residual + 0.001
                        )
                    )
                )

                if improved_enough:
                    f_mm = strict_post['f_mm']
                    rot_matrix = strict_post['rot_matrix']
                    shift_x = strict_post['shift_x']
                    shift_y = strict_post['shift_y']
                    loc_orbit = strict_post['loc_orbit']
                    solve_mode_hint_key = strict_post['hint_key']

        if f_mm is None or rot_matrix is None:
            strict_solution = solve_strict_mode(
                lines_data,
                cam.data.lens,
                "Constrained solve: focal refined",
                "Constrained solve: focal locked (insufficient constraints)",
            )
            if strict_solution is None:
                return False, iface_("Solving failed. Check line placement.")

            f_mm = strict_solution['f_mm']
            rot_matrix = strict_solution['rot_matrix']
            shift_x = strict_solution['shift_x']
            shift_y = strict_solution['shift_y']
            loc_orbit = strict_solution['loc_orbit']
            solve_mode_hint_key = strict_solution['hint_key']

    # 稳定性检查
    if f_mm is None:
        iface_ = bpy.app.translations.pgettext_iface
        return False, iface_("Math solving failed")

    # 数值有效性检查
    if not np.isfinite(f_mm):
        iface_ = bpy.app.translations.pgettext_iface
        return False, iface_("Invalid focal length value")

    if not (1.0 < f_mm < 10000.0):
        iface_ = bpy.app.translations.pgettext_iface
        return False, iface_("Abnormal focal length: ") + f"{f_mm:.1f}mm"

    if abs(shift_x) < 1e-9 and abs(shift_y) < 1e-9:
        shift_x = current_shift_x
        shift_y = current_shift_y

    # 检查 shift 是否有效
    if not (np.isfinite(shift_x) and np.isfinite(shift_y)):
        shift_x = 0.0
        shift_y = 0.0
        print("[CameraMatch] Warning: Invalid shift values, reset to 0")

    if abs(shift_x) > 10.0 or abs(shift_y) > 10.0:
        shift_x = 0.0
        shift_y = 0.0
        print("[CameraMatch] Warning: Shift overflow, reset to 0")

    # 检查旋转矩阵有效性
    if rot_matrix is not None:
        try:
            # 检查矩阵是否包含无效值
            mat_array = np.array(rot_matrix)
            if not np.all(np.isfinite(mat_array)):
                iface_ = bpy.app.translations.pgettext_iface
                return False, iface_("Invalid rotation matrix")
        except Exception as e:
            print(f"[CameraMatch] Matrix validation error: {e}")

    # 检查位置有效性
    if loc_orbit is not None:
        try:
            loc_array = np.array(loc_orbit)
            if not np.all(np.isfinite(loc_array)):
                # 回退到当前位置
                loc_orbit = cam.location.copy()
                print("[CameraMatch] Warning: Invalid location, using current position")
        except Exception as e:
            print(f"[CameraMatch] Location validation error: {e}")
            loc_orbit = cam.location.copy()

    # 4. 应用
    view_state = utils.capture_camera_view_state(context)
    try:
        properties.suppress_horizon_updates()
        cam.data.lens = f_mm
        cam.data.shift_x = shift_x
        cam.data.shift_y = shift_y

        new_rot_4x4 = rot_matrix.to_4x4()

        cam.matrix_world = mathutils.Matrix.Translation(loc_orbit) @ new_rot_4x4
        context.view_layer.update()

        target_uv_for_shift = target_cursor_uv
        if horizon_lock_active and horizon_target_uv is not None:
            target_uv_for_shift = horizon_target_uv

        if target_uv_for_shift is not None:
            try:
                utils.compensate_shift_for_target_uv(
                    scene,
                    cam,
                    cursor_location,
                    target_uv_for_shift,
                    update_view_layer=context.view_layer.update,
                    iterations=2,
                )

                if np.isfinite(cam.data.shift_x) and np.isfinite(cam.data.shift_y):
                    shift_x = float(cam.data.shift_x)
                    shift_y = float(cam.data.shift_y)

            except Exception as e:
                print(f"[CameraMatch] Cursor lock correction failed: {e}")

        scene.cmp_data.last_world_rotation = 0.0
        scene.cmp_data.last_flip_z = False
        scene.cmp_data.world_rotation = 0.0
        scene.cmp_data.flip_z_axis = False

        if current_world_rotation != 0.0:
            scene.cmp_data.world_rotation = current_world_rotation
        if current_flip_z:
            scene.cmp_data.flip_z_axis = current_flip_z

        iface_ = bpy.app.translations.pgettext_iface
        mode_hint = ""
        if solve_mode_hint_key:
            mode_hint = " " + iface_(solve_mode_hint_key)
        if parallel_axes:
            mode_hint += " " + iface_("Parallel lines (no vanishing point) on axis ") + "/".join(parallel_axes)

        msg = iface_("Success: ") + f"f={f_mm:.1f}mm," + iface_(" Shift=") + f"({shift_x:.2f}, {shift_y:.2f})" + mode_hint

    except Exception as e:
        print(f"[CameraMatch] Apply transform failed: {e}")
        iface_ = bpy.app.translations.pgettext_iface
        return False, iface_("Failed to apply camera transform")
    finally:
        properties.resume_horizon_updates()
        utils.restore_camera_view_state(view_state)

    # 自洽性收尾：解算用的主点是由**解算前**的 shift 决定的，而上面的补偿又
    # 改写了 shift。若不重解，线段坐标的含义与最终 shift 就对不上，结果会稳定
    # 停在"依赖操作历史"的解上（实测初始 shift_x=0.05 时焦距偏 1.55%，
    # 0.10 时偏 2.77%）。这里用新的 shift 再解一次；shift 变化量本身会随重解
    # 迅速收敛，因此只重试一轮即可。
    shift_drift = max(abs(shift_x - current_shift_x), abs(shift_y - current_shift_y))
    if _retry < 1 and np.isfinite(shift_drift) and shift_drift > 1e-4:
        return solve_camera_core(context, fast=fast, _retry=_retry + 1)

    return True, msg


class CMP_OT_MatchCamera(bpy.types.Operator):
    """Solve camera based on drawn lines"""
    bl_idname = "cmp.match_camera"
    bl_label = "Match Camera"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        success, msg = solve_camera_core(context)
        if success:
            context.view_layer.update()
            self.report({'INFO'}, msg)
            print(f"[CameraMatchPro] {msg}")
            return {'FINISHED'}
        else:
            self.report({'WARNING'}, msg)
            return {'CANCELLED'}


class CMP_OT_MatchBackgroundResolution(bpy.types.Operator):
    """Match render resolution to the camera background image resolution"""
    bl_idname = "cmp.match_background_resolution"
    bl_label = "Match Background Resolution"
    bl_options = {'REGISTER', 'UNDO'}

    def _notify_no_background(self, context, msg):
        iface_ = bpy.app.translations.pgettext_iface
        msg = iface_(msg)
        self.report({'WARNING'}, msg)
        print(f"[CameraMatch] {msg}")

        def draw(menu, _ctx):
            menu.layout.label(text=msg)

        # 仅在有窗口的 GUI 会话中弹窗；后台/无窗口模式调用 popup_menu 会段错误
        if not bpy.app.background:
            try:
                wm = getattr(context, "window_manager", None) or bpy.context.window_manager
                if wm is not None:
                    wm.popup_menu(draw, title="Simple Camera Match", icon='INFO')
            except Exception:
                pass

    def execute(self, context):
        scene = getattr(context, "scene", None)
        if scene is None:
            self.report({'ERROR'}, "No scene")
            return {'CANCELLED'}

        cam = scene.camera
        if cam is None or getattr(cam, "data", None) is None:
            self._notify_no_background(context, "No camera or no background image on active camera")
            return {'CANCELLED'}

        bg_image = None
        for item in cam.data.background_images:
            if item.image is not None:
                bg_image = item.image
                break

        if bg_image is None:
            self._notify_no_background(context, "No background image on active camera")
            return {'CANCELLED'}

        w, h = int(bg_image.size[0]), int(bg_image.size[1])
        if w <= 0 or h <= 0:
            self._notify_no_background(context, "Background image has no valid size")
            return {'CANCELLED'}

        render = scene.render
        render.resolution_x = w
        render.resolution_y = h
        render.resolution_percentage = 100

        iface_ = bpy.app.translations.pgettext_iface
        msg = iface_("Render resolution set to ") + "%dx%d" % (w, h)
        self.report({'INFO'}, msg)
        print(f"[CameraMatch] {msg}")
        return {'FINISHED'}


def register():
    utils.register_class_safe(CMP_OT_MatchCamera)
    utils.register_class_safe(CMP_OT_MatchBackgroundResolution)


def unregister():
    utils.unregister_class_safe(CMP_OT_MatchBackgroundResolution)
    utils.unregister_class_safe(CMP_OT_MatchCamera)
