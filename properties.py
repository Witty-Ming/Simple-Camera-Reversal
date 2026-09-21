import bpy
import numpy as np
import bpy_extras
from types import SimpleNamespace

from . import utils


_horizon_update_suppress_depth = 0


def suppress_horizon_updates():
    global _horizon_update_suppress_depth
    _horizon_update_suppress_depth += 1


def resume_horizon_updates():
    global _horizon_update_suppress_depth
    _horizon_update_suppress_depth = max(0, _horizon_update_suppress_depth - 1)


def is_horizon_updates_suppressed():
    return _horizon_update_suppress_depth > 0


def reset_horizon_update_state():
    global _horizon_update_suppress_depth
    _horizon_update_suppress_depth = 0


def _context_for_scene(scene, context):
    if scene is None:
        return None

    if context is not None and getattr(context, "scene", None) == scene:
        return context

    try:
        view_layer = scene.view_layers[0] if len(scene.view_layers) > 0 else None
    except Exception:
        view_layer = None

    if view_layer is None:
        return None

    return SimpleNamespace(
        scene=scene,
        view_layer=view_layer,
        screen=None,
        area=None,
        space_data=None,
        region=None,
        region_data=None,
    )


def _solve_horizon_from_context(context):
    scene = getattr(context, "scene", None)
    if scene is None or scene.camera is None:
        return

    if is_horizon_updates_suppressed():
        return

    cmp_data = getattr(scene, "cmp_data", None)
    # 门槛与 operators.solve_camera_core 保持一致（单点透视 1 条线即可解算）
    if cmp_data is None or len(cmp_data.lines) < 1:
        return

    try:
        from . import operators
        operators.solve_camera_core(context)
    except Exception as e:
        print(f"[CameraMatch] 地平线解算失败: {e}")


class CMP_Line(bpy.types.PropertyGroup):
    start: bpy.props.FloatVectorProperty(size=2, description="Start Point (Normalized)")
    end: bpy.props.FloatVectorProperty(size=2, description="End Point (Normalized)")
    axis: bpy.props.StringProperty(default='X', description="Axis (X, Y, Z)")


class CMP_SceneProperties(bpy.types.PropertyGroup):
    lines: bpy.props.CollectionProperty(type=CMP_Line)
    active_index: bpy.props.IntProperty(default=-1)
    lines_camera: bpy.props.PointerProperty(type=bpy.types.Object, description="Camera bound to current guide lines")

    is_drawing_mode: bpy.props.BoolProperty(default=False)
    is_creating_line: bpy.props.BoolProperty(default=False)

    last_world_rotation: bpy.props.FloatProperty(default=0.0)
    last_flip_z: bpy.props.BoolProperty(default=False)

    def _update_view_layer(self):
        scene = getattr(self, "id_data", None)
        solve_context = _context_for_scene(scene, bpy.context)
        if solve_context is None:
            return
        try:
            solve_context.view_layer.update()
        except Exception:
            pass

    def _compensate_shift_for_cursor_uv(self, scene, cam, target_uv):
        # 算法已统一到 utils.compensate_shift_for_target_uv（原先与 operators.py 里
        # 的解算后补偿逻辑是两份几乎逐行重复的实现）。
        utils.compensate_shift_for_target_uv(
            scene,
            cam,
            scene.cursor.location.copy(),
            target_uv,
            update_view_layer=self._update_view_layer,
            iterations=3,
        )

    def get_focal_length_mm(self):
        scene = getattr(self, "id_data", None)
        cam = getattr(scene, "camera", None)
        if cam is None:
            return 50.0
        return float(cam.data.lens)

    def set_focal_length_mm(self, value):
        scene = getattr(self, "id_data", None)
        cam = getattr(scene, "camera", None)
        if scene is None or cam is None:
            return

        try:
            new_lens = float(value)
        except (TypeError, ValueError):
            return
        # 属性右键 "Reset to Default Value" 会以 0.0 调用 setter，
        # 直接忽略非法值，避免把焦距打到 1mm 这种极端状态。
        if not np.isfinite(new_lens) or new_lens <= 0.0:
            return
        new_lens = min(max(new_lens, 1.0), 10000.0)

        old_lens = float(cam.data.lens)
        if abs(new_lens - old_lens) < 1e-6:
            return

        target_uv = None
        cursor_location = scene.cursor.location.copy()
        try:
            cursor_view = bpy_extras.object_utils.world_to_camera_view(scene, cam, cursor_location)
            if np.isfinite(cursor_view.x) and np.isfinite(cursor_view.y):
                target_uv = (float(cursor_view.x), float(cursor_view.y))
        except Exception:
            target_uv = None

        context = bpy.context
        view_state = None
        if getattr(context, "scene", None) == scene:
            view_state = utils.capture_camera_view_state(context)

        cam.data.lens = new_lens
        self._update_view_layer()

        if target_uv is not None:
            self._compensate_shift_for_cursor_uv(scene, cam, target_uv)

        if view_state is not None and getattr(context, "scene", None) == scene:
            utils.restore_camera_view_state(view_state)

    def get_hitchcock_focal_mm(self):
        scene = getattr(self, "id_data", None)
        cam = getattr(scene, "camera", None)
        if cam is None:
            return 50.0
        return float(cam.data.lens)

    def set_hitchcock_focal_mm(self, value):
        """
        希区柯克变焦滑块：改焦距的同时让相机沿光轴前后移动，
        使 3D 游标所在深度平面的构图（大小与位置）完全不变。

        滑块每次回调都以"当前 lens"为起点做增量 dolly，因此连续拖拽会
        从初始机位累积出一段连贯的滑动变焦，而不是每帧重新求解。
        """
        scene = getattr(self, "id_data", None)
        cam = getattr(scene, "camera", None)
        if scene is None or cam is None:
            return

        try:
            new_lens = float(value)
        except (TypeError, ValueError):
            return
        if not np.isfinite(new_lens) or new_lens <= 0.0:
            return

        context = bpy.context
        view_state = None
        if getattr(context, "scene", None) == scene:
            view_state = utils.capture_camera_view_state(context)

        try:
            ok, reason = utils.apply_hitchcock_zoom(scene, cam, new_lens)
            self._update_view_layer()
            if not ok or reason == 'degenerate':
                # 游标与相机重合/在相机后方时无法做 dolly，此时只改了焦距，
                # 构图会被破坏 —— 明确告诉用户而不是静默留下一个坏状态。
                iface_ = bpy.app.translations.pgettext_iface
                key = ("Hitchcock zoom unavailable: 3D Cursor is behind the camera"
                       if reason == 'degenerate' else "Hitchcock zoom failed")
                self.last_hitchcock_message = iface_(key)
            else:
                self.last_hitchcock_message = ""
        finally:
            if view_state is not None and getattr(context, "scene", None) == scene:
                utils.restore_camera_view_state(view_state)

    def update_rotation(self, context):
        import math
        import mathutils

        scene = getattr(self, "id_data", None)
        solve_context = _context_for_scene(scene, context)
        if solve_context is None:
            return

        cam = scene.camera
        if not cam:
            return

        pivot = scene.cursor.location.copy()
        view_state = utils.capture_camera_view_state(solve_context)

        # 用 try/finally 保证即使旋转/更新抛异常，也不会把用户的视口
        # view_camera_offset / view_camera_zoom 留在被改过的状态。
        try:
            delta_rot = self.world_rotation - self.last_world_rotation
            self.last_world_rotation = self.world_rotation

            if abs(delta_rot) > 1e-6:
                rot_mat = mathutils.Matrix.Rotation(delta_rot, 4, 'Z')
                cam.matrix_world = utils.rotate_matrix_around_point(cam.matrix_world, rot_mat, pivot)

            if self.flip_z_axis != self.last_flip_z:
                self.last_flip_z = self.flip_z_axis
                flip_mat = mathutils.Matrix.Rotation(math.pi, 4, 'X')
                cam.matrix_world = utils.rotate_matrix_around_point(cam.matrix_world, flip_mat, pivot)

            solve_context.view_layer.update()
        except Exception as e:
            print(f"[CameraMatch] 手动旋转微调失败: {e}")
        finally:
            utils.restore_camera_view_state(view_state)

    def update_horizon(self, context):
        if is_horizon_updates_suppressed():
            return

        scene = getattr(self, "id_data", None)
        if scene is None or scene.camera is None:
            return

        cmp_data = getattr(scene, "cmp_data", None)
        # 与 operators.solve_camera_core 的门槛保持一致
        if cmp_data is None or len(cmp_data.lines) < 1:
            return

        solve_context = _context_for_scene(scene, context)
        if solve_context is not None:
            _solve_horizon_from_context(solve_context)

    focal_length_mm: bpy.props.FloatProperty(
        name="Focal Length (mm)",
        description="Camera focal length in millimeters",
        default=50.0,
        min=1.0,
        max=10000.0,
        get=get_focal_length_mm,
        set=set_focal_length_mm,
    )

    hitchcock_focal_mm: bpy.props.FloatProperty(
        name="Hitchcock Zoom",
        description="Dolly zoom: change focal length while the camera moves along its own axis so the 3D Cursor plane keeps exactly the same framing",
        default=50.0,
        min=1.0,
        max=10000.0,
        soft_min=1.0,
        soft_max=2000.0,
        get=get_hitchcock_focal_mm,
        set=set_hitchcock_focal_mm,
    )

    last_hitchcock_message: bpy.props.StringProperty(default="")

    # 传感器宽度：直接改相机数据会让已解算的 lens/shift 与新传感器尺寸失去
    # 一致性，所以包一层 setter，写回后立刻重解一次。
    def get_sensor_width_mm(self):
        scene = getattr(self, "id_data", None)
        cam = getattr(scene, "camera", None)
        if cam is None or getattr(cam, "data", None) is None:
            return 36.0
        return float(cam.data.sensor_width)

    def set_sensor_width_mm(self, value):
        scene = getattr(self, "id_data", None)
        cam = getattr(scene, "camera", None)
        if scene is None or cam is None or getattr(cam, "data", None) is None:
            return
        try:
            new_value = float(value)
        except (TypeError, ValueError):
            return
        if not np.isfinite(new_value) or new_value <= 0.0:
            return
        if abs(new_value - float(cam.data.sensor_width)) < 1e-6:
            return

        cam.data.sensor_width = new_value
        self._update_view_layer()

        if len(getattr(self, "lines", [])) < 1:
            return
        try:
            from . import operators
            solve_context = _context_for_scene(scene, bpy.context)
            if solve_context is not None:
                operators.solve_camera_core(solve_context)
        except Exception as e:
            print(f"[CameraMatch] 传感器尺寸变更后重解算失败: {e}")

    sensor_width_mm: bpy.props.FloatProperty(
        name="Sensor (mm)",
        description="Camera sensor width in millimeters; changing it re-solves the camera",
        default=36.0,
        min=0.1,
        max=1000.0,
        get=get_sensor_width_mm,
        set=set_sensor_width_mm,
    )

    world_rotation: bpy.props.FloatProperty(
        name="3D Cursor Rotation",
        description="Rotate camera around 3D cursor",
        default=0.0,
        min=-3.1415926,
        max=3.1415926,
        subtype='ANGLE',
        unit='ROTATION',
        update=update_rotation
    )

    flip_z_axis: bpy.props.BoolProperty(
        name="Flip Z Axis",
        description="Flip world Z axis direction around 3D cursor (rotate 180 degrees around X axis)",
        default=False,
        update=update_rotation
    )

    horizon_enabled: bpy.props.BoolProperty(
        name="Enable Horizon",
        description="Enable horizon constraint from X/Y vanishing points",
        default=True,
        update=update_horizon
    )

    horizon_offset_px: bpy.props.FloatProperty(
        name="Horizon Offset",
        description="Move horizon up/down in pixel space",
        default=0.0,
        soft_min=-2000.0,
        soft_max=2000.0,
        update=update_horizon
    )




def register():
    reset_horizon_update_state()
    utils.register_class_safe(CMP_Line)
    utils.register_class_safe(CMP_SceneProperties)
    bpy.types.Scene.cmp_data = bpy.props.PointerProperty(type=CMP_SceneProperties)


def unregister():
    reset_horizon_update_state()

    if hasattr(bpy.types.Scene, "cmp_data"):
        del bpy.types.Scene.cmp_data

    utils.unregister_class_safe(CMP_SceneProperties)
    utils.unregister_class_safe(CMP_Line)
