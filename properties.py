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
    if cmp_data is None or len(cmp_data.lines) < 2:
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
        render = scene.render
        pixel_res_x, pixel_res_y = utils.get_effective_render_size(render)
        if pixel_res_x <= 1e-8 or pixel_res_y <= 1e-8:
            return

        cursor_location = scene.cursor.location.copy()
        shift_eps = 1e-4
        max_shift_step = 0.02

        for _ in range(3):
            cur_view = bpy_extras.object_utils.world_to_camera_view(scene, cam, cursor_location)
            if not (np.isfinite(cur_view.x) and np.isfinite(cur_view.y)):
                break

            err_u = target_uv[0] - float(cur_view.x)
            err_v = target_uv[1] - float(cur_view.y)
            err_px = np.hypot(err_u * pixel_res_x, err_v * pixel_res_y)
            if err_px < 0.25:
                break

            base_shift_x = float(cam.data.shift_x)
            base_shift_y = float(cam.data.shift_y)

            cam.data.shift_x = base_shift_x + shift_eps
            self._update_view_layer()
            view_sx = bpy_extras.object_utils.world_to_camera_view(scene, cam, cursor_location)

            cam.data.shift_x = base_shift_x
            cam.data.shift_y = base_shift_y + shift_eps
            self._update_view_layer()
            view_sy = bpy_extras.object_utils.world_to_camera_view(scene, cam, cursor_location)

            cam.data.shift_y = base_shift_y
            self._update_view_layer()

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
            self._update_view_layer()

            new_view = bpy_extras.object_utils.world_to_camera_view(scene, cam, cursor_location)
            if not (np.isfinite(new_view.x) and np.isfinite(new_view.y)):
                cam.data.shift_x = base_shift_x
                cam.data.shift_y = base_shift_y
                self._update_view_layer()
                break

            new_err_u = target_uv[0] - float(new_view.x)
            new_err_v = target_uv[1] - float(new_view.y)
            new_err_px = np.hypot(new_err_u * pixel_res_x, new_err_v * pixel_res_y)
            if (not np.isfinite(new_err_px)) or new_err_px >= err_px:
                cam.data.shift_x = base_shift_x
                cam.data.shift_y = base_shift_y
                self._update_view_layer()
                break

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

        new_lens = float(max(value, 1.0))
        if not np.isfinite(new_lens):
            return

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
        utils.restore_camera_view_state(view_state)

    def update_horizon(self, context):
        if is_horizon_updates_suppressed():
            return

        scene = getattr(self, "id_data", None)
        if scene is None or scene.camera is None:
            return

        cmp_data = getattr(scene, "cmp_data", None)
        if cmp_data is None or len(cmp_data.lines) < 2:
            return

        solve_context = _context_for_scene(scene, context)
        if solve_context is not None:
            _solve_horizon_from_context(solve_context)

    focal_length_mm: bpy.props.FloatProperty(
        name="Focal Length (mm)",
        description="Camera focal length in millimeters",
        min=1.0,
        max=10000.0,
        get=get_focal_length_mm,
        set=set_focal_length_mm,
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
