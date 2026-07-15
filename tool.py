import math
import mathutils
import bpy
import sys
import time
import numpy as np
import bpy_extras
from mathutils import Vector
from bpy_extras import view3d_utils

from . import utils, properties


class CMP_OT_DrawLine(bpy.types.Operator):
    """Draw 2D camera match reference lines (rebuilt as 3D)"""
    bl_idname = "cmp.draw_line"
    bl_label = "Draw Reference Line"
    bl_options = {'REGISTER', 'UNDO'}

    STATE_IDLE = 'IDLE'
    STATE_DRAWING = 'DRAWING'
    STATE_EDITING = 'EDITING'
    STATE_DRAGGING = 'DRAGGING'
    STATE_WAITING_DRAG = 'WAITING_DRAG'
    STATE_DRAG_HORIZON_OFFSET = 'DRAG_HORIZON_OFFSET'

    state = STATE_IDLE
    current_axis = 'X'
    active_handle = -1
    drag_start_x = 0
    drag_start_y = 0
    DRAG_THRESHOLD = 10
    SOLVE_INTERVAL = 1.0 / 28.0
    HORIZON_HANDLE_RADIUS = 14.0
    FINE_TUNE_FACTOR = 0.1

    def invoke(self, context, event):
        if not context.scene.camera:
            self.report({'WARNING'}, "Please add a camera first")
            return {'CANCELLED'}

        if getattr(context.scene.camera.data, "type", None) != 'PERSP':
            self.report({'WARNING'}, "Only perspective cameras are supported")
            return {'CANCELLED'}

        if not utils.is_camera_view(context):
            self.report({'WARNING'}, "Please switch to Camera View first")
            return {'CANCELLED'}

        self._cleanup_done = False
        self.state = self.STATE_IDLE
        self.current_axis = 'X'
        self.last_error = ""
        self.undo_stack = []
        self.redo_stack = []
        self.drag_history = None
        self.last_solve_ts = 0.0
        self.horizon_drag_start_offset = 0.0
        self.horizon_drag_start_mouse_render = None
        self.horizon_drag_shift_state = None
        self.horizon_drag_updates_suppressed = False
        self.horizon_drag_start_camera_matrix = None
        self.horizon_drag_start_f_pixels = None
        self.horizon_drag_start_normal_render = None
        self.draw_anchor_norm = None
        self.draw_anchor_value_norm = None
        self.draw_shift_state = None
        self.line_drag_start_mouse_norm = None
        self.line_drag_start_value_norm = None
        self.line_drag_shift_state = None
        self.draw_axis_constraint = None

        cmp_data = context.scene.cmp_data
        cmp_data.active_index = -1
        cmp_data.is_drawing_mode = True

        if len(cmp_data.lines) == 0:
            cmp_data.lines_camera = context.scene.camera
        elif cmp_data.lines_camera is not None and cmp_data.lines_camera != context.scene.camera:
            self.clear_all_lines(context, push_history=False, run_solve=False)
            self.undo_stack.clear()
            self.redo_stack.clear()
            self.last_error = "Active camera changed, guide lines cleared"
        try:
            from . import gpu_draw
            gpu_draw.register()
        except Exception:
            pass

        context.window_manager.modal_handler_add(self)
        self.update_header(context)
        context.area.tag_redraw()
        return {'RUNNING_MODAL'}

    def primary_modifier_pressed(self, event):
        return event.ctrl or getattr(event, "oskey", False)

    def primary_modifier_label(self):
        return 'Cmd' if sys.platform == 'darwin' else 'Ctrl'

    def update_header(self, context):
        try:
            iface_ = bpy.app.translations.pgettext_iface
            cols = {'X': iface_('Red X'), 'Y': iface_('Green Y'), 'Z': iface_('Blue Z')}
            c = cols.get(self.current_axis, "")
            mod = self.primary_modifier_label()
            base = iface_("CameraMatch [3D]: Axis %s (1/2/3). Drag to draw | Click dot to edit | Shift fine tune | Draw: X lock horizontal / Y lock vertical (toggle) | %s+Z undo | %s+Shift+Z redo | Alt+X clear all | Esc / Right click exit") % (c, mod, mod)

            if context.scene.cmp_data.lines_camera is not None and context.scene.cmp_data.lines_camera != context.scene.camera:
                camera_hint = iface_(" | Camera changed: press Alt+X to clear old guides")
                base += camera_hint

            if self.last_error:
                msg = base + iface_(" | Error: ") + iface_(self.last_error)
            else:
                msg = base

            context.area.header_text_set(msg)
        except Exception:
            pass

    def apply_draw_axis_constraint(self, start_norm, end_norm):
        if start_norm is None or end_norm is None:
            return end_norm
        if self.draw_axis_constraint == 'X':
            return Vector((float(end_norm[0]), float(start_norm[1])))
        if self.draw_axis_constraint == 'Y':
            return Vector((float(start_norm[0]), float(end_norm[1])))
        return end_norm

    def resolve_dragged_point(self, context, x, y, anchor_mouse_norm, anchor_value_norm, use_fine_tune=False):
        norm = self.screen_to_norm(context, x, y)
        if norm is None:
            return None
        if not use_fine_tune or anchor_mouse_norm is None or anchor_value_norm is None:
            return norm
        dx = float(norm[0]) - float(anchor_mouse_norm[0])
        dy = float(norm[1]) - float(anchor_mouse_norm[1])
        return Vector((
            float(anchor_value_norm[0]) + dx * self.FINE_TUNE_FACTOR,
            float(anchor_value_norm[1]) + dy * self.FINE_TUNE_FACTOR,
        ))

    def refresh_drawing_endpoint(self, context, x, y, use_fine_tune=False):
        lines = context.scene.cmp_data.lines
        if not lines:
            return
        start = lines[-1].start
        end = self.resolve_dragged_point(
            context,
            x,
            y,
            self.draw_anchor_norm,
            self.draw_anchor_value_norm,
            use_fine_tune=use_fine_tune,
        )
        if end is None:
            return
        end = self.apply_draw_axis_constraint(start, end)
        lines[-1].end = end
        self.trigger_solve(context)

    def refresh_dragging_endpoint(self, context, x, y, idx, use_fine_tune=False):
        lines = context.scene.cmp_data.lines
        if idx == -1 or idx >= len(lines):
            return
        line = lines[idx]
        point = self.resolve_dragged_point(
            context,
            x,
            y,
            self.line_drag_start_mouse_norm,
            self.line_drag_start_value_norm,
            use_fine_tune=use_fine_tune,
        )
        if point is None:
            return
        if self.active_handle == 0:
            line.start = point
        else:
            line.end = point
        self.trigger_solve(context)

    def modal(self, context, event):
        scene = getattr(context, "scene", None)
        area = getattr(context, "area", None)
        cmp_data = getattr(scene, "cmp_data", None) if scene is not None else None
        if area is None or cmp_data is None:
            self.quit(context)
            return {'CANCELLED'}

        cam = scene.camera
        if cam is None or getattr(cam.data, "type", None) != 'PERSP':
            self.quit(context)
            return {'CANCELLED'}

        area.tag_redraw()

        if event.type in {'RIGHTMOUSE', 'ESC'}:
            self.quit(context)
            return {'FINISHED'}

        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
            return {'PASS_THROUGH'}

        # 实现在绘制模式下切换相机时，自动进行数据清理和重绑，彻底告别报错弹窗
        if cmp_data.lines_camera is not None and cmp_data.lines_camera != context.scene.camera:
            self.clear_all_lines(context, push_history=False, run_solve=False)
            self.undo_stack.clear()
            self.redo_stack.clear()
            self.last_error = ""
            self.update_header(context)

        if not utils.is_camera_view(context):
            self.last_error = "Please stay in Camera View while drawing"
            self.update_header(context)
            return {'RUNNING_MODAL'}

        if event.value == 'PRESS':
            if self.primary_modifier_pressed(event) and event.shift and event.type == 'Z':
                self.redo(context)
                return {'RUNNING_MODAL'}

            if self.primary_modifier_pressed(event) and event.type == 'Z':
                self.undo(context)
                return {'RUNNING_MODAL'}

            if event.alt and event.type == 'X':
                self.clear_all_lines(context)
                return {'RUNNING_MODAL'}

            if event.type in {'ONE', 'NUMPAD_1'}:
                self.current_axis = 'X'
                self.update_header(context)
            elif event.type in {'TWO', 'NUMPAD_2'}:
                self.current_axis = 'Y'
                self.update_header(context)
            elif event.type in {'THREE', 'NUMPAD_3'}:
                self.current_axis = 'Z'
                self.update_header(context)
            elif event.type in {'X', 'Y'} and self.state in {self.STATE_DRAWING, self.STATE_WAITING_DRAG}:
                target = event.type
                self.draw_axis_constraint = None if self.draw_axis_constraint == target else target
                if self.state == self.STATE_DRAWING:
                    self.refresh_drawing_endpoint(
                        context,
                        event.mouse_region_x,
                        event.mouse_region_y,
                        use_fine_tune=bool(event.shift),
                    )
            elif event.type == 'X':
                if self.state == self.STATE_EDITING:
                    lines = context.scene.cmp_data.lines
                    idx = context.scene.cmp_data.active_index
                    if 0 <= idx < len(lines):
                        self.push_history(context)
                        lines.remove(idx)
                        self.reset_horizon_manual_offset(context)
                        context.scene.cmp_data.active_index = -1
                        self.state = self.STATE_IDLE
                        self.trigger_solve(context)

        x, y = event.mouse_region_x, event.mouse_region_y
        lines = context.scene.cmp_data.lines
        idx = context.scene.cmp_data.active_index

        if event.type == 'MOUSEMOVE':
            if self.state == self.STATE_DRAWING:
                if self.draw_shift_state is None:
                    self.draw_shift_state = bool(event.shift)
                elif bool(event.shift) != self.draw_shift_state:
                    self.draw_shift_state = bool(event.shift)
                    anchor = self.screen_to_norm(context, x, y)
                    if anchor is not None and lines:
                        self.draw_anchor_norm = anchor
                        self.draw_anchor_value_norm = Vector(lines[-1].end)
                self.refresh_drawing_endpoint(context, x, y, use_fine_tune=bool(event.shift))
            elif self.state == self.STATE_DRAGGING:
                if self.line_drag_shift_state is None:
                    self.line_drag_shift_state = bool(event.shift)
                elif bool(event.shift) != self.line_drag_shift_state:
                    self.line_drag_shift_state = bool(event.shift)
                    anchor = self.screen_to_norm(context, x, y)
                    if anchor is not None and idx != -1 and idx < len(lines):
                        self.line_drag_start_mouse_norm = anchor
                        line = lines[idx]
                        self.line_drag_start_value_norm = Vector(line.start if self.active_handle == 0 else line.end)
                self.refresh_dragging_endpoint(context, x, y, idx, use_fine_tune=bool(event.shift))
            elif self.state == self.STATE_DRAG_HORIZON_OFFSET:
                self.update_horizon_drag(context, x, y, event_shift=bool(event.shift))
            elif self.state == self.STATE_WAITING_DRAG:
                dist = np.hypot(x - self.drag_start_x, y - self.drag_start_y)
                if dist > self.DRAG_THRESHOLD:
                    self.start_drawing(context, self.drag_start_x, self.drag_start_y)
                    self.draw_shift_state = bool(event.shift)
                    self.refresh_drawing_endpoint(context, x, y, use_fine_tune=bool(event.shift))

        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            if self.state in {self.STATE_IDLE, self.STATE_EDITING}:
                horizon_handle = self.check_horizon_handle_click(context, x, y)
                if horizon_handle is not None:
                    self.begin_horizon_drag(context, horizon_handle, x, y)
                    return {'RUNNING_MODAL'}

            if self.state == self.STATE_IDLE:
                dot = self.check_dot_click(context, x, y)
                if dot != -1:
                    context.scene.cmp_data.active_index = dot
                    self.state = self.STATE_EDITING
                else:
                    self.drag_start_x = x
                    self.drag_start_y = y
                    self.state = self.STATE_WAITING_DRAG
                    self.draw_axis_constraint = None
            elif self.state == self.STATE_EDITING:
                h = self.check_endpoint_click(context, x, y, idx)
                if h != -1:
                    self.active_handle = h
                    self.begin_drag_history(context, idx)
                    self.reset_horizon_manual_offset(context)
                    self.state = self.STATE_DRAGGING
                    self.line_drag_shift_state = bool(event.shift)
                    self.line_drag_start_mouse_norm = self.screen_to_norm(context, x, y)
                    if idx != -1 and idx < len(lines):
                        line = lines[idx]
                        self.line_drag_start_value_norm = Vector(line.start if h == 0 else line.end)
                else:
                    dot = self.check_dot_click(context, x, y)
                    if dot != -1:
                        context.scene.cmp_data.active_index = dot
                    else:
                        context.scene.cmp_data.active_index = -1
                        self.state = self.STATE_IDLE

        elif event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            if self.state == self.STATE_DRAWING:
                self.state = self.STATE_IDLE
                context.scene.cmp_data.active_index = -1
                context.scene.cmp_data.is_creating_line = False
                self.draw_axis_constraint = None
                self.draw_anchor_norm = None
                self.draw_anchor_value_norm = None
                self.draw_shift_state = None
                self.trigger_solve(context, force=True)
            elif self.state == self.STATE_WAITING_DRAG:
                self.state = self.STATE_IDLE
                context.scene.cmp_data.active_index = -1
                self.draw_axis_constraint = None
            elif self.state == self.STATE_DRAGGING:
                self.finish_drag_history(context)
                self.state = self.STATE_EDITING
                self.line_drag_start_mouse_norm = None
                self.line_drag_start_value_norm = None
                self.line_drag_shift_state = None
                self.trigger_solve(context, force=True)
            elif self.state == self.STATE_DRAG_HORIZON_OFFSET:
                self.finish_drag_history(context)
                self.state = self.STATE_IDLE
                self.horizon_drag_shift_state = None
                self.end_horizon_drag_updates()
                self.trigger_solve(context, force=True)

        return {'RUNNING_MODAL'}

    def get_horizon_geometry(self, context):
        cmp_data = getattr(context.scene, "cmp_data", None)
        if cmp_data is None or not cmp_data.horizon_enabled:
            return None
        if len(cmp_data.lines) < 2:
            return None

        render = context.scene.render
        pixel_res_x, pixel_res_y = utils.get_effective_render_size(render)
        return utils.compute_horizon_overlay_geometry(
            cmp_data.lines,
            cmp_data,
            pixel_res_x,
            pixel_res_y,
            context.region.width,
            context.region.height,
            context=context,
        )


    def check_horizon_handle_click(self, context, x, y):
        geo = self.get_horizon_geometry(context)
        if geo is None:
            return None

        mouse = np.array([float(x), float(y)], dtype=float)
        offset_handle = np.array(geo['offset_handle_region'], dtype=float)

        if np.linalg.norm(mouse - offset_handle) <= self.HORIZON_HANDLE_RADIUS:
            return 'OFFSET'
        return None

    def begin_horizon_drag(self, context, handle, x, y):
        geo = self.get_horizon_geometry(context)
        if geo is None:
            return

        cmp_data = context.scene.cmp_data
        render = context.scene.render
        pixel_res_x, pixel_res_y = utils.get_effective_render_size(render)

        self.horizon_drag_start_offset = float(cmp_data.horizon_offset_px)
        self.drag_history = self.state_to_snapshot(context)

        if handle != 'OFFSET':
            return

        self.state = self.STATE_DRAG_HORIZON_OFFSET
        self.horizon_drag_start_mouse_render = utils.region_xy_to_render_centered_px(
            (float(x), float(y)),
            pixel_res_x,
            pixel_res_y,
            context.region.width,
            context.region.height,
        )
        start_normal = np.array(geo['horizon']['normal'], dtype=float)
        start_normal_norm = np.linalg.norm(start_normal)
        if start_normal_norm > 1e-8:
            start_normal = start_normal / start_normal_norm
            self.horizon_drag_start_normal_render = start_normal
        else:
            self.horizon_drag_start_normal_render = np.array([0.0, 1.0], dtype=float)

        if not self.horizon_drag_updates_suppressed:
            properties.suppress_horizon_updates()
            self.horizon_drag_updates_suppressed = True
        self.horizon_drag_start_camera_matrix = context.scene.camera.matrix_world.copy()
        self.horizon_drag_start_f_pixels = utils.get_effective_f_pixels(
            context.scene.camera.data.lens,
            context.scene.camera.data.sensor_width,
            context.scene.camera.data.sensor_height,
            context.scene.camera.data.sensor_fit,
            pixel_res_x,
            pixel_res_y,
        )

        context.scene.cmp_data.active_index = -1

    def update_horizon_drag(self, context, x, y, event_shift=False):
        if self.state != self.STATE_DRAG_HORIZON_OFFSET:
            return

        cmp_data = context.scene.cmp_data
        use_fine_tune = bool(event_shift)

        if self.horizon_drag_shift_state is None:
            self.horizon_drag_shift_state = use_fine_tune
        elif use_fine_tune != self.horizon_drag_shift_state:
            self.horizon_drag_shift_state = use_fine_tune
            render = context.scene.render
            pixel_res_x, pixel_res_y = utils.get_effective_render_size(render)
            mouse_render = utils.region_xy_to_render_centered_px(
                (float(x), float(y)),
                pixel_res_x,
                pixel_res_y,
                context.region.width,
                context.region.height,
            )
            self.horizon_drag_start_mouse_render = mouse_render
            self.horizon_drag_start_offset = float(cmp_data.horizon_offset_px)

        render = context.scene.render
        pixel_res_x, pixel_res_y = utils.get_effective_render_size(render)

        mouse_render = utils.region_xy_to_render_centered_px(
            (float(x), float(y)),
            pixel_res_x,
            pixel_res_y,
            context.region.width,
            context.region.height,
        )

        if self.horizon_drag_start_mouse_render is None:
            self.horizon_drag_start_mouse_render = mouse_render

        normal_render = self.horizon_drag_start_normal_render
        if normal_render is None or len(normal_render) != 2:
            normal_render = np.array([0.0, 1.0], dtype=float)

        norm_len = np.linalg.norm(normal_render)
        if norm_len <= 1e-8:
            normal_render = np.array([0.0, 1.0], dtype=float)
        else:
            normal_render = np.array(normal_render, dtype=float) / norm_len

        delta_render = mouse_render - self.horizon_drag_start_mouse_render
        delta_offset = float(np.dot(delta_render, normal_render))
        if use_fine_tune:
            delta_offset *= self.FINE_TUNE_FACTOR

        cmp_data.horizon_offset_px = self.horizon_drag_start_offset + delta_offset
        self.apply_horizon_drag_camera_preview(context)

    def apply_horizon_drag_camera_preview(self, context):
        if self.state != self.STATE_DRAG_HORIZON_OFFSET:
            return

        scene = context.scene
        cam = scene.camera
        if cam is None:
            return

        start_matrix = self.horizon_drag_start_camera_matrix
        if start_matrix is None:
            start_matrix = cam.matrix_world.copy()
            self.horizon_drag_start_camera_matrix = start_matrix.copy()

        render = scene.render
        pixel_res_x, pixel_res_y = utils.get_effective_render_size(render)

        f_pixels = self.horizon_drag_start_f_pixels
        if f_pixels is None or not np.isfinite(f_pixels) or f_pixels <= 1e-8:
            f_pixels = utils.get_effective_f_pixels(
                cam.data.lens,
                cam.data.sensor_width,
                cam.data.sensor_height,
                cam.data.sensor_fit,
                pixel_res_x,
                pixel_res_y,
            )

        if f_pixels is None or not np.isfinite(f_pixels) or f_pixels <= 1e-8:
            return

        delta_offset = float(scene.cmp_data.horizon_offset_px - self.horizon_drag_start_offset)
        if not np.isfinite(delta_offset):
            return

        delta_pitch = float(np.arctan2(delta_offset, max(float(f_pixels), 1e-8)))
        if not np.isfinite(delta_pitch):
            return

        if abs(delta_pitch) <= 1e-9:
            new_matrix = start_matrix.copy()
        else:
            normal_render = self.horizon_drag_start_normal_render
            if normal_render is not None and len(normal_render) == 2:
                local_axis = mathutils.Vector((float(normal_render[1]), float(-normal_render[0]), 0.0))
            else:
                local_axis = mathutils.Vector((1.0, 0.0, 0.0))
                
            rot_axis_world = start_matrix.to_3x3() @ local_axis
            if rot_axis_world.length > 1e-8:
                rot_axis_world.normalize()
            else:
                rot_axis_world = mathutils.Vector((1.0, 0.0, 0.0))
                
            rot_obj = mathutils.Matrix.Rotation(-delta_pitch, 4, rot_axis_world)
            new_matrix = utils.rotate_matrix_around_point(
                start_matrix,
                rot_obj,
                scene.cursor.location.copy(),
            )

        view_state = utils.capture_camera_view_state(context)
        cam.matrix_world = new_matrix
        context.view_layer.update()
        utils.restore_camera_view_state(view_state)

    def end_horizon_drag_updates(self):
        if not self.horizon_drag_updates_suppressed:
            return
        self.horizon_drag_updates_suppressed = False
        self.horizon_drag_start_camera_matrix = None
        self.horizon_drag_start_f_pixels = None
        self.horizon_drag_start_mouse_render = None
        self.horizon_drag_start_normal_render = None
        self.horizon_drag_shift_state = None
        properties.resume_horizon_updates()

    def line_to_dict(self, line):
        return {
            'start': (float(line.start[0]), float(line.start[1])),
            'end': (float(line.end[0]), float(line.end[1])),
            'axis': str(line.axis),
        }

    def camera_to_snapshot(self, camera):
        if camera is None or getattr(camera, "data", None) is None:
            return None

        return {
            'name': str(camera.name),
            'matrix_world': tuple(
                tuple(float(value) for value in row)
                for row in camera.matrix_world
            ),
            'lens': float(camera.data.lens),
            'shift_x': float(camera.data.shift_x),
            'shift_y': float(camera.data.shift_y),
        }

    def state_to_snapshot(self, context):
        cmp_data = context.scene.cmp_data
        lines_data = [self.line_to_dict(line) for line in cmp_data.lines]
        return {
            'lines': lines_data,
            'active_index': int(cmp_data.active_index),
            'lines_camera_name': cmp_data.lines_camera.name if cmp_data.lines_camera is not None else None,
            'camera': self.camera_to_snapshot(context.scene.camera),
            'horizon_enabled': bool(cmp_data.horizon_enabled),
            'horizon_offset_px': float(cmp_data.horizon_offset_px),
            'world_rotation': float(cmp_data.world_rotation),
            'last_world_rotation': float(cmp_data.last_world_rotation),
            'flip_z_axis': bool(cmp_data.flip_z_axis),
            'last_flip_z': bool(cmp_data.last_flip_z),
        }

    def restore_snapshot(self, context, snapshot):
        if snapshot is None:
            return

        cmp_data = context.scene.cmp_data
        view_state = utils.capture_camera_view_state(context)
        properties.suppress_horizon_updates()
        try:
            cmp_data.lines.clear()
            for item in snapshot.get('lines', []):
                line = cmp_data.lines.add()
                start = item.get('start', (0.0, 0.0))
                end = item.get('end', (0.0, 0.0))
                line.start = (float(start[0]), float(start[1]))
                line.end = (float(end[0]), float(end[1]))
                line.axis = str(item.get('axis', 'X'))

            line_count = len(cmp_data.lines)
            active_index = int(snapshot.get('active_index', -1))
            cmp_data.active_index = active_index if 0 <= active_index < line_count else -1

            cam_name = snapshot.get('lines_camera_name')
            cmp_data.lines_camera = bpy.data.objects.get(cam_name) if cam_name else None

            cmp_data.horizon_enabled = bool(snapshot.get('horizon_enabled', cmp_data.horizon_enabled))
            cmp_data.horizon_offset_px = float(snapshot.get('horizon_offset_px', cmp_data.horizon_offset_px))

            saved_world_rotation = float(snapshot.get('world_rotation', cmp_data.world_rotation))
            saved_last_world_rotation = float(snapshot.get('last_world_rotation', saved_world_rotation))
            saved_flip_z = bool(snapshot.get('flip_z_axis', cmp_data.flip_z_axis))
            saved_last_flip_z = bool(snapshot.get('last_flip_z', saved_flip_z))

            # 先让更新回调看到零增量，避免恢复滑块时再次旋转相机。
            cmp_data.last_world_rotation = saved_world_rotation
            cmp_data.world_rotation = saved_world_rotation
            cmp_data.last_flip_z = saved_flip_z
            cmp_data.flip_z_axis = saved_flip_z
            cmp_data.last_world_rotation = saved_last_world_rotation
            cmp_data.last_flip_z = saved_last_flip_z

            camera_state = snapshot.get('camera')
            camera = context.scene.camera
            if camera_state and camera is not None and camera.name == camera_state.get('name'):
                matrix_world = camera_state.get('matrix_world')
                if matrix_world is not None:
                    camera.matrix_world = mathutils.Matrix(matrix_world)
                camera.data.lens = float(camera_state.get('lens', camera.data.lens))
                camera.data.shift_x = float(camera_state.get('shift_x', camera.data.shift_x))
                camera.data.shift_y = float(camera_state.get('shift_y', camera.data.shift_y))
                context.view_layer.update()
        finally:
            properties.resume_horizon_updates()
            utils.restore_camera_view_state(view_state)

    def push_history(self, context):
        snapshot = self.state_to_snapshot(context)
        self.undo_stack.append(snapshot)
        if len(self.undo_stack) > 128:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def undo(self, context):
        if not self.undo_stack:
            return
        current = self.state_to_snapshot(context)
        self.redo_stack.append(current)
        snapshot = self.undo_stack.pop()
        self.restore_snapshot(context, snapshot)
        self.last_error = ""
        self.update_header(context)

    def redo(self, context):
        if not self.redo_stack:
            return
        current = self.state_to_snapshot(context)
        self.undo_stack.append(current)
        snapshot = self.redo_stack.pop()
        self.restore_snapshot(context, snapshot)
        self.last_error = ""
        self.update_header(context)

    def begin_drag_history(self, context, _idx=-1):
        if self.drag_history is None:
            self.drag_history = self.state_to_snapshot(context)

    def finish_drag_history(self, context):
        if self.drag_history is None:
            return
        current = self.state_to_snapshot(context)
        if current != self.drag_history:
            self.undo_stack.append(self.drag_history)
            if len(self.undo_stack) > 128:
                self.undo_stack.pop(0)
            self.redo_stack.clear()
        self.drag_history = None

    def reset_horizon_manual_offset(self, context):
        cmp_data = context.scene.cmp_data
        if cmp_data.horizon_offset_px != 0.0:
            cmp_data.horizon_offset_px = 0.0

    def clear_all_lines(self, context, push_history=True, run_solve=True):
        cmp_data = context.scene.cmp_data
        if push_history and (len(cmp_data.lines) > 0 or cmp_data.active_index != -1):
            self.push_history(context)

        self.reset_horizon_manual_offset(context)
        
        # 彻底清理解算相关的旋转与翻转状态，避免残留影响新绘制的线段或新相机
        cmp_data.last_world_rotation = 0.0
        cmp_data.world_rotation = 0.0
        cmp_data.last_flip_z = False
        cmp_data.flip_z_axis = False

        cmp_data.lines.clear()
        cmp_data.active_index = -1
        cmp_data.lines_camera = context.scene.camera

        self.state = self.STATE_IDLE
        self.draw_axis_constraint = None
        self.active_handle = -1
        self.drag_history = None

        self.draw_anchor_norm = None
        self.draw_anchor_value_norm = None
        self.draw_shift_state = None
        self.line_drag_start_mouse_norm = None
        self.line_drag_start_value_norm = None
        self.line_drag_shift_state = None

        self.end_horizon_drag_updates()

        if run_solve:
            self.last_error = ""
            self.update_header(context)

    def start_drawing(self, context, x, y):
        norm = self.screen_to_norm(context, x, y)
        if not norm:
            return
        self.push_history(context)
        self.reset_horizon_manual_offset(context)
        line = context.scene.cmp_data.lines.add()
        line.start = norm
        line.end = norm
        line.axis = self.current_axis
        context.scene.cmp_data.active_index = len(context.scene.cmp_data.lines) - 1
        context.scene.cmp_data.is_creating_line = True
        self.draw_anchor_norm = Vector(norm)
        self.draw_anchor_value_norm = Vector(norm)
        self.state = self.STATE_DRAWING

    def should_run_realtime_solve(self, context):
        cmp_data = context.scene.cmp_data
        return len(cmp_data.lines) >= 2

    def trigger_solve(self, context, force=False):
        try:
            cmp_data = context.scene.cmp_data
            if cmp_data.lines_camera is not None and cmp_data.lines_camera != context.scene.camera:
                # 保底处理：如果触发了解算时还是另一个相机，这里直接自动重置
                self.clear_all_lines(context, push_history=False, run_solve=False)
                return

            if len(cmp_data.lines) < 2:
                self.last_error = ""
                self.update_header(context)
                return

            if not force and self.state in {
                self.STATE_DRAWING,
                self.STATE_DRAGGING,
                self.STATE_DRAG_HORIZON_OFFSET,
            }:
                if not self.should_run_realtime_solve(context):
                    return

            now = time.time()
            if not force and (now - self.last_solve_ts) < self.SOLVE_INTERVAL:
                return

            from . import operators
            success, msg = operators.solve_camera_core(context)
            if not success:
                self.last_error = msg
            else:
                self.last_error = ""
            self.last_solve_ts = now
            self.update_header(context)
        except Exception as e:
            print(e)
            pass

    def quit(self, context):
        if getattr(self, "_cleanup_done", False):
            return
        self._cleanup_done = True

        try:
            self.end_horizon_drag_updates()
        except Exception:
            properties.reset_horizon_update_state()

        scene = getattr(context, "scene", None)
        cmp_data = getattr(scene, "cmp_data", None) if scene is not None else None
        if cmp_data is not None:
            cmp_data.is_drawing_mode = False
            cmp_data.is_creating_line = False
            cmp_data.active_index = -1

        area = getattr(context, "area", None)
        if area is not None:
            try:
                area.header_text_set(None)
                area.tag_redraw()
            except Exception:
                pass

        try:
            from . import gpu_draw
            gpu_draw.unregister()
        except Exception:
            pass

    def cancel(self, context):
        self.quit(context)

    def screen_to_norm(self, context, x, y):
        if not utils.is_camera_view(context):
            return None
        region = context.region
        rv3d = context.region_data
        cam = context.scene.camera
        if not cam:
            return None
        try:
            vec = view3d_utils.region_2d_to_vector_3d(region, rv3d, (x, y))
            loc = view3d_utils.region_2d_to_origin_3d(region, rv3d, (x, y)) + vec * 10
            co = bpy_extras.object_utils.world_to_camera_view(context.scene, cam, loc)
            if not (np.isfinite(co.x) and np.isfinite(co.y)):
                return None
            u = float(np.clip(co.x, -0.5, 1.5))
            v = float(np.clip(co.y, -0.5, 1.5))
            return Vector((u, v))
        except Exception:
            return None

    def check_dot_click(self, context, x, y):
        lines = context.scene.cmp_data.lines
        cam = context.scene.camera
        if not cam:
            return -1
        TR, TL, BL, BR = utils.get_ordered_frame_points(context)
        if not TR:
            return -1

        region = context.region
        rv3d = context.region_data
        mw = cam.matrix_world
        mouse_pos = Vector((x, y))
        best_dist = 20.0
        best_idx = -1

        for i, line in enumerate(lines):
            top = TL.lerp(TR, line.start[0])
            bot = BL.lerp(BR, line.start[0])
            p1 = mw @ bot.lerp(top, line.start[1])
            top = TL.lerp(TR, line.end[0])
            bot = BL.lerp(BR, line.end[0])
            p2 = mw @ bot.lerp(top, line.end[1])

            s = view3d_utils.location_3d_to_region_2d(region, rv3d, p1)
            e = view3d_utils.location_3d_to_region_2d(region, rv3d, p2)

            if s and e:
                mid = (s + e) * 0.5
                dist = (mouse_pos - mid).length
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
        return best_idx

    def check_endpoint_click(self, context, x, y, idx):
        if idx == -1:
            return -1
        lines = context.scene.cmp_data.lines
        if idx >= len(lines):
            return -1
        line = lines[idx]
        cam = context.scene.camera
        TR, TL, BL, BR = utils.get_ordered_frame_points(context)
        if not TR:
            return -1

        region = context.region
        rv3d = context.region_data
        mw = cam.matrix_world

        top = TL.lerp(TR, line.start[0])
        bot = BL.lerp(BR, line.start[0])
        p1 = mw @ bot.lerp(top, line.start[1])
        top = TL.lerp(TR, line.end[0])
        bot = BL.lerp(BR, line.end[0])
        p2 = mw @ bot.lerp(top, line.end[1])

        s = view3d_utils.location_3d_to_region_2d(region, rv3d, p1)
        e = view3d_utils.location_3d_to_region_2d(region, rv3d, p2)
        m = Vector((x, y))

        if s and (m - s).length < 20:
            return 0
        if e and (m - e).length < 20:
            return 1
        return -1


def register():
    utils.register_class_safe(CMP_OT_DrawLine)


def unregister():
    utils.unregister_class_safe(CMP_OT_DrawLine)
