import bpy
import sys
import numpy as np
import bpy_extras
from mathutils import Vector
from bpy_extras import view3d_utils

from . import utils


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

    state = STATE_IDLE
    current_axis = 'X'
    active_handle = -1
    drag_start_x = 0
    drag_start_y = 0
    DRAG_THRESHOLD = 10

    def invoke(self, context, event):
        if not context.scene.camera:
            self.report({'WARNING'}, "Please add a camera first")
            return {'CANCELLED'}

        if not utils.is_camera_view(context):
            self.report({'WARNING'}, "Please switch to Camera View first")
            return {'CANCELLED'}

        self.state = self.STATE_IDLE
        self.current_axis = 'X'
        self.last_error = ""
        self.undo_stack = []
        self.redo_stack = []
        self.drag_history = None
        context.scene.cmp_data.active_index = -1
        context.scene.cmp_data.is_drawing_mode = True
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
            base = iface_("CameraMatch [3D]: Axis %s (1/2/3). Drag to draw | Click dot to edit | %s+Z undo | %s+Shift+Z redo | Alt+X clear all | Esc / Right click exit") % (c, mod, mod)

            if self.last_error:
                msg = base + iface_(" | Error: ") + iface_(self.last_error)
            else:
                msg = base

            context.area.header_text_set(msg)
        except Exception:
            pass

    def modal(self, context, event):
        context.area.tag_redraw()

        if event.type in {'RIGHTMOUSE', 'ESC'}:
            self.quit(context)
            return {'FINISHED'}

        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
            return {'PASS_THROUGH'}

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
            elif event.type == 'X':
                if self.state == self.STATE_EDITING:
                    lines = context.scene.cmp_data.lines
                    idx = context.scene.cmp_data.active_index
                    if 0 <= idx < len(lines):
                        self.push_history(context)
                        lines.remove(idx)
                        context.scene.cmp_data.active_index = -1
                        self.state = self.STATE_IDLE
                        self.trigger_solve(context)

        x, y = event.mouse_region_x, event.mouse_region_y
        lines = context.scene.cmp_data.lines
        idx = context.scene.cmp_data.active_index

        if event.type == 'MOUSEMOVE':
            if self.state == self.STATE_DRAWING:
                norm = self.screen_to_norm(context, x, y)
                if norm and lines:
                    lines[-1].end = norm
                    self.trigger_solve(context)
            elif self.state == self.STATE_DRAGGING:
                norm = self.screen_to_norm(context, x, y)
                if norm and idx != -1:
                    line = lines[idx]
                    if self.active_handle == 0:
                        line.start = norm
                    else:
                        line.end = norm
                    self.trigger_solve(context)
            elif self.state == self.STATE_WAITING_DRAG:
                dist = np.hypot(x - self.drag_start_x, y - self.drag_start_y)
                if dist > self.DRAG_THRESHOLD:
                    self.start_drawing(context, self.drag_start_x, self.drag_start_y)
                    norm = self.screen_to_norm(context, x, y)
                    if norm and lines:
                        lines[-1].end = norm
                        self.trigger_solve(context)

        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            if self.state == self.STATE_IDLE:
                dot = self.check_dot_click(context, x, y)
                if dot != -1:
                    context.scene.cmp_data.active_index = dot
                    self.state = self.STATE_EDITING
                else:
                    self.drag_start_x = x
                    self.drag_start_y = y
                    self.state = self.STATE_WAITING_DRAG
            elif self.state == self.STATE_EDITING:
                h = self.check_endpoint_click(context, x, y, idx)
                if h != -1:
                    self.active_handle = h
                    self.begin_drag_history(context, idx)
                    self.state = self.STATE_DRAGGING
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
            elif self.state == self.STATE_WAITING_DRAG:
                self.state = self.STATE_IDLE
                context.scene.cmp_data.active_index = -1
            elif self.state == self.STATE_DRAGGING:
                self.finish_drag_history(context)
                self.state = self.STATE_EDITING

        return {'RUNNING_MODAL'}

    def line_to_dict(self, line):
        return {
            'start': tuple(line.start),
            'end': tuple(line.end),
            'axis': line.axis,
        }

    def snapshot_lines(self, context):
        return [self.line_to_dict(line) for line in context.scene.cmp_data.lines]

    def restore_snapshot(self, context, snapshot):
        lines = context.scene.cmp_data.lines
        while len(lines) > 0:
            lines.remove(len(lines) - 1)
        for line_data in snapshot:
            line = lines.add()
            line.start = line_data['start']
            line.end = line_data['end']
            line.axis = line_data['axis']
        context.scene.cmp_data.active_index = -1
        self.state = self.STATE_IDLE
        context.scene.cmp_data.is_creating_line = False
        self.trigger_solve(context)

    def push_history(self, context):
        self.undo_stack.append(self.snapshot_lines(context))
        self.redo_stack.clear()

    def undo(self, context):
        current = self.snapshot_lines(context)
        if not self.undo_stack:
            return
        snapshot = self.undo_stack.pop()
        self.redo_stack.append(current)
        self.restore_snapshot(context, snapshot)

    def redo(self, context):
        current = self.snapshot_lines(context)
        if not self.redo_stack:
            return
        snapshot = self.redo_stack.pop()
        self.undo_stack.append(current)
        self.restore_snapshot(context, snapshot)

    def begin_drag_history(self, context, idx):
        if 0 <= idx < len(context.scene.cmp_data.lines):
            self.drag_history = self.snapshot_lines(context)
        else:
            self.drag_history = None

    def finish_drag_history(self, context):
        if self.drag_history is None:
            return
        current = self.snapshot_lines(context)
        if current != self.drag_history:
            self.undo_stack.append(self.drag_history)
            self.redo_stack.clear()
        self.drag_history = None

    def clear_all_lines(self, context):
        lines = context.scene.cmp_data.lines
        if len(lines) == 0:
            return
        self.push_history(context)
        while len(lines) > 0:
            lines.remove(len(lines) - 1)
        context.scene.cmp_data.active_index = -1
        self.state = self.STATE_IDLE
        self.last_error = ""
        self.update_header(context)
        context.area.tag_redraw()
        self.trigger_solve(context)

    def start_drawing(self, context, x, y):
        norm = self.screen_to_norm(context, x, y)
        if not norm:
            return
        self.push_history(context)
        l = context.scene.cmp_data.lines.add()
        l.start = norm
        l.end = norm
        l.axis = self.current_axis
        context.scene.cmp_data.active_index = len(context.scene.cmp_data.lines) - 1
        context.scene.cmp_data.is_creating_line = True
        self.state = self.STATE_DRAWING

    def trigger_solve(self, context):
        try:
            from . import operators
            success, msg = operators.solve_camera_core(context)
            if not success:
                self.last_error = msg
            else:
                self.last_error = ""
            self.update_header(context)
        except Exception as e:
            print(e)
            pass

    def quit(self, context):
        context.scene.cmp_data.is_drawing_mode = False
        context.scene.cmp_data.is_creating_line = False

        context.area.header_text_set(None)
        context.window.cursor_modal_restore()

        try:
            from . import gpu_draw
            gpu_draw.unregister()
        except Exception:
            pass

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
            return Vector((co.x, co.y))
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
