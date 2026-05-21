import math
import os

import bpy

from .constants import THEME as T
from .draw import (
    art_text,
    begin_scissor,
    dashed_line,
    end_scissor,
    image_rect,
    image_size,
    line,
    receipt_body,
    rect,
    rounded_rect,
    rounded_rect_gradient_y,
    set_viewport,
    soft_circle,
    swatch,
    text,
)
from .node_utils import (
    apply_color_to_node,
    color_at_mouse,
    color_tuple,
    is_material_shader_editor,
    node_under_mouse,
    window_region,
)
from .properties import add_color, add_group, ensure_palette, remove_color, remove_group

iface_ = bpy.app.translations.pgettext_iface


def version_text(info):
    return ".".join(str(part) for part in info.get("version", ()))


class RA_OT_ColorPaletteHUD(bpy.types.Operator):
    bl_idname = "wittyming_color_palette.hud"
    bl_label = "GPU Color Palette"
    bl_options = {"REGISTER"}

    _running = None
    _bl_info = None
    _PRINTER_H = 40
    _PRINTER_PAD_X = 20
    _PRINTER_OVERLAP = 20
    _PAPER_RETRACTED_H = 32
    _NEAR_MARGIN = 26
    _SNAP_DISTANCE = 20
    _TOP_CLEARANCE = 8
    _DRAG_SNAP_DISTANCE = 6
    _PAPER_NOTCH_RADIUS = 10.5
    _PINHOLE_RADIUS = 4.15

    @classmethod
    def is_running(cls):
        return bool(cls._running and getattr(cls._running, "_draw_handle", None))

    def _reset_state(self):
        self._draw_handle = None
        self._timer = None
        self._drag_panel = False
        self._drag_color = None
        self._drag_color_index = None
        self._hover_key = None
        self._swatch_rects = []
        self._add_rects = []
        self._remove_rects = []
        self._group_rects = []
        self._notch_points = []
        self._printer_rect = (0, 0, 0, 0)
        self._panel_rect = (24, 120, 304, 156)
        self._last_mouse = (0, 0)
        self._record_source = None
        self._record_candidate_color = None
        self._record_active = False
        self._record_group_hover = None
        self._area = None
        self._region = None
        self._open_progress = 0.0
        self._target_open = 0.0
        self._save_position_on_stop = False

    def invoke(self, context, event):
        self._reset_state()
        if not is_material_shader_editor(context):
            self.report({"WARNING"}, iface_("Please use this in the material node editor"))
            return {"CANCELLED"}
        if RA_OT_ColorPaletteHUD.is_running():
            RA_OT_ColorPaletteHUD._running.stop(context)
            RA_OT_ColorPaletteHUD._running = None
            return {"FINISHED"}

        self._area = context.area
        self._region = window_region(context.area)
        if not self._region:
            self.report({"WARNING"}, iface_("Node window region not found"))
            return {"CANCELLED"}

        ensure_palette(context.scene)
        self._ensure_group(context.scene)
        paper_top = self._region.height - self._PRINTER_H + self._PRINTER_OVERLAP - self._TOP_CLEARANCE
        self._panel_rect = (36, max(8, paper_top - 190), 312, 190)
        self._restore_position(context.scene)
        self._layout(context.scene)
        self._save_position_on_stop = True
        self._draw_handle = bpy.types.SpaceNodeEditor.draw_handler_add(
            self._draw, (), "WINDOW", "POST_PIXEL"
        )
        RA_OT_ColorPaletteHUD._running = self
        context.window_manager.modal_handler_add(self)
        context.area.tag_redraw()
        return {"RUNNING_MODAL"}

    def stop(self, context):
        if self._save_position_on_stop and context and getattr(context, "scene", None):
            self._save_position(context.scene)
        if self._draw_handle:
            bpy.types.SpaceNodeEditor.draw_handler_remove(self._draw_handle, "WINDOW")
            self._draw_handle = None
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        if context.area:
            context.area.tag_redraw()

    def _event_xy(self, event):
        region = self._region
        if region and hasattr(region, "x") and hasattr(region, "y"):
            return event.mouse_x - region.x, event.mouse_y - region.y
        return event.mouse_region_x, event.mouse_region_y

    def modal(self, context, event):
        if RA_OT_ColorPaletteHUD._running is not self:
            self.stop(context)
            return {"CANCELLED"}
        if not is_material_shader_editor(context) or context.area != self._area:
            self.stop(context)
            RA_OT_ColorPaletteHUD._running = None
            return {"CANCELLED"}

        if event.type == "TIMER":
            self._step_open_animation(context)
            return {"RUNNING_MODAL"}

        mx, my = self._event_xy(event)
        if event.type in {"LEFTMOUSE", "MIDDLEMOUSE"}:
            self._layout(context.scene)

        if event.type == "MOUSEMOVE":
            was_hover_key = self._hover_key
            was_group_hover = self._record_group_hover
            self._update_printer_rect()
            inside_panel = self._hit_panel(mx, my)
            near_interface = self._near_interface(mx, my)
            self._set_open_target(context, 1.0 if near_interface or self._drag_panel or self._drag_color or self._record_active else 0.0)
            if self._record_source and self._record_candidate_color is None and inside_panel:
                self._record_candidate_color = color_at_mouse(context, self._region, self._record_source)
                if not self._record_candidate_color:
                    self._record_source = None
            self._hover_key = self._hit_key(mx, my) if inside_panel else None
            if self._record_source and self._record_candidate_color and inside_panel and not self._record_active:
                self._record_active = True
                self._record_group_hover = self._group_at(mx, my)
            elif self._record_active:
                self._record_group_hover = self._group_at(mx, my) if inside_panel else None
            else:
                self._record_group_hover = None
            if self._drag_panel:
                dx = mx - self._last_mouse[0]
                dy = my - self._last_mouse[1]
                x, y, w, h = self._panel_rect
                self._panel_rect = (x + dx, y + dy, w, h)
                self._snap_panel(context, self._DRAG_SNAP_DISTANCE)
                self._layout(context.scene)
            self._last_mouse = (mx, my)
            if self._drag_panel or self._drag_color or was_hover_key != self._hover_key or was_group_hover != self._record_group_hover:
                context.area.tag_redraw()
            if self._drag_panel or self._drag_color or self._record_active:
                return {"RUNNING_MODAL"}
            return {"RUNNING_MODAL"} if inside_panel or self._rect_hit(self._printer_rect, mx, my) else {"PASS_THROUGH"}

        if event.type == "MIDDLEMOUSE":
            if event.value == "PRESS" and self._hit_interface(mx, my):
                self._drag_panel = True
                self._last_mouse = (mx, my)
                self._set_open_target(context, 1.0)
                return {"RUNNING_MODAL"}
            if event.value == "RELEASE":
                if self._drag_panel:
                    self._drag_panel = False
                    self._snap_panel(context)
                    self._save_position(context.scene)
                    self._set_open_target(context, 1.0 if self._near_interface(mx, my) else 0.0)
                    context.area.tag_redraw()
                    return {"RUNNING_MODAL"}
                return {"PASS_THROUGH"}

        if event.type == "LEFTMOUSE":
            if event.value == "PRESS":
                if self._record_source:
                    self._record_source = None
                    self._record_candidate_color = None
                    self._record_active = False
                    self._record_group_hover = None
                if self._hit_panel(mx, my):
                    key = self._hit_key(mx, my)
                    if self._handle_panel_press(context, key):
                        return {"RUNNING_MODAL"}
                    return {"PASS_THROUGH"}
                if self._rect_hit(self._printer_rect, mx, my):
                    return {"RUNNING_MODAL"}
                self._record_candidate_color = None
                self._record_source = (mx, my)
                self._record_active = False
                self._record_group_hover = None
                return {"PASS_THROUGH"}

            if event.value == "RELEASE" and self._drag_color:
                applied = False
                removed = False
                if not self._hit_panel(mx, my):
                    node = node_under_mouse(context, self._region, (mx, my))
                    if node:
                        applied = apply_color_to_node(node, self._drag_color)
                    else:
                        remove_color(context.scene, self._drag_color_index)
                        removed = True
                if applied:
                    self.report({"INFO"}, iface_("Color applied"))
                elif removed:
                    self.report({"INFO"}, iface_("Color deleted"))
                self._drag_color = None
                self._drag_color_index = None
                self._set_open_target(context, 1.0 if self._near_interface(mx, my) else 0.0)
                context.area.tag_redraw()
                return {"RUNNING_MODAL"}

            if event.value == "RELEASE" and self._record_source:
                was_active = self._record_active
                if was_active and self._hit_panel(mx, my):
                    color = self._record_candidate_color
                    if color:
                        group_index = self._group_at(mx, my)
                        add_color(context.scene, color, group_index)
                        self.report({"INFO"}, iface_("Color recorded"))
                        self._record_source = None
                        self._record_candidate_color = None
                        self._record_active = False
                        self._record_group_hover = None
                        self._set_open_target(context, 1.0 if self._near_interface(mx, my) else 0.0)
                        context.area.tag_redraw()
                        return {"RUNNING_MODAL"}
                self._record_source = None
                self._record_candidate_color = None
                self._record_active = False
                self._record_group_hover = None
                if was_active:
                    self._set_open_target(context, 1.0 if self._near_interface(mx, my) else 0.0)
                    context.area.tag_redraw()
                    return {"RUNNING_MODAL"}

        return {"PASS_THROUGH"}

    def _handle_panel_press(self, context, key):
        if not key:
            return False
        kind = key[0]
        if kind == "swatch":
            color_index = key[1]
            self._drag_color = color_tuple(context.scene.WittyMing_color_palette_colors[color_index].color)
            self._drag_color_index = color_index
            return True
        if kind == "add":
            add_group(context.scene)
            return True
        if kind == "remove":
            remove_group(context.scene, key[1])
            return True
        return False

    def _layout(self, scene):
        self._ensure_group(scene)
        groups = scene.WittyMing_color_palette_groups
        colors = scene.WittyMing_color_palette_colors
        x, y, w, _h = self._panel_rect
        pad = 26
        swatch_size = 34
        gap = 4
        header_h = 106
        footer_h = 42
        divider_to_grid = 22
        group_gap = 24
        control_size = 21
        cols = max(1, int((w - pad * 2 + gap) / (swatch_size + gap)))

        self._swatch_rects = []
        self._add_rects = []
        self._remove_rects = []
        self._group_rects = []
        self._notch_points = []
        group_items = []
        content_h = 0
        for group_index in range(len(groups)):
            indices = [i for i, item in enumerate(colors) if item.group == group_index]
            rows = max(1, math.ceil(max(len(indices), 1) / cols))
            group_items.append((group_index, indices, rows))
            content_h += divider_to_grid + rows * swatch_size + max(0, rows - 1) * gap + group_gap

        h = header_h + content_h + footer_h
        if self._region:
            x = max(8, min(self._region.width - w - 8, x))
            current_printer_top = y + _h - self._PRINTER_OVERLAP + self._PRINTER_H
            printer_top = min(self._region.height - self._TOP_CLEARANCE, max(8 + self._PRINTER_H, current_printer_top))
            y = printer_top - self._PRINTER_H + self._PRINTER_OVERLAP - h
            min_y = 8 - max(0, h - self._PAPER_RETRACTED_H)
            y = max(min_y, min(self._region.height - self._TOP_CLEARANCE - h, y))
        self._panel_rect = (x, y, w, h)
        self._update_printer_rect()

        cursor = y + h - header_h
        for group_index, indices, rows in group_items:
            divider_y = cursor
            self._add_rects.append((group_index, x + pad, divider_y - control_size * 0.5, control_size, control_size))
            self._remove_rects.append((
                group_index,
                x + w - pad - control_size,
                divider_y - control_size * 0.5,
                control_size,
                control_size,
            ))
            grid_top = divider_y - divider_to_grid
            group_bottom = grid_top - rows * swatch_size - max(0, rows - 1) * gap - 10
            group_height = divider_y - group_bottom + 12
            self._group_rects.append((group_index, x + 18, group_bottom, w - 36, group_height))
            self._notch_points.append(group_bottom + group_height * 0.5)
            for local_index, color_index in enumerate(indices):
                col = local_index % cols
                row = local_index // cols
                self._swatch_rects.append((
                    color_index,
                    x + pad + col * (swatch_size + gap),
                    grid_top - swatch_size - row * (swatch_size + gap),
                    swatch_size,
                    swatch_size,
                ))
            cursor = grid_top - rows * swatch_size - max(0, rows - 1) * gap - group_gap

    def _draw(self):
        context = bpy.context
        if not is_material_shader_editor(context) or context.area != self._area:
            return
        info = self._bl_info or {}
        scene = context.scene
        self._layout(scene)
        x, y, w, h = self._panel_rect
        if self._region:
            set_viewport(self._region.width, self._region.height)
        pad = 26
        colors = scene.WittyMing_color_palette_colors
        groups = scene.WittyMing_color_palette_groups

        border = T["border"]
        border_width = 1.05
        visible_y, visible_h = self._visible_panel_rect()

        if visible_h > 1:
            self._draw_printer_back()
            scissor = begin_scissor(x - 18, visible_y, w + 36, visible_h + 18)
            self._draw_receipt_shadow(x, visible_y, w, visible_h)
            receipt_body(
                x,
                visible_y,
                w,
                visible_h,
                T["paper"],
                border,
                border_width,
                None,
                self._paper_side_notches(y, h, visible_y, visible_h),
                self._paper_bottom_notches(x, w),
            )
            self._draw_paper_tint(x, y, w, h)
            self._draw_header(x, y, w, h, info)

            if len(colors) == 0:
                text(iface_("Drag color here to record"), x + w * 0.5, y + h - 126, 11, T["muted"], "CENTER", True)

            for group_index in range(len(groups)):
                self._draw_group_divider(x, w, pad, group_index)

            self._draw_swatch_backplates(colors)
            for color_index, sx, sy, size, _h in self._swatch_rects:
                item = colors[color_index]
                swatch(sx, sy, size, color_tuple(item.color), self._hover_key == ("swatch", color_index))

            end_scissor()

        self._draw_printer_front()

        if self._drag_color:
            mx, my = self._last_mouse
            rounded_rect(mx - 23, my - 23, 46, 46, 14, (0.0, 0.0, 0.0, 0.14), None)
            swatch(mx - 18, my - 18, 36, self._drag_color, True)

    def _paper_top_y(self):
        _x, y, _w, h = self._panel_rect
        return y + h

    def _printer_top_y(self):
        return self._paper_top_y() - self._PRINTER_OVERLAP + self._PRINTER_H

    def _update_printer_rect(self):
        x, _y, w, _h = self._panel_rect
        paper_top = self._paper_top_y()
        self._printer_rect = (
            x - self._PRINTER_PAD_X,
            paper_top - self._PRINTER_OVERLAP,
            w + self._PRINTER_PAD_X * 2,
            self._PRINTER_H,
        )

    def _rect_hit(self, rect_value, mx, my, margin=0):
        x, y, w, h = rect_value
        return x - margin <= mx <= x + w + margin and y - margin <= my <= y + h + margin

    def _visible_panel_rect(self):
        x, y, w, h = self._panel_rect
        paper_top = y + h
        visible_h = min(h, self._PAPER_RETRACTED_H + max(0.0, h - self._PAPER_RETRACTED_H) * self._open_progress)
        return paper_top - visible_h, visible_h

    def _paper_side_notches(self, y, h, visible_y, visible_h):
        min_y = max(y + 28, visible_y + self._PAPER_NOTCH_RADIUS + 2)
        max_y = min(y + h - 44, visible_y + visible_h - self._PAPER_NOTCH_RADIUS - 2)
        notches = []
        for notch_y in self._notch_points:
            if min_y <= notch_y <= max_y:
                notches.append((notch_y, self._PAPER_NOTCH_RADIUS))
        return notches

    def _paper_bottom_notches(self, x, w):
        count = max(18, int(w / 12))
        spacing = w / max(1, count - 1)
        return [(x + spacing * 0.5 + index * spacing, self._PINHOLE_RADIUS) for index in range(count - 1)]

    def _set_open_target(self, context, target):
        target = max(0.0, min(1.0, target))
        if abs(self._target_open - target) <= 0.001 and abs(self._open_progress - target) <= 0.001:
            return
        self._target_open = target
        if not self._timer:
            self._timer = context.window_manager.event_timer_add(0.025, window=context.window)
        context.area.tag_redraw()

    def _step_open_animation(self, context):
        delta = self._target_open - self._open_progress
        if abs(delta) <= 0.018:
            self._open_progress = self._target_open
            if self._timer:
                context.window_manager.event_timer_remove(self._timer)
                self._timer = None
        else:
            self._open_progress += delta * 0.34
        context.area.tag_redraw()

    def _snap_targets(self, context):
        if not self._region:
            return []
        return [self._region.height - self._TOP_CLEARANCE]

    def _snap_panel(self, context, distance=None):
        if not self._region:
            return
        distance = self._SNAP_DISTANCE if distance is None else distance
        x, y, w, h = self._panel_rect
        printer_top = self._printer_top_y()
        best_target = None
        best_distance = distance + 1
        for target in self._snap_targets(context):
            target_distance = abs(printer_top - target)
            if target_distance < best_distance:
                best_distance = target_distance
                best_target = target
        if best_target is not None and best_distance <= distance:
            paper_top = best_target - self._PRINTER_H + self._PRINTER_OVERLAP
            self._panel_rect = (x, paper_top - h, w, h)
            self._update_printer_rect()

    def _save_position(self, scene):
        if not scene:
            return
        x, _y, _w, _h = self._panel_rect
        try:
            scene.WittyMing_color_palette_hud_x = float(x)
            scene.WittyMing_color_palette_hud_printer_top = float(self._printer_top_y())
        except Exception:
            pass

    def _restore_position(self, scene):
        if not scene or not self._region:
            return
        try:
            saved_x = float(getattr(scene, "WittyMing_color_palette_hud_x", -1.0))
            saved_printer_top = float(getattr(scene, "WittyMing_color_palette_hud_printer_top", -1.0))
        except Exception:
            return
        if saved_x < 0.0 or saved_printer_top < 0.0:
            return
        _x, _y, w, h = self._panel_rect
        x = max(8, min(self._region.width - w - 8, saved_x))
        printer_top = max(8 + self._PRINTER_H, min(self._region.height - self._TOP_CLEARANCE, saved_printer_top))
        y = printer_top - self._PRINTER_H + self._PRINTER_OVERLAP - h
        self._panel_rect = (x, y, w, h)
        self._update_printer_rect()

    def _draw_printer_back(self):
        px, py, pw, ph = self._printer_rect
        paper_top = self._paper_top_y()
        slot_h = 12
        slot_y = paper_top - slot_h * 0.5
        shell_r = ph * 0.5

        rounded_rect(px + 5, py - 4, pw - 10, 8, 4, (0.0, 0.0, 0.0, 0.070), None)
        rounded_rect(px + 1, py - 1, pw - 2, ph + 2, shell_r, (0.10, 0.15, 0.26, 0.055), None)
        rounded_rect_gradient_y(
            px,
            py,
            pw,
            ph,
            shell_r,
            (
                (0.00, (0.62, 0.72, 0.90, 0.99)),
                (0.24, (0.86, 0.93, 1.00, 0.99)),
                (0.56, (0.97, 0.99, 1.00, 0.99)),
                (1.00, (0.82, 0.91, 1.00, 0.99)),
            ),
            24,
            T["printer_shell_edge"],
            0.85,
        )
        rounded_rect(px + 4, py + ph - 10, pw - 8, 6, 4, (1.0, 1.0, 1.0, 0.26), None)
        rounded_rect(px + 10, py + ph - 16, pw - 20, 2, 1, (1.0, 1.0, 1.0, 0.18), None)
        rounded_rect(px + 7, py + 5, pw - 14, 8, 4, (0.34, 0.45, 0.68, 0.13), None)
        rect(px + 19, py + 5, pw - 38, 1, (0.16, 0.24, 0.42, 0.13))

        rounded_rect(px + 13, slot_y - 2, pw - 26, slot_h + 4, 6, (0.10, 0.15, 0.28, 0.24), None)
        rounded_rect_gradient_y(
            px + 17,
            slot_y,
            pw - 34,
            slot_h,
            4.5,
            (
                (0.00, (0.015, 0.024, 0.070, 0.98)),
                (0.45, (0.040, 0.060, 0.135, 0.98)),
                (1.00, (0.090, 0.130, 0.250, 0.98)),
            ),
            12,
            T["printer_slot_edge"],
            0.7,
        )
        rect(px + 30, slot_y + slot_h - 2, pw - 60, 1, (0.0, 0.0, 0.0, 0.32))
        rect(px + 34, slot_y + 2, pw - 68, 1, (0.66, 0.78, 1.0, 0.12))
        rect(px + 34, slot_y + slot_h - 1, pw - 68, 1, (1.0, 1.0, 1.0, 0.08))

    def _draw_printer_front(self):
        px, _py, pw, _ph = self._printer_rect
        paper_top = self._paper_top_y()
        paper_x = px + self._PRINTER_PAD_X
        paper_w = pw - self._PRINTER_PAD_X * 2
        paper_shadow = (0.30, 0.24, 0.18, 0.045)

        rect(paper_x + 4, paper_top - 2, paper_w - 8, 2, paper_shadow)
        rect(paper_x + 9, paper_top - 7, paper_w - 18, 4, (0.30, 0.24, 0.18, 0.014))
        rect(px + 34, paper_top + 3, pw - 68, 1, (0.0, 0.0, 0.0, 0.12))
        rect(px + 40, paper_top + 5, pw - 80, 1, (0.62, 0.75, 1.0, 0.07))

    def _draw_header(self, x, y, w, h, info):
        line(((x + 26, y + h - 82), (x + w - 26, y + h - 82)), T["divider"], 1.0)
        title = iface_(info.get("name", "Color Palette"))
        title_y = y + h - 65
        title_asset_x = x + 34
        title_asset_y = y + h - 80
        title_asset_w = w - 68
        title_asset_h = 44
        if not self._draw_title_asset(title_asset_x, title_asset_y, title_asset_w, title_asset_h):
            art_text(title, x + w * 0.5 + 1.2, title_y - 1.3, 25, (0.78, 0.63, 0.42, 0.30), "CENTER", True)
            art_text(title, x + w * 0.5 - 0.7, title_y + 0.6, 25, (0.35, 0.18, 0.08, 0.18), "CENTER", True)
            art_text(title, x + w * 0.5, title_y, 25, T["ink"], "CENTER", True)
            line(((x + w * 0.5 - 19, title_y - 3.2), (x + w * 0.5 + 19, title_y - 3.2)), (0.72, 0.32, 0.10, 0.40), 1.15)
        text(
            f"{info.get('author', '')}  v{version_text(info)}",
            x + w * 0.5,
            y + 34,
            8,
            T["muted"],
            "CENTER",
            False,
        )

    def _draw_title_asset(self, x, y, w, h):
        if not self._use_chinese_title_asset():
            return False
        path = os.path.join(os.path.dirname(__file__), "assets", "palette_title.png")
        size = image_size(path)
        if not size:
            return False
        image_w, image_h = size
        scale = min(w / image_w, h / image_h)
        draw_w = image_w * scale
        draw_h = image_h * scale
        return image_rect(path, x + (w - draw_w) * 0.5, y + (h - draw_h) * 0.5, draw_w, draw_h)

    def _use_chinese_title_asset(self):
        language = getattr(bpy.context.preferences.view, "language", "")
        return language in {"zh_CN", "zh_HANS", "zh_TW", "zh_HANT"}

    def _draw_receipt_shadow(self, x, y, w, h):
        side_h = max(8, h - 38)
        rounded_rect(x + 5, y + 18, 3, side_h, 2, (0.0, 0.0, 0.0, 0.025), None)
        rounded_rect(x + w - 2, y + 20, 4, max(8, h - 46), 2, (0.0, 0.0, 0.0, 0.045), None)

    def _draw_paper_tint(self, x, y, w, h):
        rounded_rect(x + 18, y + 34, w - 36, max(24, h - 126), 8, T["paper_deep"], None)
        rect(x + 28, y + 31, w - 56, 1, (1.0, 0.98, 0.90, 0.36))
        rect(x + 28, y + h - 82, w - 56, 1, (0.55, 0.45, 0.34, 0.08))

    def _draw_group_divider(self, x, w, pad, group_index):
        add_rect = next((rect for gi, *rect in self._add_rects if gi == group_index), None)
        remove_rect = next((rect for gi, *rect in self._remove_rects if gi == group_index), None)
        if not add_rect or not remove_rect:
            return
        ax, ay, aw, ah = add_rect
        rx, ry, rw, rh = remove_rect
        y = ay + ah * 0.5
        dashed_line(x + pad + 31, y, x + w - pad - 31, T["divider"], dash=3.0, gap=5.0)
        self._draw_control(ax, ay, aw, ah, "+", self._hover_key == ("add", group_index))
        self._draw_control(rx, ry, rw, rh, "-", self._hover_key == ("remove", group_index))

    def _draw_control(self, x, y, w, h, label, hover):
        color = T["control_hover"] if hover else T["control"]
        cx = x + w * 0.5
        cy = y + h * 0.5
        soft_circle(cx + 0.8, cy - 1.0, w * 0.5, (0.0, 0.0, 0.0, 0.10))
        soft_circle(cx, cy, w * 0.5, color, T["border"], 0.95)
        self._draw_control_icon(cx, cy, label)

    def _draw_control_icon(self, cx, cy, label):
        arm = 5.2
        thickness = 2.15
        radius = thickness * 0.5
        color = T["ink"]
        rounded_rect(cx - arm, cy - thickness * 0.5, arm * 2.0, thickness, radius, color, None)
        if label == "+":
            rounded_rect(cx - thickness * 0.5, cy - arm, thickness, arm * 2.0, radius, color, None)

    def _draw_swatch_backplates(self, colors):
        for group_index, x, y, w, h in self._group_rects:
            active = self._record_active and self._record_group_hover == group_index
            fill = T["alert_soft"] if active else (0.88, 0.80, 0.67, 0.13)
            edge = (0.70, 0.32, 0.10, 0.26) if active else (1.0, 0.96, 0.86, 0.11)
            rounded_rect(x, y, w, h, 9, fill, None)
            rounded_rect(
                x + 1,
                y + 1,
                w - 2,
                h - 2,
                8,
                (0.0, 0.0, 0.0, 0.0),
                edge,
                0.85 if active else 0.65,
            )

    def _hit_panel(self, mx, my):
        x, y, w, h = self._panel_rect
        visible_y, visible_h = self._visible_panel_rect()
        return x <= mx <= x + w and visible_y <= my <= visible_y + visible_h

    def _hit_interface(self, mx, my):
        return self._hit_panel(mx, my) or self._rect_hit(self._printer_rect, mx, my)

    def _near_interface(self, mx, my):
        if self._rect_hit(self._printer_rect, mx, my, self._NEAR_MARGIN):
            return True
        x, _y, w, _h = self._panel_rect
        visible_y, visible_h = self._visible_panel_rect()
        return self._rect_hit((x, visible_y, w, visible_h), mx, my, self._NEAR_MARGIN)

    def _hit_key(self, mx, my):
        if not self._hit_panel(mx, my):
            return None
        for group_index, x, y, w, h in self._add_rects:
            if x <= mx <= x + w and y <= my <= y + h:
                return ("add", group_index)
        for group_index, x, y, w, h in self._remove_rects:
            if x <= mx <= x + w and y <= my <= y + h:
                return ("remove", group_index)
        for color_index, x, y, w, h in self._swatch_rects:
            if x <= mx <= x + w and y <= my <= y + h:
                return ("swatch", color_index)
        return None

    def _group_at(self, mx, my):
        if not self._hit_panel(mx, my):
            return None
        for group_index, x, y, w, h in self._group_rects:
            if x <= mx <= x + w and y <= my <= y + h:
                return group_index
        return None

    def _ensure_group(self, scene):
        if len(scene.WittyMing_color_palette_groups) == 0:
            add_group(scene, sync=False)
        max_group = len(scene.WittyMing_color_palette_groups) - 1
        for item in scene.WittyMing_color_palette_colors:
            if item.group < 0 or item.group > max_group:
                item.group = 0
