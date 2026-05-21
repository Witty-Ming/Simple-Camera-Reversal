bl_info = {
    "name": "Color Palette",
    "author": "WittyMing",
    "version": (1, 2, 1),
    "blender": (4, 0, 0),
    "location": "Shader Editor > N Panel > Color Palette",
    "description": "Record and drag colors in the material node editor",
    "category": "Node",
}

import bpy
from bpy.app.handlers import persistent

from .constants import ZERO_COLOR
from .hud import RA_OT_ColorPaletteHUD
from .panel import RA_PT_ColorPalettePanel
from .properties import (
    RA_ColorPaletteGroup,
    RA_ColorPaletteSlot,
    WittyMingColorPalettePreferences,
    capture_color_update,
    ensure_palette,
)
from . import translation


RA_OT_ColorPaletteHUD._bl_info = bl_info
RA_PT_ColorPalettePanel.bl_label = bl_info["name"]


_classes = (
    WittyMingColorPalettePreferences,
    RA_ColorPaletteGroup,
    RA_ColorPaletteSlot,
    RA_OT_ColorPaletteHUD,
    RA_PT_ColorPalettePanel,
)


_scene_props = (
    "WittyMing_color_palette_hud_printer_top",
    "WittyMing_color_palette_hud_x",
    "WittyMing_color_palette_capture",
    "WittyMing_color_palette_active_group",
    "WittyMing_color_palette_colors",
    "WittyMing_color_palette_groups",
)


def _safe_unregister_class(cls):
    registered_cls = getattr(bpy.types, cls.__name__, None)
    try:
        bpy.utils.unregister_class(registered_cls or cls)
    except (RuntimeError, ValueError):
        pass


def _safe_register_class(cls):
    try:
        bpy.utils.register_class(cls)
    except (RuntimeError, ValueError) as exc:
        if "already registered" not in str(exc):
            raise
        _safe_unregister_class(cls)
        bpy.utils.register_class(cls)


def _clear_scene_props():
    for prop_name in _scene_props:
        if hasattr(bpy.types.Scene, prop_name):
            delattr(bpy.types.Scene, prop_name)


def _stop_hud_if_running():
    hud = RA_OT_ColorPaletteHUD._running
    if not hud:
        return
    try:
        hud.stop(bpy.context)
    except Exception:
        pass
    RA_OT_ColorPaletteHUD._running = None


@persistent
def _restore_palette_after_load(_dummy):
    ensure_palette(bpy.context.scene)


def _ensure_handlers():
    if _restore_palette_after_load not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_restore_palette_after_load)


def _remove_handlers():
    if _restore_palette_after_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_restore_palette_after_load)


def register():
    translation.register()
    _stop_hud_if_running()
    _clear_scene_props()
    for cls in reversed(_classes):
        _safe_unregister_class(cls)
    RA_OT_ColorPaletteHUD._bl_info = bl_info
    RA_PT_ColorPalettePanel.bl_label = bl_info["name"]
    for cls in _classes:
        _safe_register_class(cls)
    bpy.types.Scene.WittyMing_color_palette_groups = bpy.props.CollectionProperty(type=RA_ColorPaletteGroup)
    bpy.types.Scene.WittyMing_color_palette_colors = bpy.props.CollectionProperty(type=RA_ColorPaletteSlot)
    bpy.types.Scene.WittyMing_color_palette_active_group = bpy.props.IntProperty(default=0, min=0)
    bpy.types.Scene.WittyMing_color_palette_hud_x = bpy.props.FloatProperty(default=-1.0)
    bpy.types.Scene.WittyMing_color_palette_hud_printer_top = bpy.props.FloatProperty(default=-1.0)
    bpy.types.Scene.WittyMing_color_palette_capture = bpy.props.FloatVectorProperty(
        name="Record Color",
        subtype="COLOR",
        size=4,
        min=0.0,
        max=1.0,
        default=ZERO_COLOR,
        update=capture_color_update,
    )
    ensure_palette(bpy.context.scene)
    _ensure_handlers()


def unregister():
    _stop_hud_if_running()
    _remove_handlers()

    _clear_scene_props()

    for cls in reversed(_classes):
        _safe_unregister_class(cls)

    translation.unregister()


if __name__ == "__main__":
    register()
