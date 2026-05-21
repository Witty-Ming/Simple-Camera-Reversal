import json

import bpy

from .constants import ZERO_COLOR

iface_ = bpy.app.translations.pgettext_iface


_CAPTURE_LOCK = False
_SYNC_LOCK = False
_PALETTE_DATA_VERSION = 1


def _addon_preferences():
    package = __package__ or __name__.partition(".")[0]
    addon = bpy.context.preferences.addons.get(package)
    if addon:
        return addon.preferences
    for key, addon in bpy.context.preferences.addons.items():
        if key.endswith("wittyming_color_palette"):
            return addon.preferences
    return None


def _scene_has_palette_props(scene):
    return bool(
        scene
        and hasattr(scene, "WittyMing_color_palette_groups")
        and hasattr(scene, "WittyMing_color_palette_colors")
    )


def _clear_scene_palette(scene):
    while len(scene.WittyMing_color_palette_colors):
        scene.WittyMing_color_palette_colors.remove(0)
    while len(scene.WittyMing_color_palette_groups):
        scene.WittyMing_color_palette_groups.remove(0)


def _add_default_group(scene):
    group = scene.WittyMing_color_palette_groups.add()
    group.name = "Group 00"
    scene.WittyMing_color_palette_active_group = 0
    return group


def _palette_payload(scene):
    groups = [
        {"name": group.name}
        for group in scene.WittyMing_color_palette_groups
    ]
    colors = [
        {"name": item.name, "group": int(item.group), "color": [float(value) for value in item.color]}
        for item in scene.WittyMing_color_palette_colors
    ]
    return {
        "version": _PALETTE_DATA_VERSION,
        "active_group": int(scene.WittyMing_color_palette_active_group),
        "groups": groups,
        "colors": colors,
    }


def sync_palette(scene):
    if _SYNC_LOCK or not _scene_has_palette_props(scene):
        return
    prefs = _addon_preferences()
    if not prefs:
        return
    try:
        prefs.palette_data = json.dumps(_palette_payload(scene), separators=(",", ":"))
    except Exception:
        pass


def restore_palette(scene):
    global _SYNC_LOCK
    if not _scene_has_palette_props(scene):
        return False
    prefs = _addon_preferences()
    data = getattr(prefs, "palette_data", "") if prefs else ""
    if not data:
        return False
    try:
        payload = json.loads(data)
        groups = payload.get("groups") or []
        colors = payload.get("colors") or []
    except Exception:
        return False

    _SYNC_LOCK = True
    try:
        _clear_scene_palette(scene)
        if not groups:
            _add_default_group(scene)
        else:
            for index, group_data in enumerate(groups):
                group = scene.WittyMing_color_palette_groups.add()
                group.name = str(group_data.get("name") or f"Group {index:02d}")
        max_group = max(0, len(scene.WittyMing_color_palette_groups) - 1)
        for index, color_data in enumerate(colors):
            color = color_data.get("color") or (1.0, 1.0, 1.0, 1.0)
            item = scene.WittyMing_color_palette_colors.add()
            item.name = str(color_data.get("name") or f"{index:02d}")
            item.group = max(0, min(int(color_data.get("group", 0)), max_group))
            values = [float(value) for value in color[:4]]
            while len(values) < 4:
                values.append(1.0)
            item.color = tuple(values[:4])
        active_group = int(payload.get("active_group", 0))
        scene.WittyMing_color_palette_active_group = max(0, min(active_group, max_group))
    finally:
        _SYNC_LOCK = False
    return True


def ensure_palette(scene):
    if not _scene_has_palette_props(scene):
        return
    if restore_palette(scene):
        return
    if len(scene.WittyMing_color_palette_groups) == 0:
        _add_default_group(scene)
    else:
        sync_palette(scene)


def add_color(scene, color, group_index=None):
    if len(scene.WittyMing_color_palette_groups) == 0:
        add_group(scene, sync=False)
    if group_index is None:
        group_index = scene.WittyMing_color_palette_active_group
    group_index = max(0, min(group_index, len(scene.WittyMing_color_palette_groups) - 1))
    item = scene.WittyMing_color_palette_colors.add()
    item.name = f"{len(scene.WittyMing_color_palette_colors):02d}"
    item.color = color
    item.group = group_index
    scene.WittyMing_color_palette_active_group = group_index
    sync_palette(scene)


def add_group(scene, sync=True):
    group = scene.WittyMing_color_palette_groups.add()
    group.name = f"Group {len(scene.WittyMing_color_palette_groups):02d}"
    for item in scene.WittyMing_color_palette_colors:
        item.group += 1
    scene.WittyMing_color_palette_active_group = 0
    if sync:
        sync_palette(scene)
    return group


def remove_group(scene, index):
    groups = scene.WittyMing_color_palette_groups
    if not (0 <= index < len(groups)) or len(groups) <= 1:
        return
    remove_indices = [i for i, item in enumerate(scene.WittyMing_color_palette_colors) if item.group == index]
    for item_index in reversed(remove_indices):
        scene.WittyMing_color_palette_colors.remove(item_index)
    for item in scene.WittyMing_color_palette_colors:
        if item.group > index:
            item.group -= 1
    groups.remove(index)
    scene.WittyMing_color_palette_active_group = max(0, min(scene.WittyMing_color_palette_active_group, len(groups) - 1))
    sync_palette(scene)


def remove_color(scene, index):
    if 0 <= index < len(scene.WittyMing_color_palette_colors):
        scene.WittyMing_color_palette_colors.remove(index)
        sync_palette(scene)


def capture_color_update(scene, context):
    global _CAPTURE_LOCK
    if _CAPTURE_LOCK:
        return
    color = tuple(scene.WittyMing_color_palette_capture)
    if color == ZERO_COLOR:
        return
    add_color(scene, color)
    _CAPTURE_LOCK = True
    try:
        scene.WittyMing_color_palette_capture = ZERO_COLOR
    finally:
        _CAPTURE_LOCK = False


class RA_ColorPaletteSlot(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(name="Name", default="Color")
    group: bpy.props.IntProperty(name="Group", default=0, min=0)
    color: bpy.props.FloatVectorProperty(
        name="Color",
        subtype="COLOR",
        size=4,
        min=0.0,
        max=1.0,
        default=(1.0, 1.0, 1.0, 1.0),
    )


class RA_ColorPaletteGroup(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(name="Name", default="Group")


class WittyMingColorPalettePreferences(bpy.types.AddonPreferences):
    bl_idname = __package__ or __name__.partition(".")[0]

    palette_data: bpy.props.StringProperty(default="", options={"HIDDEN"})

    def draw(self, context):
        self.layout.label(text=iface_("Color palette records are stored automatically."))
