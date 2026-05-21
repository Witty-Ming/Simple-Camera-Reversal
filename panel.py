import bpy

from .hud import RA_OT_ColorPaletteHUD
from .node_utils import is_material_shader_editor

iface_ = bpy.app.translations.pgettext_iface


class RA_PT_ColorPalettePanel(bpy.types.Panel):
    bl_idname = "RA_PT_color_palette"
    bl_label = "Color Palette"
    bl_space_type = "NODE_EDITOR"
    bl_region_type = "UI"
    bl_category = "Palette"

    @classmethod
    def poll(cls, context):
        return is_material_shader_editor(context)

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = False
        layout.use_property_decorate = False
        icon = "HIDE_OFF" if RA_OT_ColorPaletteHUD.is_running() else "HIDE_ON"
        label = "Close GPU Color Palette" if RA_OT_ColorPaletteHUD.is_running() else "Open GPU Color Palette"
        layout.operator(RA_OT_ColorPaletteHUD.bl_idname, text=iface_(label), icon=icon)
