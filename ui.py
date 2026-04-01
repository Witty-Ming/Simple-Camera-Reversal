import bpy

class CMP_PT_MainPanel(bpy.types.Panel):
    bl_label = "Simple Camera Match"
    bl_idname = "CMP_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'CameraMatch'

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        # 1. 绘制工具
        layout.label(text="Step 1: Drawing", icon='GREASEPENCIL')
        row = layout.row()
        row.scale_y = 1.5
        row.operator("cmp.draw_line", text="Start Drawing (Brush)", icon='GREASEPENCIL')
        
        layout.separator()
        
        # 2. 解算工具
        layout.label(text="Step 2: Solve", icon='CHECKMARK')
        col = layout.column(align=True)
        col.scale_y = 1.3
        col.operator("cmp.match_camera", text="Match Camera (Solve)", icon='CAMERA_DATA')
        
        col.separator()
        col.separator()
        row = col.row()
        row.prop(scene.cmp_data, "world_rotation", text="World Rotation (XY)")
        row.prop(scene.cmp_data, "flip_z_axis", text="Flip Z Axis", icon='TRIA_UP' if not scene.cmp_data.flip_z_axis else 'TRIA_DOWN', toggle=True)
        

        layout.separator()

        # 3. 说明
        box = layout.box()
        box.label(text="Instructions:", icon='INFO')
        col = box.column(align=True)
        col.label(text="1/2/3 Key : Switch X/Y/Z Axis")
        col.label(text="Drag : Draw | Click : Edit")
        col.label(text="X Key : Delete Line")
        col.label(text="Right Click : Exit")
        col.label(text="--------------------------")
        col.label(text="Tip: Draw one or no parallel edges")
        col.label(text="Tip: Draw at least 3 perspective edges")
        
        layout.separator()
        
        # 4. 信息
        if scene.camera:
            col = layout.column(align=True)
            col.prop(scene.camera.data, "lens", text="Focal Length (mm)")
            col.prop(scene.camera.data, "sensor_width", text="Sensor (mm)")
        else:
            layout.alert = True
            layout.label(text="Warning: No Active Camera!", icon='ERROR')

def register():
    try:
        bpy.utils.register_class(CMP_PT_MainPanel)
    except ValueError:
        bpy.utils.unregister_class(CMP_PT_MainPanel)
        bpy.utils.register_class(CMP_PT_MainPanel)

def unregister():
    try:
        bpy.utils.unregister_class(CMP_PT_MainPanel)
    except: pass
