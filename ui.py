import bpy

class CMP_PT_MainPanel(bpy.types.Panel):
    bl_label = "简易相机反求"
    bl_idname = "CMP_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = '简易相机反求'

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        # 1. 绘制工具
        layout.label(text="第一步: 绘制", icon='GREASEPENCIL')
        row = layout.row()
        row.scale_y = 1.5
        row.operator("cmp.draw_line", text="开始绘制 (画笔)", icon='GREASEPENCIL')
        
        layout.separator()
        
        # 2. 解算工具
        layout.label(text="第二步: 开始拖拽绘制", icon='CHECKMARK')
        col = layout.column(align=True)
        col.scale_y = 1.3
        col.operator("cmp.match_camera", text="匹配相机 (解算)", icon='CAMERA_DATA')
        
        col.separator()
        col.separator()
        row = col.row()
        row.prop(scene.cmp_data, "world_rotation", text="水平旋转(XY平面)")
        row.prop(scene.cmp_data, "flip_z_axis", text="翻转Z轴", icon='TRIA_UP' if not scene.cmp_data.flip_z_axis else 'TRIA_DOWN', toggle=True)
        # 使用 toggle 按钮或者 checkbox。这里用 icon toggle 看起来紧凑配合 slide?
        # 或者仅仅放在下面。
        # 考虑到 text="" 可能太窄，让它显示文字 "翻转Z" 更直观
        # row.prop(scene.cmp_data, "flip_z_axis", text="翻转Z", toggle=True)
        
        row = col.row()
        row.operator("cmp.clear_lines", text="清除所有线条", icon='TRASH')
        
        layout.separator()

        # 3. 说明
        box = layout.box()
        box.label(text="操作说明:", icon='INFO')
        col = box.column(align=True)
        col.label(text="1/2/3键 : 切换 X/Y/Z 轴")
        col.label(text="拖拽 : 绘制 | 点击 : 编辑")
        col.label(text="X键 : 删除线条")
        col.label(text="右键 : 退出")
        col.label(text="--------------------------")
        col.label(text="建议:平行边只画一条或者不画")
        col.label(text="建议:透视边至少画三条或以上")
        
        layout.separator()
        
        # 4. 信息
        if scene.camera:
            col = layout.column(align=True)
            col.prop(scene.camera.data, "lens", text="焦距(mm)")
            col.prop(scene.camera.data, "sensor_width", text="传感器(mm)")
        else:
            layout.alert = True
            layout.label(text="警告: 场景无相机!", icon='ERROR')

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
