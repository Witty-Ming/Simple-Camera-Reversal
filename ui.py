import bpy

from . import utils


class CMP_PT_MainPanel(bpy.types.Panel):
    bl_label = "Simple Camera Match"
    bl_idname = "CMP_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'CameraMatch'

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        cmp_data = getattr(scene, "cmp_data", None)
        if cmp_data is None:
            warning = layout.row()
            warning.alert = True
            warning.label(text="CameraMatch data unavailable (add-on not registered)", icon='ERROR')
            return

        # 0. 背景图分辨率匹配（置于面板最上方）
        row = layout.row()
        row.scale_y = 1.5
        row.operator(
            "cmp.match_background_resolution",
            text="Match Background Resolution",
            icon='IMAGE_DATA',
        )

        layout.separator()

        is_perspective = bool(
            scene.camera
            and getattr(scene.camera.data, "type", None) == 'PERSP'
        )

        # 1. 绘制工具
        layout.label(text="Step 1: Drawing", icon='GREASEPENCIL')
        row = layout.row()
        row.scale_y = 1.5
        row.enabled = is_perspective
        row.operator("cmp.draw_line", text="Start Drawing (Brush)", icon='GREASEPENCIL')

        layout.separator()

        # 2. 解算工具
        layout.label(text="Step 2: Solve", icon='CHECKMARK')
        col = layout.column(align=True)
        col.scale_y = 1.3
        col.enabled = is_perspective
        col.operator("cmp.match_camera", text="Match Camera (Solve)", icon='CAMERA_DATA')

        # 线条检查状态：常驻显示，任何时刻都能看到当前输入能不能用。
        #   有异常       -> 红框 + NOT_FOUND 图标，逐行列出原因
        #   解算过无异常 -> 绿勾 NODE_SOCKET_GEOMETRY
        #   还没解算过   -> 中性提示（否则会把"没检查"误读成"检查通过"）
        #
        # scale_y 取 1.2 而不是 1.0：只有一行内容的 box 在深色主题下边框很淡、
        # 高度又只有一行，实测用户会直接忽略掉这一块（"你说的框在哪？"）。
        # 把行撑高一些，框的轮廓才看得见。
        warning_text = getattr(cmp_data, "last_solve_warning", "")
        solve_checked = bool(getattr(cmp_data, "last_solve_checked", False))
        check_box = layout.box()
        check_col = check_box.column(align=True)
        check_col.scale_y = 1.2
        if warning_text:
            # alert 必须在**子项创建之前**设在真正承载子项的那个 layout 上。
            # Blender 的 layout 只在创建子 layout/子控件的那一刻继承父级的
            # alert 标志：check_box.alert 改的是 box 自己，而 check_col 是在它
            # 之前就 column() 出来的、alert 仍是 False，所以 label 拿到的是
            # 普通文字色 —— 实测截图里文字是 (79,89,104)（偏蓝的灰），一点红
            # 都没有，就是这个原因。box 上也设一次，主题若支持会额外画描边。
            check_box.alert = True
            check_col.alert = True
            # 提示用换行分隔（Blender 的 label 不自动换行，拼成一整条会被截断）
            for part in [p.strip() for p in warning_text.split("\n") if p.strip()]:
                # 用空操作符当"红底标签"。Blender 的 alert 对纯 label 不生效
                # （实测同一 alert 容器里 label 文字始终是普通灰 (75,84,99)），
                # 只对 button / prop 这类带 widget 的控件生效 —— 换成 button
                # 后立刻拿到红底 (237,138,138) + 白字。所以这里挂
                # cmp.line_check_row（点击无副作用的空操作符）当画布。
                check_col.operator("cmp.line_check_row", text=part, icon='ERROR')
        elif solve_checked:
            check_col.label(text="Line check passed", icon='NODE_SOCKET_GEOMETRY')
        else:
            check_col.label(text="Not solved yet", icon='INFO')

        col.separator()
        col.separator()
        row = col.row()
        row.prop(cmp_data, "world_rotation", text="3D Cursor Rotation (XY)")
        row.prop(cmp_data, "flip_z_axis", text="Flip Z Axis",
                 icon='TRIA_UP' if not cmp_data.flip_z_axis else 'TRIA_DOWN', toggle=True)

        layout.separator()

        horizon_box = layout.box()
        horizon_box.enabled = is_perspective
        horizon_box.label(text="Horizon Constraint", icon='HIDE_OFF')
        horizon_box.prop(cmp_data, "horizon_enabled", text="Enable Horizon")

        layout.separator()

        # 3. 说明
        box = layout.box()
        box.label(text="Instructions:", icon='INFO')
        col = box.column(align=True)
        col.label(text="1/2/3 Key : Switch X/Y/Z Axis")
        col.label(text="Drag : Draw | Click : Edit")
        col.label(text="X Key : Delete Line")
        col.label(text="Esc / Right Click : Exit")
        col.label(text="--------------------------")
        col.label(text="Tip: Draw one or no parallel edges")
        col.label(text="Tip: Draw at least 3 perspective edges")

        layout.separator()

        # 4. 信息
        if is_perspective:
            col = layout.column(align=True)
            col.prop(cmp_data, "focal_length_mm", text="Focal Length (mm)")
            # 走 cmp_data 的 setter：改传感器尺寸后会自动重解算，
            # 否则 lens/shift 会与新传感器尺寸失去一致性。
            col.prop(cmp_data, "sensor_width_mm", text="Sensor (mm)")
        elif scene.camera:
            warning = layout.row()
            warning.alert = True
            warning.label(text="Only perspective cameras are supported", icon='ERROR')
        else:
            warning = layout.row()
            warning.alert = True
            warning.label(text="Warning: No Active Camera!", icon='ERROR')

        layout.separator()

        # 5. 希区柯克变焦（滑动变焦）：改焦距的同时相机沿光轴前后移动，
        #    3D 游标所在深度平面的构图完全不变，只有背景透视被压缩/扩张。
        zoom_box = layout.box()
        zoom_box.enabled = is_perspective
        zoom_box.label(text="Hitchcock Zoom (Dolly Zoom)", icon='ARROW_LEFTRIGHT')
        zoom_box.prop(
            cmp_data,
            "hitchcock_focal_mm",
            text="Dolly Focal Length (mm)",
            slider=True,
        )
        hint = zoom_box.column(align=True)
        hint.scale_y = 0.85
        hint.label(text="Camera dollies along its axis so the", icon='INFO')
        hint.label(text="3D Cursor keeps the same framing.")

        message = getattr(cmp_data, "last_hitchcock_message", "")
        if message:
            warn = zoom_box.row()
            warn.alert = True
            warn.label(text=message, icon='ERROR')


def register():
    utils.register_class_safe(CMP_PT_MainPanel)


def unregister():
    utils.unregister_class_safe(CMP_PT_MainPanel)
