import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader
import bpy_extras
from mathutils import Vector
from bpy_extras import view3d_utils

class CMP_OT_DrawLine(bpy.types.Operator):
    """绘制 2D 相机匹配参考线 (重构为 3D)"""
    bl_idname = "cmp.draw_line"
    bl_label = "绘制参考线"
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
    DRAG_THRESHOLD = 10 # 像素
    
    def invoke(self, context, event):
        if not context.scene.camera:
            self.report({'WARNING'}, "请先添加摄像机")
            return {'CANCELLED'}
            
        self.state = self.STATE_IDLE
        self.current_axis = 'X'
        self.last_error = "" # 初始化 last_error
        context.scene.cmp_data.active_index = -1
        context.scene.cmp_data.is_drawing_mode = True 
        
        context.window_manager.modal_handler_add(self)
        self.update_header(context)
        context.area.tag_redraw()
        return {'RUNNING_MODAL'}
        
    def update_header(self, context):
        try:
            cols = {'X':'红X', 'Y':'绿Y', 'Z':'蓝Z'}
            c = cols.get(self.current_axis, "")
            base = f"CameraMatch [3D模式]: 当前轴 {c} (1/2/3). 拖拽绘制 | 点圆点编辑 | Ctrl+Z 撤销 | 右键退出"
            
            if self.last_error:
                msg = f"{base} | 错误: {self.last_error}"
            else:
                msg = base
                
            context.area.header_text_set(msg)
        except: pass

    def modal(self, context, event):
        context.area.tag_redraw()
        
        if event.type in {'RIGHTMOUSE', 'ESC'}:
            self.quit(context)
            return {'FINISHED'}
            
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
            return {'PASS_THROUGH'}
            
        # 快捷键
        if event.value == 'PRESS':
            if event.ctrl and event.type == 'Z':
                # 撤销上一条线
                lines = context.scene.cmp_data.lines
                if len(lines) > 0:
                    lines.remove(len(lines)-1)
                    context.scene.cmp_data.active_index = -1
                    self.state = self.STATE_IDLE
                    self.trigger_solve(context)
                return {'RUNNING_MODAL'}
                
            if event.type in {'ONE', 'NUMPAD_1'}: self.current_axis = 'X'; self.update_header(context)
            elif event.type in {'TWO', 'NUMPAD_2'}: self.current_axis = 'Y'; self.update_header(context)
            elif event.type in {'THREE', 'NUMPAD_3'}: self.current_axis = 'Z'; self.update_header(context)
            elif event.type == 'X':
                if self.state == self.STATE_EDITING:
                    lines = context.scene.cmp_data.lines
                    idx = context.scene.cmp_data.active_index
                    if 0 <= idx < len(lines):
                        lines.remove(idx)
                        context.scene.cmp_data.active_index = -1
                        self.state = self.STATE_IDLE
                        self.trigger_solve(context)
        
        # 逻辑
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
                    if self.active_handle == 0: line.start = norm
                    else: line.end = norm
                    self.trigger_solve(context)
            elif self.state == self.STATE_WAITING_DRAG:
                # 检查距离
                dist = np.hypot(x - self.drag_start_x, y - self.drag_start_y)
                if dist > self.DRAG_THRESHOLD:
                    # 距离足够，开始绘制
                    self.start_drawing(context, self.drag_start_x, self.drag_start_y)
                    # 立即更新当前位置到鼠标位置
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
                    # 不直接开始绘制，而是等待拖拽
                    self.drag_start_x = x
                    self.drag_start_y = y
                    self.state = self.STATE_WAITING_DRAG
                    # self.start_drawing(context, x, y)
            elif self.state == self.STATE_EDITING:
                h = self.check_endpoint_click(context, x, y, idx)
                if h != -1:
                    self.active_handle = h
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
                 # 松开时如果还在等待，说明没有拖拽 -> 是单击 -> 取消
                 self.state = self.STATE_IDLE
                 context.scene.cmp_data.active_index = -1
            elif self.state == self.STATE_DRAGGING:
                self.state = self.STATE_EDITING
                
        return {'RUNNING_MODAL'}

    def start_drawing(self, context, x, y):
        norm = self.screen_to_norm(context, x, y)
        if not norm: return
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
            print(e) # 调试
            pass

    def quit(self, context):
        
        context.scene.cmp_data.is_drawing_mode = False
        context.scene.cmp_data.is_creating_line = False
            
        context.area.header_text_set(None)
        context.window.cursor_modal_restore()

    # --- 辅助函数 ---
    def screen_to_norm(self, context, x, y):
        # 交互所需的逆投影
        region = context.region
        rv3d = context.region_data
        cam = context.scene.camera
        if not cam: return None
        try:
            vec = view3d_utils.region_2d_to_vector_3d(region, rv3d, (x, y))
            loc = view3d_utils.region_2d_to_origin_3d(region, rv3d, (x, y)) + vec * 10
            co = bpy_extras.object_utils.world_to_camera_view(context.scene, cam, loc)
            return Vector((co.x, co.y))
        except: return None
        
    # --- 碰撞测试 ---
    def check_dot_click(self, context, x, y):
        lines = context.scene.cmp_data.lines
        cam = context.scene.camera
        if not cam: return -1
        TR, TL, BL, BR = get_ordered_frame_points(context)
        if not TR: return -1 # 修复崩溃
        
        region = context.region; rv3d = context.region_data
        mw = cam.matrix_world
        
        m_vec = Vector((x, y))
        best_dist = 20
        best_idx = -1
        
        for i, line in enumerate(lines):
            top = TL.lerp(TR, line.start[0]); bot = BL.lerp(BR, line.start[0]); p1 = mw @ bot.lerp(top, line.start[1])
            top = TL.lerp(TR, line.end[0]); bot = BL.lerp(BR, line.end[0]); p2 = mw @ bot.lerp(top, line.end[1])
            
            s = view3d_utils.location_3d_to_region_2d(region, rv3d, p1)
            e = view3d_utils.location_3d_to_region_2d(region, rv3d, p2)
            
            if s and e:
                mid_2d = (s+e)/2
                if (m_vec - mid_2d).length < best_dist:
                    best_idx = i
        return best_idx

    def check_endpoint_click(self, context, x, y, idx):
        if idx == -1: return -1
        lines = context.scene.cmp_data.lines # Fix index out of range check needed?
        if idx >= len(lines): return -1
        line = lines[idx]
        cam = context.scene.camera
        TR, TL, BL, BR = get_ordered_frame_points(context)
        if not TR: return -1 # 修复崩溃
        
        region = context.region; rv3d = context.region_data
        mw = cam.matrix_world
        
        top = TL.lerp(TR, line.start[0]); bot = BL.lerp(BR, line.start[0]); p1 = mw @ bot.lerp(top, line.start[1])
        top = TL.lerp(TR, line.end[0]); bot = BL.lerp(BR, line.end[0]); p2 = mw @ bot.lerp(top, line.end[1])
        
        s = view3d_utils.location_3d_to_region_2d(region, rv3d, p1)
        e = view3d_utils.location_3d_to_region_2d(region, rv3d, p2)
        m = Vector((x, y))
        
        if s and (m-s).length < 20: return 0
        if e and (m-e).length < 20: return 1
        return -1
    


# --- 全局辅助函数 ---
def get_ordered_frame_points(context):
    cam = context.scene.camera
    if not cam: return None, None, None, None
    frame = cam.data.view_frame(scene=context.scene)
    
    TR, TL, BL, BR = None, None, None, None
    for v in frame:
        if v.x > 0 and v.y > 0: TR = v
        elif v.x < 0 and v.y > 0: TL = v
        elif v.x < 0 and v.y < 0: BL = v
        elif v.x > 0 and v.y < 0: BR = v
    
    # 如果找不到特定角（例如正交相机），回退到默认顺序
    if not all([TR, TL, BL, BR]):
        # 此顺序基于原始代码的回退
        TR, TL, BL, BR = frame[0], frame[1], frame[3], frame[2]
        
    return TR, TL, BL, BR

def register():
    try:
        bpy.utils.register_class(CMP_OT_DrawLine)
    except ValueError:
        bpy.utils.unregister_class(CMP_OT_DrawLine)
        bpy.utils.register_class(CMP_OT_DrawLine)

def unregister():
    try:
        bpy.utils.unregister_class(CMP_OT_DrawLine)
    except: pass
