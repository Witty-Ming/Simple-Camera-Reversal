import bpy
import numpy as np
import mathutils
from . import utils

def solve_camera_core(context):
    """
    核心解算函数
    :return: (success, message)
    """
    scene = context.scene
    cam = scene.camera
    if not cam: return False, "无激活相机"
    
    lines = scene.cmp_data.lines
    if len(lines) < 2: return False, "线条数量不足"
    
    render = scene.render
    res_x = render.resolution_x
    res_y = render.resolution_y
    
    cx = res_x / 2.0
    cy = res_y / 2.0
    
    # 1. 准备数据
    lines_data = {'X': [], 'Y': [], 'Z': []}
    
    for line in lines:
        u1, v1 = line.start
        u2, v2 = line.end
        
        px1 = u1 * res_x - cx
        py1 = v1 * res_y - cy
        px2 = u2 * res_x - cx
        py2 = v2 * res_y - cy
        
        dx = px2 - px1
        dy = py2 - py1
        length = np.hypot(dx, dy)
        
        if length < 10: continue # 放宽阈值
        
        a = -dy / length
        b = dx / length
        c = -(a * px1 + b * py1)
        

        lines_data[line.axis].append([a, b, c, length])
        
    # 重置旋转属性 (在解算前或后)
    # 策略：解算产生“归零”的状态。
    # 我们应该重置 UI 值为 0，并同步 last_value 以防触发 Delta。
    scene.cmp_data.last_world_rotation = 0.0
    scene.cmp_data.world_rotation = 0.0
    scene.cmp_data.last_flip_z = False
    scene.cmp_data.flip_z_axis = False
        
    # 2. 求解消失点 (VPs)
    vp_data = {}
    axis_weights = {} # 存储每轴线段数量
    
    for axis in ['X', 'Y', 'Z']:
        data = lines_data[axis]
        count = len(data)
        axis_weights[axis] = count
        
        if count >= 1: # 从 2 减少到 1 以允许单线
            if count >= 2:
                arr = np.array(data)
                lines_abc = arr[:, :3]
                weights = arr[:, 3]
                # 计算图像对角线以用于动态阈值
                image_diag = np.hypot(res_x, res_y)
                vp = utils.solve_vanishing_point_2d(lines_abc, weights, image_diag=image_diag)
                if vp is not None:
                    vp_data[axis] = vp
            else:
                # 单线：无法求解 VP，但会在约束求解器中处理
                pass
    
    # 3. 求解相机参数
    current_dist = cam.location.length
    if current_dist < 0.1: current_dist = 10.0
    
    # 检查是否有足够的 VP 进行完整求解
    if len(vp_data) >= 2:
        f_mm, rot_matrix, shift_x, shift_y, loc_orbit = utils.calculate_camera_transform(
            vp_data, 
            cam.data.sensor_width, 
            cam.data.sensor_height, 
            cam.data.sensor_fit, 
            res_x, res_y, current_dist, 
            default_f_mm=cam.data.lens,
            axis_weights=axis_weights
        )
    else:
        # 回退：约束旋转求解 (单线模式)
        # 至少需要有线条的 2 个轴
        active_axes = [a for a in ['X', 'Y', 'Z'] if len(lines_data[a]) >= 1]
        if len(active_axes) < 2:
            return False, "需至少两个轴向(每个轴至少1条线)"
            
        # 使用当前焦距
        f_mm = cam.data.lens
        shift_x = cam.data.shift_x
        shift_y = cam.data.shift_y
        
        # 计算 f_pixels
        f_pixels = utils.get_effective_f_pixels(
            f_mm, 
            cam.data.sensor_width, 
            cam.data.sensor_height, 
            cam.data.sensor_fit, 
            res_x, res_y
        )
            
        rot_matrix = utils.solve_camera_rotation_constrained(
            lines_data, f_pixels, cx, cy, cam.matrix_world.to_3x3()
        )
        
        if rot_matrix is None:
            return False, "单线模式解算失败"
            
        # 保持当前轨道逻辑      
        # 为稳定性强制偏移为 0
        shift_x = 0.0
        shift_y = 0.0
        
        principal_point = np.array([0.0, 0.0])
        ray_cam = np.array([0.0, 0.0, -f_pixels])
        ray_cam /= np.linalg.norm(ray_cam)
        p_org_cam = ray_cam * current_dist
        
        # 位置同样的修复
        loc_orbit = -(rot_matrix @ mathutils.Vector(p_org_cam))

    # 稳定性检查
    if f_mm is None:
        return False, "数学解算失败"
        
    if not (1.0 < f_mm < 10000.0): # 放宽限制
        return False, f"焦距异常: {f_mm:.1f}mm"
        
    if abs(shift_x) > 10.0 or abs(shift_y) > 10.0:
        return False, "Shift溢出"
        
    # 4. 应用 (安全应用)
    cam.data.lens = f_mm
    cam.data.shift_x = shift_x
    cam.data.shift_y = shift_y
    
    new_rot_4x4 = rot_matrix.to_4x4()
    
    # 基础解算结果
    cam.matrix_world = mathutils.Matrix.Translation(loc_orbit) @ new_rot_4x4
    
    # 注意：不再这里应用 world_rotation，因为现在它是基于 Delta 的后期调整。
    # 解算总是返回“未旋转”的初始状态。
    
    return True, f"成功: f={f_mm:.1f}mm, Shift=({shift_x:.2f}, {shift_y:.2f})"

class CMP_OT_MatchCamera(bpy.types.Operator):
    """根据绘制的线条解算相机"""
    bl_idname = "cmp.match_camera"
    bl_label = "匹配相机"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        success, msg = solve_camera_core(context)
        if success:
            context.view_layer.update()
            context.view_layer.update()
            # 显示详细报告
            self.report({'INFO'}, msg)
            print(f"[CameraMatchPro] {msg}") # 打印到系统控制台
            return {'FINISHED'}
        else:
            self.report({'WARNING'}, msg)
            return {'CANCELLED'}

class CMP_OT_ClearLines(bpy.types.Operator):
    """清除所有线条"""
    bl_idname = "cmp.clear_lines"
    bl_label = "清除所有线条"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        context.scene.cmp_data.lines.clear()
        context.area.tag_redraw()
        return {'FINISHED'}

def register():
    try:
        bpy.utils.register_class(CMP_OT_MatchCamera)
    except ValueError:
        bpy.utils.unregister_class(CMP_OT_MatchCamera)
        bpy.utils.register_class(CMP_OT_MatchCamera)
        
    try:
        bpy.utils.register_class(CMP_OT_ClearLines)
    except ValueError:
        bpy.utils.unregister_class(CMP_OT_ClearLines)
        bpy.utils.register_class(CMP_OT_ClearLines)

def unregister():
    try:
        bpy.utils.unregister_class(CMP_OT_ClearLines)
    except: pass
    
    try:
        bpy.utils.unregister_class(CMP_OT_MatchCamera)
    except: pass

