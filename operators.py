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
    if not cam: return False, "No Active Camera"
    
    lines = scene.cmp_data.lines
    if len(lines) < 2: return False, "Not enough lines"
    
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
        
        if length < 10: continue

        a = -dy / length
        b = dx / length
        c = -(a * px1 + b * py1)
        
        lines_data[line.axis].append([a, b, c, length])
        
    # 重置旋转属性
    scene.cmp_data.last_world_rotation = 0.0
    scene.cmp_data.world_rotation = 0.0
    scene.cmp_data.last_flip_z = False
    scene.cmp_data.flip_z_axis = False
        
    # 2. 求解消失点 (VPs)
    vp_data = {}
    axis_weights = {}
    
    for axis in ['X', 'Y', 'Z']:
        data = lines_data[axis]
        count = len(data)
        axis_weights[axis] = count
        
        if count >= 2:
            arr = np.array(data)
            lines_abc = arr[:, :3]
            weights = arr[:, 3]
            
            # 当只有2条线时，使用均匀权重（所有权重=1）
            # 因为2条线时线长度差异可能导致权重不平衡，从而产生不稳定的消失点
            if count == 2:
                weights = np.ones(count)
            
            image_diag = np.hypot(res_x, res_y)
            vp = utils.solve_vanishing_point_2d(lines_abc, weights, image_diag=image_diag)
            if vp is not None:
                vp_data[axis] = vp
    
    # 3. 求解相机参数（渐进式解算）
    # 使用3D游标作为世界原点锚定
    cursor_location = scene.cursor.location.copy()
    
    # 计算相机到3D游标的方向
    cam_to_cursor = cursor_location - cam.location
    current_dist = cam_to_cursor.length
    if current_dist < 0.1: current_dist = 10.0
    
    f_mm = None
    rot_matrix = None
    shift_x = 0.0
    shift_y = 0.0
    loc_orbit = None
    
    # 渐进式解算：优先使用最可靠的解算方法
    if len(vp_data) >= 2:
        try:
            f_mm, rot_matrix, shift_x, shift_y, loc_orbit = utils.calculate_camera_transform(
                vp_data, 
                cam.data.sensor_width, 
                cam.data.sensor_height, 
                cam.data.sensor_fit, 
                res_x, res_y, current_dist, 
                default_f_mm=cam.data.lens,
                axis_weights=axis_weights
            )
        except Exception as e:
            print(f"[CameraMatch] Full solve failed: {e}")
            f_mm = None
    
    # 回退方案：如果完整解算失败，尝试单线模式
    if f_mm is None or rot_matrix is None:
        active_axes = [a for a in ['X', 'Y', 'Z'] if len(lines_data[a]) >= 1]
        
        if len(active_axes) < 2:
            iface_ = bpy.app.translations.pgettext_iface
            return False, iface_("Requires at least two axes with valid lines")
        
        try:
            f_mm = cam.data.lens
            shift_x = 0.0
            shift_y = 0.0
            
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
                iface_ = bpy.app.translations.pgettext_iface
                return False, iface_("Rotation solving failed. Try drawing more lines.")
            
            principal_point = np.array([0.0, 0.0])
            ray_cam = np.array([0.0, 0.0, -f_pixels])
            ray_cam /= np.linalg.norm(ray_cam)
            p_org_cam = ray_cam * current_dist
            
            loc_orbit = -(rot_matrix @ mathutils.Vector(p_org_cam))
            
            iface_ = bpy.app.translations.pgettext_iface
            print(f"[CameraMatch] Using fallback mode (single-line)")
            
        except Exception as e:
            print(f"[CameraMatch] Fallback solve failed: {e}")
            iface_ = bpy.app.translations.pgettext_iface
            return False, iface_("Solving failed. Check line placement.")

    # 稳定性检查
    if f_mm is None:
        iface_ = bpy.app.translations.pgettext_iface
        return False, iface_("Math solving failed")
    
    # 数值有效性检查
    if not np.isfinite(f_mm):
        iface_ = bpy.app.translations.pgettext_iface
        return False, iface_("Invalid focal length value")
        
    if not (1.0 < f_mm < 10000.0):
        iface_ = bpy.app.translations.pgettext_iface
        return False, iface_("Abnormal focal length: ") + f"{f_mm:.1f}mm"
    
    # 检查 shift 是否有效
    if not (np.isfinite(shift_x) and np.isfinite(shift_y)):
        shift_x = 0.0
        shift_y = 0.0
        print("[CameraMatch] Warning: Invalid shift values, reset to 0")
        
    if abs(shift_x) > 10.0 or abs(shift_y) > 10.0:
        shift_x = 0.0
        shift_y = 0.0
        print("[CameraMatch] Warning: Shift overflow, reset to 0")
    
    # 检查旋转矩阵有效性
    if rot_matrix is not None:
        try:
            # 检查矩阵是否包含无效值
            mat_array = np.array(rot_matrix)
            if not np.all(np.isfinite(mat_array)):
                iface_ = bpy.app.translations.pgettext_iface
                return False, iface_("Invalid rotation matrix")
        except Exception as e:
            print(f"[CameraMatch] Matrix validation error: {e}")
    
    # 检查位置有效性
    if loc_orbit is not None:
        try:
            loc_array = np.array(loc_orbit)
            if not np.all(np.isfinite(loc_array)):
                # 回退到当前位置
                loc_orbit = cam.location.copy()
                print("[CameraMatch] Warning: Invalid location, using current position")
        except Exception as e:
            print(f"[CameraMatch] Location validation error: {e}")
            loc_orbit = cam.location.copy()
        
    # 4. 应用
    try:
        cam.data.lens = f_mm
        cam.data.shift_x = shift_x
        cam.data.shift_y = shift_y
        
        new_rot_4x4 = rot_matrix.to_4x4()
        
        cam.matrix_world = mathutils.Matrix.Translation(loc_orbit) @ new_rot_4x4
        
        iface_ = bpy.app.translations.pgettext_iface
        msg = iface_("Success: ") + f"f={f_mm:.1f}mm," + iface_(" Shift=") + f"({shift_x:.2f}, {shift_y:.2f})"
        return True, msg
        
    except Exception as e:
        print(f"[CameraMatch] Apply transform failed: {e}")
        iface_ = bpy.app.translations.pgettext_iface
        return False, iface_("Failed to apply camera transform")

class CMP_OT_MatchCamera(bpy.types.Operator):
    """Solve camera based on drawn lines"""
    bl_idname = "cmp.match_camera"
    bl_label = "Match Camera"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        success, msg = solve_camera_core(context)
        if success:
            context.view_layer.update()
            context.view_layer.update()
            self.report({'INFO'}, msg)
            print(f"[CameraMatchPro] {msg}")
            return {'FINISHED'}
        else:
            self.report({'WARNING'}, msg)
            return {'CANCELLED'}

def register():
    try:
        bpy.utils.register_class(CMP_OT_MatchCamera)
    except ValueError:
        bpy.utils.unregister_class(CMP_OT_MatchCamera)
        bpy.utils.register_class(CMP_OT_MatchCamera)

def unregister():
    try:
        bpy.utils.unregister_class(CMP_OT_MatchCamera)
    except: pass
