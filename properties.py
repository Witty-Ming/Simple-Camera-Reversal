import bpy

class CMP_Line(bpy.types.PropertyGroup):
    start: bpy.props.FloatVectorProperty(size=2, description="起点 (归一化)")
    end: bpy.props.FloatVectorProperty(size=2, description="终点 (归一化)")
    axis: bpy.props.StringProperty(default='X', description="轴 (X, Y, Z)")

class CMP_SceneProperties(bpy.types.PropertyGroup):
    lines: bpy.props.CollectionProperty(type=CMP_Line)
    # 新增: 用于 UI 同步的活动索引
    active_index: bpy.props.IntProperty(default=-1)
    
    # 新增: 标记是否处于绘制/编辑模式
    is_drawing_mode: bpy.props.BoolProperty(default=False)
    # 新增: 标记是否正在创建新线（拖拽中）
    is_creating_line: bpy.props.BoolProperty(default=False)
    
    # 内部变量，用于计算 Delta
    last_world_rotation: bpy.props.FloatProperty(default=0.0)
    last_flip_z: bpy.props.BoolProperty(default=False)

    def update_rotation(self, context):
        import mathutils
        import math
        
        cam = context.scene.camera
        if not cam: return
        
        # 1. 处理旋转 Delta
        delta_rot = self.world_rotation - self.last_world_rotation
        self.last_world_rotation = self.world_rotation
        
        if abs(delta_rot) > 1e-6:
            # 绕世界 Z 轴旋转
            rot_mat = mathutils.Matrix.Rotation(delta_rot, 4, 'Z')
            cam.matrix_world = rot_mat @ cam.matrix_world
            
        # 2. 处理翻转 Delta
        if self.flip_z_axis != self.last_flip_z:
            self.last_flip_z = self.flip_z_axis
            # 绕世界 X 轴翻转 180 度
            flip_mat = mathutils.Matrix.Rotation(math.pi, 4, 'X')
            cam.matrix_world = flip_mat @ cam.matrix_world
            
    world_rotation: bpy.props.FloatProperty(
        name="世界旋转",
        description="绕世界原点旋转相机",
        default=0.0,
        min=-3.1415926,
        max=3.1415926,
        subtype='ANGLE',
        unit='ROTATION',
        update=update_rotation
    )
    
    flip_z_axis: bpy.props.BoolProperty(
        name="翻转 Z 轴",
        description="翻转世界 Z 轴方向 (绕 X 轴旋转 180 度)",
        default=False,
        update=update_rotation
    )
    
def register():
    try:
        bpy.utils.register_class(CMP_Line)
    except ValueError:
        bpy.utils.unregister_class(CMP_Line)
        bpy.utils.register_class(CMP_Line)
        
    try:
        bpy.utils.register_class(CMP_SceneProperties)
    except ValueError:
        bpy.utils.unregister_class(CMP_SceneProperties)
        bpy.utils.register_class(CMP_SceneProperties)
        
    bpy.types.Scene.cmp_data = bpy.props.PointerProperty(type=CMP_SceneProperties)

def unregister():
    if hasattr(bpy.types.Scene, "cmp_data"):
        del bpy.types.Scene.cmp_data
        
    try:
        bpy.utils.unregister_class(CMP_SceneProperties)
    except: pass
    
    try:
        bpy.utils.unregister_class(CMP_Line)
    except: pass
