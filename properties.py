import bpy

class CMP_Line(bpy.types.PropertyGroup):
    start: bpy.props.FloatVectorProperty(size=2, description="Start Point (Normalized)")
    end: bpy.props.FloatVectorProperty(size=2, description="End Point (Normalized)")
    axis: bpy.props.StringProperty(default='X', description="Axis (X, Y, Z)")

class CMP_SceneProperties(bpy.types.PropertyGroup):
    lines: bpy.props.CollectionProperty(type=CMP_Line)
    active_index: bpy.props.IntProperty(default=-1)
    
    is_drawing_mode: bpy.props.BoolProperty(default=False)
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
            rot_mat = mathutils.Matrix.Rotation(delta_rot, 4, 'Z')
            cam.matrix_world = rot_mat @ cam.matrix_world
            
        # 2. 处理翻转 Delta
        if self.flip_z_axis != self.last_flip_z:
            self.last_flip_z = self.flip_z_axis
            flip_mat = mathutils.Matrix.Rotation(math.pi, 4, 'X')
            cam.matrix_world = flip_mat @ cam.matrix_world
            
    world_rotation: bpy.props.FloatProperty(
        name="World Rotation",
        description="Rotate camera around world origin",
        default=0.0,
        min=-3.1415926,
        max=3.1415926,
        subtype='ANGLE',
        unit='ROTATION',
        update=update_rotation
    )
    
    flip_z_axis: bpy.props.BoolProperty(
        name="Flip Z Axis",
        description="Flip world Z axis direction (rotate 180 degrees around X axis)",
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
