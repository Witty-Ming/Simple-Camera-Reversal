bl_info = {
    "name": "简易相机反求",
    "author": "WittyMing",
    "version": (1, 0),
    "blender": (3, 0, 0),
    "location": "View3D > N-Panel > CameraMatch",
    "description": "通过绘制线条反求相机视角",
    "warning": "",
    "doc_url": "",
    "category": "Camera",
}

import bpy
from . import properties
from . import gpu_draw
from . import ui
from . import operators
from . import tool

def register():
    properties.register()
    gpu_draw.register()
    ui.register()
    operators.register()
    tool.register()

def unregister():
    tool.unregister()
    operators.unregister()
    ui.unregister()
    gpu_draw.unregister()
    properties.unregister()

if __name__ == "__main__":
    register()
