bl_info = {
    "name": "Simple Camera Match",
    "author": "WittyMing",
    "version": (1, 1, 2),
    "blender": (4, 2, 0),
    "location": "View3D > N-Panel > CameraMatch",
    "description": "Reconstruct camera perspective by drawing lines",
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
from . import translation

def register():
    translation.register()
    properties.register()
    operators.register()
    tool.register()
    ui.register()

def unregister():
    ui.unregister()
    tool.unregister()
    operators.unregister()
    gpu_draw.unregister()
    properties.unregister()
    translation.unregister()

if __name__ == "__main__":
    register()
