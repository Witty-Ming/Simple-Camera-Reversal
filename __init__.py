bl_info = {
    "name": "Simple Camera Match",
    "author": "WittyMing",
    "version": (1, 4, 0),
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


def _safe(step, func):
    """单步清理失败不能阻断后续清理（否则 draw handler 会永久泄漏）。"""
    try:
        func()
    except Exception as e:
        print(f"[SimpleCameraMatch] unregister step '{step}' failed: {e}")


def unregister():
    _safe("ui", ui.unregister)
    _safe("tool", tool.unregister)
    _safe("operators", operators.unregister)
    # gpu_draw 的绘制回调与定时器必须无条件清理（force=True 绕过引用计数），
    # 否则重载插件后会留下重复的 draw handler。
    _safe("gpu_draw", lambda: gpu_draw.unregister(force=True))
    # 插件重载后 is_drawing_mode 等运行时标志若残留为 True，叠加层会在没有
    # 绘制模式的情况下继续画控制点。
    _safe("runtime_state", gpu_draw.reset_runtime_state)
    _safe("properties", properties.unregister)
    _safe("translation", translation.unregister)


if __name__ == "__main__":
    register()
