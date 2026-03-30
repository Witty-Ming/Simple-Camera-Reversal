
import bpy
try:
    cam = bpy.data.cameras.new("TestCam")
    print(f"传感器高度存在: {hasattr(cam, 'sensor_height')}")
    print(f"传感器宽度: {cam.sensor_width}")
    print(f"传感器高度: {cam.sensor_height}")
    print(f"传感器适配: {cam.sensor_fit}")
except Exception as e:
    print(f"错误: {e}")
