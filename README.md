# Simple Camera Match (简易相机反求插件)

这是一个用于 Blender 的轻量级、高效率相机反求与透视匹配插件。它允许你直接在相机视口中对照着背景参考图，通过手动绘制 X、Y、Z 三个维度的透视消失线，来实时、自动地解算出相机的精准焦距、位置与旋转角度。

---

## 🚀 核心特性

- **直观的交互绘制**：完全在相机视口的 2D 空间内操作，实时视觉反馈，所见即所得。
- **高阶地平线约束 (Horizon Constraint)**：在地平线系统辅助下，支持直接拖拽地平线把手来灵活微调相机的俯仰视角 (Pitch)。
- **丝滑的多相机支持**：无论是切换相机还是新建相机，插件都能在后台自动实现无缝接管与数据清理。
- **历史记录安全**：内置独立的绘制操作撤销 / 重做栈，避免与 Blender 庞大混乱的全局撤销树冲突。

---

## 📖 详细使用说明

### 第一步：准备好你的相机背景图

### 第二步：绘制你的透视线
在 `CameraMatch` 侧边栏中点击最显眼的 **Draw Reference Line (绘制参考线)** 按钮，此时你会进入沉浸式的交互绘制模式（观察屏幕顶部的操作提示栏会改变）：

#### 🎨 基础绘制操作：
- **画线**：在视口空白处 `左键点击并拖动`，即可拉出一条透视参考线。
- **改线**：先点击线段中点激活该线段，再用左键拖拽端点修改位置。
- **删线**：先点击线段中点进入编辑状态，再按下键盘 **`X`** 键删除当前线段。
- **退出**：按下键盘 **`Esc`** 或是单击鼠标 `右键` 随时退出绘制模式。

#### ⌨️ 切换维度轴向 (非常重要)：
*绘制透视线时，你需要告诉插件你这根线代表什么空间朝向。*
- 按键盘 **`1`** (或小键盘 1)：切换进入 **X 轴向**（红色，通常代表物体的宽度或左右深度）。
- 按键盘 **`2`** (或小键盘 2)：切换进入 **Y 轴向**（绿色，通常代表物体的另一侧深度）。
- 按键盘 **`3`** (或小键盘 3)：切换进入 **Z 轴向**（蓝色，通常代表垂直于地面的高度）。
*(建议：找准相片中的建筑、盒子等直角参照物，分别顺着它们的边缘画出 X、Y 和 Z 轴)*
***单点透视***:只绘制一个轴向的透视线段,当你需要物体时正确朝上的的时候,只推荐绘制Y轴(键盘2)绘制

#### 🎯 绘制辅助与微调：
- **微调减速**：在拖拽线条端点时，按住 **`Shift`** 键，鼠标的移动速度将被减弱，非常适合用来将线点完美贴合到背景图的像素边缘。
- **正交锁定线段**：在画线拖动时（或者准备画之前），按一次键盘上的 **`X`** 键，可强制让线段保持**绝对水平**；按一次 **`Y`** 键，可让线段保持**绝对垂直**。（再次按下对应按键即可解除锁定）

#### 🧹 清理与重置：
- 撤销 / 重做：绘制画错时，按 **`Ctrl + Z`** (**Mac 为 `Cmd + Z`**) 撤销上一步线段编辑。 也可以配合 `Shift` 进行重做。
- **终极清理键**：按下 **`Alt + X`** 将**瞬间清空当前相机的所有相关透视线以及全部解算偏移缓存**，相当于对当前视角来一张完美的纯净“白纸”，在匹配陷入混乱想要重新开始时极其好用！

---

## 🌅 高级功能：地平线把控 (Horizon Constraint)

在绘制面板的下方，你能看到一个 `Horizon Constraint`（地平线约束）模块。
当地平线被开启后，插件会根据你画出的 X 轴、Y 轴自动推测并展示出一条蓝色的天际线。

👉 **拖拽地平线把手**（需要保持绘制模式）： 
1. 观察视口，在地平线的中央你会看到一个稍大一点的 **菱形手柄**。
2. 在绘制模式中，将鼠标按住该手柄**上下拖动**，插件就能强行扭转覆盖相机的俯仰角 (Pitch)。退出绘制模式后，参考线和地平线叠加层会被关闭。
3. **注意：** 拖拽手柄是一种由你主观引入的“强制偏差叠加”。因此，**每当你之后再度尝试去改动或绘制核心的透视线段时，之前手动拖拉产生的地平线偏移量将会被立刻重置清零**。保证最严谨的数学解算永远是以透视线为尊。



<br><br><br>

---
---

# Simple Camera Match (English Version)

This is a lightweight and highly efficient plugin for camera reconstruction and perspective matching in Blender. It allows you to automatically and precisely solve the camera's focal length, location, and rotation in real-time by manually drawing X, Y, and Z vanishing lines directly within the camera viewport, using a background image as reference.

---

## 🚀 Core Features

- **Intuitive Interactive Drawing**: Operate entirely within the 2D space of the camera viewport with real-time visual feedback. What you see is what you get.
- **Advanced Horizon Constraint**: Supported by a horizon visual system, it allows you to flexibly tweak the camera's pitch by directly dragging the horizon handle.
- **Seamless Multi-Camera Support**: Whether you switch between cameras or create a new one, the plugin automatically handles data cleanup and seamless transitions in the background.
- **Safe History Tracking**: Features an independent undo/redo stack for drawing operations to avoid conflicts with Blender’s global undo history.

---

## 📖 Detailed Instructions

### Step 1: Prepare Your Camera 
Make sure you have at least one camera tracking your scene. Navigate to Camera View (Numpad 0) and add your reference images via the Camera Properties -> Background Images panel.

### Step 2: Draw Your Perspective Lines
Click the prominent **Draw Reference Line** button in the `CameraMatch` sidebar to enter the immersive interactive drawing mode:

#### 🎨 Basic Drawing Operations:
- **Draw Line**: `Left Click and Drag` in an empty area of the viewport to create a perspective reference line.
- **Edit Line**: Click the middle dot to activate a line, then left-click and drag one of its endpoints.
- **Delete Line**: Click the middle dot of a line to enter edit mode, then press **`X`** to delete the active line.
- **Exit**: Press **`Esc`** or `Right Click` to exit drawing mode at any time.

#### ⌨️ Switch Dimension Axes (Crucial):
*When drawing perspective lines, you must tell the plugin which spatial direction the line represents.*
- Press **`1`** (or Numpad 1): Switch to the **X Axis** (Red, usually represents width or lateral depth).
- Press **`2`** (or Numpad 2): Switch to the **Y Axis** (Green, usually represents the other side's depth).
- Press **`3`** (or Numpad 3): Switch to the **Z Axis** (Blue, usually represents height perpendicular to the ground).
*(Tip: Find orthogonal references like buildings or boxes in the photo, and draw X, Y, and Z axes along their edges respectively.)*

***Single-Point Perspective***: Draw perspective lines for only one axis. When you need reconstructed objects to orient perfectly upright, it is strictly recommended to only draw the **Y Axis (Key 2)**.

#### 🎯 Drawing Aids and Fine-Tuning:
- **Precision Slowdown**: Hold **`Shift`** while dragging an endpoint to decrease mouse movement speed. This is perfect for perfectly aligning points to pixel edges of the background image.
- **Orthogonal Lock**: While dragging to draw a line (or before drawing), press **`X`** once to force the line to be **absolutely horizontal**; press **`Y`** once to force it to be **absolutely vertical**. (Press the key again to unlock).

#### 🧹 Cleanup and Reset:
- **Undo / Redo**: If you make a mistake, press **`Ctrl + Z`** (**`Cmd + Z` on Mac**) to undo the last line edit. Use with `Shift` to redo.
- **Ultimate Reset Key**: Press **`Alt + X`** to **instantly clear all perspective lines and solve caches for the current camera**. This provides a perfectly clean "blank slate" for the current view. It is extremely useful when your match becomes messy and you want to start over!

---

## 🌅 Advanced Feature: Horizon Constraint

Below the drawing panel, you will find the `Horizon Constraint` module.
When enabled, the plugin automatically infers and displays a blue horizon line based on your drawn X and Y axes.

👉 **Drag the Horizon Handle** (drawing mode required):
1. Look at the viewport. You will see a slightly larger **diamond-shaped handle** in the center of the horizon.
2. While drawing mode is active, click and **drag this handle up or down** to forcefully adjust the camera's Pitch. The guide overlay is removed when drawing mode ends.
3. **Note:** Dragging the handle introduces an intentional, manual "offset variation". Therefore, **whenever you subsequently attempt to modify or draw core perspective lines, any previously dragged manual horizon offset will be immediately reset to zero**. This ensures that rigorous mathematical solving is always prioritized based on your perspective lines.

## 🤖 Automation Details
If you are immersed in matching lines for one camera and suddenly switch to another camera in the Outliner—**Don't worry!**
The plugin instantly detects the viewport change and **automatically wipes all the messy history from the previous camera**. You can immediately start drawing your new camera with zero errors or interruptions!
