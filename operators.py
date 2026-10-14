import bpy
import numpy as np
import mathutils
import bpy_extras
from . import utils, properties


HORIZON_LOCK_THRESHOLD_PX = 8.0

# 线条一致性阈值（度）。实测（三条同轴长线、端点独立高斯噪声）：
#   ±8px 极端手抖下"好线"的留一法夹角 p50 4.2 / p90 9.7 / 最大 20.7 度
#   第三条线偏离 20 度 -> 76.8 度；偏离 45 度 -> 47.9 度；偏离 90 度 -> 42.7 度
# 所以 30 度既能容忍极端手抖，又能稳定抓住"画错轴"的线。
CONSISTENCY_ANGLE_LIMIT_DEG = 30.0


def collect_input_warnings(lines_data, pixel_res_x, pixel_res_y, lines,
                           principal_u=0.5, principal_v=0.5,
                           vp_data_raw=None, solve_mode=None):
    """只看线段输入判断"能不能解、解出来可不可信"，完全不碰相机。

    返回警告字符串列表。两处共用同一实现，避免阈值/文案漂移：
      * solve_camera_core 解算成功时用它写面板状态；
      * CameraMatchTool.refresh_line_check 在绘制过程中做轻量刷新 —— 实时解算
        会被"少于 2 条线"和帧率节流挡掉，如果检查框只在解算里更新，用户画完/
        改完线就会一直看到旧状态（"这个框怎么从来不变红"）。

    三项检查：
      1. 某个轴画了 >=2 条线却求不出消失点 => 这些线互相平行。此时该轴对旋转
         几乎没有约束力（实测 2 条平行线会给出 78° 的旋转误差），但过去仍报
         "成功"。
      2. 能定义消失点的轴少于 2 个 => 焦距在数学上不可确定（残差随焦距单调
         变化、越大越小），插件会保持焦距不动，必须说明原因和解决办法。
      3. 同一轴的几条线根本不是一组空间平行线。判据用**留一法夹角**：轮流拿掉
         每条线，用剩下的线求消失点，再看被拿掉的那条是否也指向同一个消失点。
         不要用"线段到消失点的垂距" —— 实测好线在 ±8px 端点噪声下垂距 p50 就有
         98px、最大 1578px，而画错 90 度的线也才 989px，两者完全重叠；夹角判据
         把垂距除以"中点到消失点的距离"，正好抵消这个放大。
         留一法要求该轴至少 3 条线（2 条线永远"精确"交于一点，无从判断），
         max_angle 为 None 时按"无法判断"处理，不算问题。
    """
    parts = []

    # solve_camera_core 是在函数体内取的这个翻译函数，模块级没有；本函数要被
    # tool.py 的轻量刷新直接调用，所以自己取一份。
    iface_ = bpy.app.translations.pgettext_iface

    if vp_data_raw is None:
        try:
            vp_data_raw, _ = utils.solve_vanishing_points(lines_data, pixel_res_x, pixel_res_y)
        except Exception:
            vp_data_raw = {}

    parallel_axes = [
        axis for axis in ('X', 'Y', 'Z')
        if len(lines_data.get(axis, [])) >= 2 and axis not in vp_data_raw
    ]
    if parallel_axes:
        parts.append(iface_("Parallel lines (no vanishing point) on axis ") + "/".join(parallel_axes))

    vp_capable_axes = [axis for axis in ('X', 'Y', 'Z') if len(lines_data.get(axis, [])) >= 2]
    if len(vp_capable_axes) < 2 and solve_mode != 'ONE_POINT':
        parts.append(iface_("Focal length not solvable: needs 2+ lines on each of two axes"))

    try:
        consistency = utils.compute_axis_consistency(
            lines_data, pixel_res_x, pixel_res_y,
            lines=lines, principal_u=principal_u, principal_v=principal_v)
        poor_axes = []
        for axis in ('X', 'Y', 'Z'):
            info = consistency.get(axis)
            if not info:
                continue
            angle = info.get('max_angle')
            if angle is not None and float(angle) > CONSISTENCY_ANGLE_LIMIT_DEG:
                poor_axes.append(axis)
    except Exception as e:
        print(f"[CameraMatch] Axis consistency check failed: {e}")
        poor_axes = []
    if poor_axes:
        parts.append(iface_("Inconsistent lines (not one family of parallel lines) on axis ") + "/".join(poor_axes))

    return parts


def solve_camera_core(context, fast=False):
    """解算入口（薄包装）：把"解算失败"也变成面板上的红框。

    `_solve_camera_core_impl` 有十几处提前 return False —— 没有活动相机、线段
    属于别的相机、线段数量不足、线段全在画面外、约束解算失败、焦距异常……这些
    分支过去只把原因塞进返回的 msg，而 `last_solve_checked` 仍是 False，面板
    于是显示中性的"尚未解算"。用户根本分不清"还没试过"和"试过但失败了"，
    反馈就是"这个框怎么从来不变红"。这里统一补上：失败同样算"检查过了"，
    并把失败原因交给面板显示（红色 + NOT_FOUND 图标）。
    """
    ok, msg = _solve_camera_core_impl(context, fast=fast)
    try:
        cmp_data = context.scene.cmp_data
        if not ok:
            cmp_data.last_solve_warning = str(msg)
        cmp_data.last_solve_checked = True
    except Exception:
        pass
    return ok, msg


def _solve_camera_core_impl(context, fast=False):
    """
    核心解算函数

    :param fast: True 表示绘制/拖拽过程中的实时解算：降低焦距扫描密度以保住
                 交互帧率，并加强"向当前焦距正则化"的强度（手工绘制的线段
                 不可能精确交于同一消失点，实测端点差几像素消失点就能偏出
                 几十像素；由于 current_f 在拖拽中就是上一帧的结果，该正则项
                 实际起时间平滑作用）。松手与点击 Match Camera 按钮时用
                 False：完整扫描精度 + 弱正则，保证最终解算精度。
    :return: (success, message)
    """
    scene = context.scene
    cam = scene.camera
    if not cam:
        return False, "No Active Camera"

    iface_ = bpy.app.translations.pgettext_iface

    if getattr(cam.data, "type", None) != 'PERSP':
        return False, iface_("Only perspective cameras are supported")

    cmp_data = scene.cmp_data

    # 每次解算重新评估：上一次的警告不能留到这一次，否则用户补好线之后提示
    # 仍然挂在面板上，反而误导。checked 标志一并清掉，面板回到"尚未解算"，
    # 直到本次解算成功后才重新置位。
    cmp_data.last_solve_warning = ""
    cmp_data.last_solve_checked = False

    # 参考线是"属于某台相机"的。相机切换后的自动清理只发生在绘制模式里，
    # 一旦 DrawLine 没在运行（Esc 退出、重开文件、直接点面板按钮），这里就是
    # 唯一的防线 —— 否则会把 A 相机的线解算到 B 相机上（实测会把 80mm 的
    # 相机直接搬成 35mm）。
    lines_cam = getattr(cmp_data, "lines_camera", None)
    if lines_cam is not None and lines_cam != cam:
        return False, iface_("Guide lines belong to another camera")
    if lines_cam is None:
        # 未绑定（例如快照恢复失败）：认领给当前相机，避免功能卡死
        cmp_data.lines_camera = cam

    lines = cmp_data.lines
    # 单点透视（ONE_POINT）只需要一个轴向的线段即可解算：另外两个轴向会由
    # build_perspective_mode_constraints 注入"画面水平/垂直"合成约束补足。
    if len(lines) < 1:
        return False, "Not enough lines"

    # 完全落在画面外的参考线没有意义（用户看不到背景图就没法对齐），
    # 全部越界时直接拒绝，避免解出 f=11mm 这种荒谬结果还报"成功"。
    if utils.count_offscreen_lines(lines) >= len(lines):
        return False, iface_("All guide lines are outside the camera frame")

    render = scene.render
    pixel_res_x, pixel_res_y = utils.get_effective_render_size(render)

    # 1. 计算真实的像主点 (考虑 shift_x/y)
    cursor_location = scene.cursor.location.copy()
    current_dist = (cam.location - cursor_location).length
    if current_dist < 0.1: current_dist = 10.0

    principal_u, principal_v = utils.compute_principal_point_uv(scene, cam)

    # 2. 准备数据
    lines_data, vp_data_raw, vp_data, axis_weights, horizon_data = utils.solve_horizon_data(
        lines,
        pixel_res_x,
        pixel_res_y,
        scene.cmp_data.horizon_enabled,
        scene.cmp_data.horizon_offset_px,
        principal_u,
        principal_v,
    )

    active_axes = [axis for axis in ('X', 'Y', 'Z') if len(lines_data.get(axis, [])) >= 1]
    if len(active_axes) < 1:
        iface_ = bpy.app.translations.pgettext_iface
        return False, iface_("Requires at least one axis (min 1 line per axis)")

    perspective_constraints = utils.build_perspective_mode_constraints(
        lines_data,
        pixel_res_x,
        pixel_res_y,
        finite_vp_axes=list(vp_data_raw.keys()),
    )

    # 保留当前的旋转微调状态，并在新解算结果上重新应用
    current_world_rotation = scene.cmp_data.world_rotation
    current_flip_z = scene.cmp_data.flip_z_axis
    current_shift_x = cam.data.shift_x
    current_shift_y = cam.data.shift_y
    cursor_location = scene.cursor.location.copy()
    current_dist = (cam.location - cursor_location).length
    if current_dist < 0.1: current_dist = 10.0

    anchor_screen_offset = None
    target_cursor_uv = None
    horizon_target_uv = None
    horizon_lock_active = False
    try:
        cursor_view = bpy_extras.object_utils.world_to_camera_view(scene, cam, cursor_location)

        # principal_u 和 principal_v 已经在开头计算好了

        if np.isfinite(cursor_view.x) and np.isfinite(cursor_view.y):
            target_cursor_uv = (float(cursor_view.x), float(cursor_view.y))
            anchor_screen_offset = (
                float((cursor_view.x - principal_u) * pixel_res_x),
                float((cursor_view.y - principal_v) * pixel_res_y),
            )

            if scene.cmp_data.horizon_enabled and horizon_data is not None:
                cursor_px = utils.uv_to_centered_px(target_cursor_uv, pixel_res_x, pixel_res_y, principal_u, principal_v)
                dist_to_horizon = abs(utils.signed_distance_to_line_2d(cursor_px, horizon_data['line']))
                if dist_to_horizon <= HORIZON_LOCK_THRESHOLD_PX:
                    proj_px = utils.project_point_to_line_2d(cursor_px, horizon_data['point'], horizon_data['direction'])
                    horizon_target_uv = utils.centered_px_to_uv(proj_px, pixel_res_x, pixel_res_y, principal_u, principal_v)

                    proj_offset = (
                        float((horizon_target_uv[0] - principal_u) * pixel_res_x),
                        float((horizon_target_uv[1] - principal_v) * pixel_res_y),
                    )
                    anchor_screen_offset = proj_offset
                    horizon_lock_active = True
    except Exception:
        anchor_screen_offset = None

    # 符号消歧与旋转迭代的种子必须是"不含手动微调"的基础姿态。
    #
    # 手动微调（world_rotation / flip_z_axis）是用户主观叠加的旋转，与几何
    # 约束无关。若把它算进种子，解算会在符号歧义处选到另一组解 —— 等价于把
    # 微调"吸收"进基础姿态，而应用阶段又会重新应用一次微调，两次叠加的结果
    # 与用户预期完全无关。实测：手动微调 π 后再点解算，得到的姿态与微调前
    # 一模一样；再解算一次又变回去，来回震荡。
    #
    # properties.update_rotation 的应用顺序是
    #     M = A_flip @ A_rot @ M_base        （都绕世界轴、绕游标左乘）
    # 所以剥离时按 A_rot⁻¹ @ A_flip⁻¹ 左乘。
    def _base_seed_rot():
        m = cam.matrix_world.to_3x3().copy()
        if current_flip_z:
            m = mathutils.Matrix.Rotation(-math.pi, 3, 'X') @ m
        if abs(current_world_rotation) > 1e-9:
            m = mathutils.Matrix.Rotation(-current_world_rotation, 3, 'Z') @ m
        return m

    # 焦距正则化的参考值（也是"从相机当前状态出发"的先验）。
    #
    # 实时模式必须向"上一帧的焦距"正则化才不会逐帧乱跳（实测端点加 ±8px
    # 噪声就能让"最优残差"的谷底从 35mm 偏到 32~45mm）。精确模式同样取相机
    # 当前的 lens，而不是扫描中心 —— 扫描中心是消失点法给出的焦距，它在噪声
    # 下可能离谱到 +131%（实测 level_side 视角：真值 1866.67px 的焦距被估成
    # 4304.44px）。若拿它当正则基准，扫描会被这个错误值锁住，越错越锁。
    # 相机 lens 是独立于消失点估计的先验：无噪声下残差谷底很深，实测
    # 20/50/85/135 各种初值都收敛到 35.000，所以用它做基准不会妨碍收敛，
    # 又不会放大消失点的错误。
    smooth_ref_mm = float(cam.data.lens)

    # 解算前的焦距，用于实时模式的平滑（见应用阶段）
    f_mm = None
    rot_matrix = None
    shift_x = 0.0
    shift_y = 0.0
    loc_orbit = None
    solve_mode_hint_key = None

    def solve_strict_mode(
        lines_for_solve,
        f_seed_mm,
        hint_refined_key,
        hint_locked_key,
        allow_focal_refine=True,
        hint_fixed_key=None,
        rot_seed=None,
        smooth_ref_mm=None,
        extra_seeds=None,
        multistart_random=None,
    ):
        try:
            rot_init = rot_seed if rot_seed is not None else _base_seed_rot()
            # 多起点：交替投影只做局部收敛，实测从"相机当前姿态"出发会卡在残差
            # 0.1441 的局部极小（相机被解成仰视 29.6 度），多起点能找到 0.0238 的
            # 解（俯视 12.4 度）。exact 用 24 个随机起点，fast 用 8 个（实时开销敏感）。
            ms_random = (8 if fast else 24) if multistart_random is None else int(multistart_random)
            strict_result = utils.solve_strict_mode_constrained(
                lines_for_solve,
                f_seed_mm,
                cam.data.sensor_width,
                cam.data.sensor_height,
                cam.data.sensor_fit,
                pixel_res_x,
                pixel_res_y,
                rot_init,
                allow_focal_refine=allow_focal_refine,
                fast=fast,
                smooth_ref_mm=smooth_ref_mm,
                extra_seeds=extra_seeds,
                multistart_random=ms_random,
            )
            if not strict_result.get('ok', False):
                return None

            f_val = float(strict_result.get('f_mm', f_seed_mm))
            rot_val = strict_result.get('rot_matrix')
            if rot_val is None:
                return None

            f_pixels_val = utils.get_effective_f_pixels(
                f_val,
                cam.data.sensor_width,
                cam.data.sensor_height,
                cam.data.sensor_fit,
                pixel_res_x,
                pixel_res_y,
            )
            if not np.isfinite(f_pixels_val) or f_pixels_val <= 1e-8:
                return None

            target_px, target_py = (0.0, 0.0)
            if anchor_screen_offset is not None:
                target_px, target_py = anchor_screen_offset

            ray_cam = np.array([target_px, target_py, -f_pixels_val])
            ray_cam /= np.linalg.norm(ray_cam)
            p_org_cam = ray_cam * current_dist
            loc_val = cursor_location - (rot_val @ mathutils.Vector(p_org_cam))

            focal_state = strict_result.get('focal_state', 'locked')
            if focal_state == 'refined':
                hint_key = hint_refined_key
            elif focal_state == 'consistent':
                # 焦距已由消失点确定、无需改变，没有需要提醒用户的事。
                hint_key = None
            elif focal_state == 'fixed' and hint_fixed_key is not None:
                hint_key = hint_fixed_key
            else:
                hint_key = hint_locked_key

            return {
                'f_mm': f_val,
                'rot_matrix': rot_val,
                'shift_x': 0.0,
                'shift_y': 0.0,
                'loc_orbit': loc_val,
                'hint_key': hint_key,
                'strict_result': strict_result,
            }
        except Exception as e:
            print(f"[CameraMatch] Strict solve failed: {e}")
            return None

    iface_ = bpy.app.translations.pgettext_iface

    solve_mode = perspective_constraints.get('mode', 'INSUFFICIENT')
    guided_lines_data = perspective_constraints.get('guided_lines_data', lines_data)

    # 焦距可辨识性：焦距由两个正交轴的消失点共同决定，而每个消失点至少要 2 条
    # 同轴线段。因此"能出消失点的轴"少于 2 个时，焦距不可辨识，必须完全不碰它。
    #
    # 这是与 utils.refine_focal_length_for_constrained_rotation 入口那道门槛配套的
    # 第二道保险：那里拦住"精化"，这里连"是否允许精化"都直接关掉，两条路径
    # （hybrid 后的 strict_post 与 fallback 的 strict_solution）都走同一个开关。
    #
    # 实测用户工程 I:\空净\IN19\4.blend（X 1 条 + Y 3 条 + Z 1 条）：残差随焦距
    # 单调下降没有谷底，插件把 50mm 一路推到 257mm 还在涨，姿态随之被拽歪。
    vp_capable_axes = [axis for axis in ('X', 'Y', 'Z') if len(lines_data.get(axis, [])) >= 2]
    focal_identifiable = len(vp_capable_axes) >= 2

    if solve_mode == 'ONE_POINT':
        strict_solution = solve_strict_mode(
            guided_lines_data,
            cam.data.lens,
            "Constrained solve: focal refined",
            "Constrained solve: focal not updated",
            allow_focal_refine=False,
            hint_fixed_key="One-point strict: focal fixed",
        )
        if strict_solution is None:
            return False, iface_("Solving failed. Check line placement.")

        f_mm = strict_solution['f_mm']
        rot_matrix = strict_solution['rot_matrix']
        shift_x = strict_solution['shift_x']
        shift_y = strict_solution['shift_y']
        loc_orbit = strict_solution['loc_orbit']
        solve_mode_hint_key = strict_solution['hint_key']

    else:
        vp_for_full_solve = dict(vp_data)
        # f_mm / rot_matrix 必须先初始化：消失点少于 2 个时（例如用户画的两条
        # 同轴线段彼此接近平行，消失点落在无穷远、compute_vanishing_point 返回
        # None）下面的 if 不会进入，而 297 行与 373 行的判断都要读 f_mm，会直接
        # NameError 逃出函数 —— 调用方 trigger_solve 吞掉异常，用户看到的是
        # "点了匹配相机毫无反应"。
        f_mm = None
        rot_matrix = None
        if len(vp_for_full_solve) >= 2:
            try:
                solve_axis_weights = {axis: axis_weights.get(axis, 0) for axis in vp_for_full_solve.keys()}
                f_mm, rot_matrix, shift_x, shift_y, loc_orbit = utils.calculate_camera_transform(
                    vp_for_full_solve,
                    cam.data.sensor_width,
                    cam.data.sensor_height,
                    cam.data.sensor_fit,
                    pixel_res_x, pixel_res_y, current_dist,
                    default_f_mm=cam.data.lens,
                    axis_weights=solve_axis_weights,
                    anchor_location=cursor_location,
                    anchor_screen_offset=anchor_screen_offset,
                    current_rot_matrix=cam.matrix_world.to_3x3(),
                )
            except Exception as e:
                print(f"[CameraMatch] Hybrid full solve failed: {e}")
                f_mm = None

        if f_mm is not None and rot_matrix is not None:
            # 姿态种子必须用"当前相机姿态"，不能用 hybrid（消失点法）的姿态。
            # solve_camera_rotation_constrained 是交替投影迭代，只做局部收敛；
            # 而两点透视下"相机轴 ⊥ 各轴平面法线"这个约束系统只剩一个自由度：
            # u⊥nX 给 2、v⊥nY 给 2、w 自由 3，减去 3 个正交约束与 3 个单位长度
            # 约束，恰好剩 1 —— 就是绕视轴的 roll。迭代本身对 roll 没有约束力，
            # 因为残差 |axis·n| 对绕视轴旋转不敏感（消失点只是沿各自直线滑动）。
            # 实测 ±8px 端点噪声下：用 hybrid 姿态作种子，roll 偏 -21.8~18.9°、
            # 整体旋转误差 15.8~77.7°；改用当前姿态作种子后 roll 误差降到 ±1.4°、
            # 整体旋转误差 0.47~1.88°。三点透视有唯一解，种子不影响结果。
            # 多起点：主起点仍是"当前相机姿态"（两点透视下绕视轴的 roll 自由度
            # 只能由种子保持，见上面的长注释），但额外把消失点法给出的姿态也作为
            # 起点 —— 交替投影只做局部收敛，单起点会卡在错误的局部极小上。实测
            # 用户的工程（4096x4096、X/Y/Z 各 3 条线）：只用当前姿态作起点时残差
            # 0.1441、相机被解成仰视 29.6 度（画面整个歪掉、Z 轴方向看起来不对）；
            # 加入其它起点后残差降到 0.0238、俯视 12.4 度（up_z=0.977），焦距曲线
            # 也从"单调下降、没有谷底"变成"明确谷底在 45mm"。
            strict_post = solve_strict_mode(
                lines_data,
                f_mm,
                "Constrained solve: focal refined",
                "Constrained solve: focal not updated",
                allow_focal_refine=focal_identifiable,
                rot_seed=None,
                smooth_ref_mm=smooth_ref_mm,
                extra_seeds=[rot_matrix],
            )
            if strict_post is not None:
                strict_result = strict_post.get('strict_result', {})
                strict_residual = float(strict_result.get('residual', float('inf')))

                hybrid_f_pixels = utils.get_effective_f_pixels(
                    f_mm,
                    cam.data.sensor_width,
                    cam.data.sensor_height,
                    cam.data.sensor_fit,
                    pixel_res_x,
                    pixel_res_y,
                )
                hybrid_residual = utils.compute_rotation_constraint_residual(lines_data, rot_matrix, hybrid_f_pixels)
                improvement = hybrid_residual - strict_residual

                # 用户手动拖拽地平线产生的偏移属于"强制俯仰覆盖"，必须优先于
                # 基于线条的 strict 解，否则松手后俯仰会被瞬间回退（strict 解
                # 以原始线条为尊，会把偏移带来的俯仰还原掉）。
                #
                # 必须同时要求 horizon_enabled：horizon_offset_px 是拖拽留下的
                # 残留值，取消勾选 Enable Horizon 并不会把它归零（只有
                # reset_horizon_manual_offset 会清）。只看偏移量的话，"关掉地平线
                # 但残留 40px 偏移"会让约束解一直被跳过，用户看到的是"关掉地平线
                # 后线条拖不动相机"。
                horizon_override_active = (
                    bool(scene.cmp_data.horizon_enabled)
                    and abs(scene.cmp_data.horizon_offset_px) > 1e-9
                )

                # 实时模式（fast）无条件采用 strict_post：它的焦距已经向"上一帧"
                # 做过平滑，而 hybrid 的焦距是当帧独立解出的、在手工线条的噪声下
                # 会逐帧乱跳（实测 ±8px 端点噪声能让最优焦距在 28~47mm 间摆动）。
                # 时间平滑的本意就是"宁可当前帧残差略差，也要保持稳定"，所以这
                # 里不能再拿残差去卡它，否则平滑会被整体绕过。
                # 约束法（solve_camera_rotation_constrained）的姿态远比消失点法稳健：
                # 实测俯视 65°、每轴 3 条线、±8px 端点噪声下，消失点法给出的整体旋转
                # 误差高达 15.8~77.9°，而约束法只有 0.47~2.47°；即使传给约束法的焦距
                # 完全错误（13.71mm / 109.99mm），误差也只到 2.47°——因为平面法线的
                # 方向主要由直线系数 (a, b) 决定，对 f_pixels 不敏感。
                # 残差判据分不清"几何正确"和"过拟合噪声"：消失点法直接拟合消失点，
                # 残差天然更小，于是稳健解被丢掉，roll 在两点透视下能偏 20°。
                # 因此只要约束法成功（ok），就优先采用它的姿态。
                improved_enough = (
                    not horizon_override_active
                    and np.isfinite(hybrid_residual)
                    and np.isfinite(strict_residual)
                    and (
                        fast
                        or strict_result.get('ok', False)
                        or strict_residual <= hybrid_residual * 0.985
                        or improvement >= 0.0015
                        or (
                            strict_result.get('focal_state') == 'refined'
                            and strict_residual <= hybrid_residual + 0.001
                        )
                    )
                )

                if improved_enough:
                    f_mm = strict_post['f_mm']
                    rot_matrix = strict_post['rot_matrix']
                    shift_x = strict_post['shift_x']
                    shift_y = strict_post['shift_y']
                    loc_orbit = strict_post['loc_orbit']
                    solve_mode_hint_key = strict_post['hint_key']

        if f_mm is None or rot_matrix is None:
            strict_solution = solve_strict_mode(
                lines_data,
                cam.data.lens,
                "Constrained solve: focal refined",
                "Constrained solve: focal not updated",
                allow_focal_refine=focal_identifiable,
                smooth_ref_mm=smooth_ref_mm,
            )
            if strict_solution is None:
                return False, iface_("Solving failed. Check line placement.")

            f_mm = strict_solution['f_mm']
            rot_matrix = strict_solution['rot_matrix']
            shift_x = strict_solution['shift_x']
            shift_y = strict_solution['shift_y']
            loc_orbit = strict_solution['loc_orbit']
            solve_mode_hint_key = strict_solution['hint_key']

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

    if abs(shift_x) < 1e-9 and abs(shift_y) < 1e-9:
        shift_x = current_shift_x
        shift_y = current_shift_y

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
    view_state = utils.capture_camera_view_state(context)
    try:
        properties.suppress_horizon_updates()

        # 实时模式对焦距做"低通 + 单帧限幅"。拖拽中线条端点每帧都在变，消失点随之
        # 抖动，而"哪个焦距最优"在端点噪声下本身就是随机的 —— 实测 ±8px 噪声、20
        # 帧连续解算（1920x1080、真值 35mm、初值 50mm、XYZ 各 3 条），判据放行时给
        # 出的候选焦距在 13.2~66.1mm 之间乱跳，纯低通（alpha=0.35）只把 spread 从
        # 35.4mm 压到 24.9mm。根治要靠"单帧不允许大幅跳变"：限幅 6%/帧时从 50mm
        # 走到 35mm 需要 6 帧（28Hz 下 0.21 秒），用户感知不到延迟，但单帧的噪声
        # 跳变会被削平。
        #
        # 只对 fast 生效：exact 是用户点按钮/松手时的最终解算，必须立刻到位。
        if fast and f_mm is not None and np.isfinite(f_mm):
            prev_lens = float(cam.data.lens)
            if np.isfinite(prev_lens) and 1.0 < prev_lens < 10000.0:
                target_f = prev_lens + 0.5 * (float(f_mm) - prev_lens)
                delta_f = target_f - prev_lens
                max_step = max(0.4, prev_lens * 0.06)
                if abs(delta_f) <= max_step:
                    f_mm = target_f
                else:
                    f_mm = prev_lens + (max_step if delta_f > 0 else -max_step)

        # 尺度补偿（希区柯克式 dolly zoom）：解算出的焦距与解算前不同时，沿光轴
        # 移动相机，使"游标深度平面"上的物体在画面中大小不变（保持 f / D 不变）。
        #
        # 为什么需要：位置是沿"游标射线"放置的（射线方向由 f_pixels 决定），只改
        # 焦距而不改距离，画面里的模型就会随焦距一起缩放 —— 用户看到的是"物体
        # 飞来飞去、突然变大变小"。补偿之后焦距变化只改变透视（消失点位置），
        # 不再改变模型大小，视觉上等价于希区柯克变焦。
        #
        # 严格保持大小的只有"游标所在深度"这一个平面（这正是 dolly zoom 的定义），
        # 其他深度的物体仍有正常的透视缩放。消失点只由姿态和焦距决定，所以这个
        # 补偿不会影响透视匹配本身。
        #
        # f_before_apply 必须在写入新焦距之前取；current_dist 是解算前相机到游标
        # 的距离。
        f_before_apply = float(cam.data.lens)
        if (f_mm is not None and np.isfinite(f_mm)
                and np.isfinite(f_before_apply) and f_before_apply > 1e-6
                and abs(float(f_mm) - f_before_apply) > 1e-9
                and rot_matrix is not None
                and anchor_screen_offset is not None):
            try:
                f_px_comp = utils.get_effective_f_pixels(
                    float(f_mm),
                    cam.data.sensor_width,
                    cam.data.sensor_height,
                    cam.data.sensor_fit,
                    pixel_res_x,
                    pixel_res_y,
                )
                if np.isfinite(f_px_comp) and f_px_comp > 1e-8:
                    ray_comp = np.array(
                        [anchor_screen_offset[0], anchor_screen_offset[1], -f_px_comp],
                        dtype=float,
                    )
                    ray_norm_comp = float(np.linalg.norm(ray_comp))
                    if np.isfinite(ray_norm_comp) and ray_norm_comp > 1e-9:
                        ray_comp /= ray_norm_comp
                        scale_comp = float(f_mm) / f_before_apply
                        # 极端初值保护：初值 1mm 解到 35mm 时比例是 35，补偿后
                        # 相机会退到 495 单位外（默认 clip_end 只有 100，模型会
                        # 被裁掉、看起来"消失"）。比例超出这个区间时不补偿，
                        # 保持当前距离 —— 模型会缩放，但至少还在画面里。
                        if 0.05 <= scale_comp <= 20.0:
                            dist_comp = current_dist * scale_comp
                            if np.isfinite(dist_comp) and dist_comp > 1e-6:
                                rot3_comp = rot_matrix.to_3x3() if hasattr(rot_matrix, 'to_3x3') else rot_matrix
                                loc_orbit = cursor_location - (rot3_comp @ mathutils.Vector(ray_comp * dist_comp))
            except Exception as e:
                print(f"[CameraMatch] Scale compensation failed: {e}")

        cam.data.lens = f_mm
        cam.data.shift_x = shift_x
        cam.data.shift_y = shift_y

        new_rot_4x4 = rot_matrix.to_4x4()

        cam.matrix_world = mathutils.Matrix.Translation(loc_orbit) @ new_rot_4x4
        context.view_layer.update()

        target_uv_for_shift = target_cursor_uv
        if horizon_lock_active and horizon_target_uv is not None:
            target_uv_for_shift = horizon_target_uv

        # 游标对齐（有限差分微调 shift 把游标钉到目标 UV）失败时必须让用户知道：
        # 它返回 False 表示迭代没有收敛到 tolerance_px 以内，画面看上去"成功"了，
        # 但游标并不在目标位置上。以前这里直接丢弃返回值，用户只能靠肉眼判断。
        cursor_lock_failed = False
        if target_uv_for_shift is not None:
            try:
                _lock_ok = utils.compensate_shift_for_target_uv(
                    scene,
                    cam,
                    cursor_location,
                    target_uv_for_shift,
                    update_view_layer=context.view_layer.update,
                    iterations=2,
                )
                if _lock_ok is False:
                    cursor_lock_failed = True

                if np.isfinite(cam.data.shift_x) and np.isfinite(cam.data.shift_y):
                    shift_x = float(cam.data.shift_x)
                    shift_y = float(cam.data.shift_y)

            except Exception as e:
                print(f"[CameraMatch] Cursor lock correction failed: {e}")
                cursor_lock_failed = True

        scene.cmp_data.last_world_rotation = 0.0
        scene.cmp_data.last_flip_z = False
        scene.cmp_data.world_rotation = 0.0
        scene.cmp_data.flip_z_axis = False

        if current_world_rotation != 0.0:
            scene.cmp_data.world_rotation = current_world_rotation
        if current_flip_z:
            scene.cmp_data.flip_z_axis = current_flip_z

        iface_ = bpy.app.translations.pgettext_iface
        mode_hint = ""
        # 同一条提示既拼进状态栏用的 mode_hint，也单独存进 warning_parts。
        # 面板要用它逐行显示 —— Blender 的 label 不换行，拼成一整条会被截断。
        warning_parts = []
        if solve_mode_hint_key:
            mode_hint = " " + iface_(solve_mode_hint_key)
            warning_parts.append(iface_(solve_mode_hint_key))

        if cursor_lock_failed:
            mode_hint += " " + iface_("Cursor lock did not converge")
            warning_parts.append(iface_("Cursor lock did not converge"))

        # 纯输入侧的三项检查（某轴线段互相平行 / 焦距不可辨识 / 线条不是一组
        # 空间平行线）走 collect_input_warnings —— 绘制过程中的轻量刷新
        # （CameraMatchTool.refresh_line_check）复用同一实现，避免两处阈值和
        # 文案漂移。判据细节与实测依据见该函数的 docstring。
        for _input_warning in collect_input_warnings(
                lines_data, pixel_res_x, pixel_res_y, lines,
                principal_u=principal_u, principal_v=principal_v,
                vp_data_raw=vp_data_raw, solve_mode=solve_mode):
            mode_hint += " " + _input_warning
            warning_parts.append(_input_warning)

        msg = iface_("Success: ") + f"f={f_mm:.1f}mm," + iface_(" Shift=") + f"({shift_x:.2f}, {shift_y:.2f})" + mode_hint
        # 把"能解算、但输入可疑"的提示持久化下来，供 N 面板与绘制模式状态栏
        # 常驻显示 —— self.report 只在按钮触发时闪一下，实时绘制时更看不到。
        cmp_data.last_solve_warning = "\n".join(warning_parts)
        cmp_data.last_solve_checked = True

    except Exception as e:
        print(f"[CameraMatch] Apply transform failed: {e}")
        iface_ = bpy.app.translations.pgettext_iface
        return False, iface_("Failed to apply camera transform")
    finally:
        properties.resume_horizon_updates()
        utils.restore_camera_view_state(view_state)

    # 关于 shift 自洽性：解算用的主点由**解算前**的 shift 决定，而上面的补偿
    # 会改写 shift。曾经在这里用新 shift 递归重解一轮，但那会让整条流程
    # （含位置锚点、游标屏幕偏移、符号消歧种子）在"已被上一次解算改过的相机"
    # 上重跑一遍，实测会与手动微调互相干扰、姿态来回震荡，因此移除。
    # 实际工作流不受影响：线段是"贴死在焦平面上"的，用户总是在当时的 shift
    # 下画的线，所以输入天然自洽（实测线段与 shift 一致时焦距误差为 0%）。
    return True, msg


class CMP_OT_MatchCamera(bpy.types.Operator):
    """Solve camera based on drawn lines"""
    bl_idname = "cmp.match_camera"
    bl_label = "Match Camera"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        success, msg = solve_camera_core(context)
        if success:
            context.view_layer.update()
            self.report({'INFO'}, msg)
            print(f"[CameraMatchPro] {msg}")
            return {'FINISHED'}
        else:
            self.report({'WARNING'}, msg)
            return {'CANCELLED'}


class CMP_OT_LineCheckRow(bpy.types.Operator):
    """Line check result (display only)"""
    bl_idname = "cmp.line_check_row"
    bl_label = "Line Check"
    bl_description = "Line check result (display only, nothing to click)"
    bl_options = {'INTERNAL'}

    def execute(self, context):
        # 这是一个"只用来显示"的按钮。N 面板的线条检查结果需要红色高亮，而
        # Blender 的 alert 对纯 label 不生效 —— 实测同一个 alert 容器里，
        # label 的文字始终是普通灰 (75,84,99)，而 button / prop 会拿到红底
        # (237,138,138) 白字。所以注册一个空操作符当画布，靠 alert 换红底。
        return {'FINISHED'}


class CMP_OT_MatchBackgroundResolution(bpy.types.Operator):
    """Match render resolution to the camera background image resolution"""
    bl_idname = "cmp.match_background_resolution"
    bl_label = "Match Background Resolution"
    bl_options = {'REGISTER', 'UNDO'}

    def _notify_no_background(self, context, msg):
        iface_ = bpy.app.translations.pgettext_iface
        msg = iface_(msg)
        self.report({'WARNING'}, msg)
        print(f"[CameraMatch] {msg}")

        def draw(menu, _ctx):
            menu.layout.label(text=msg)

        # 仅在有窗口的 GUI 会话中弹窗；后台/无窗口模式调用 popup_menu 会段错误
        if not bpy.app.background:
            try:
                wm = getattr(context, "window_manager", None) or bpy.context.window_manager
                if wm is not None:
                    wm.popup_menu(draw, title="Simple Camera Match", icon='INFO')
            except Exception:
                pass

    def execute(self, context):
        scene = getattr(context, "scene", None)
        if scene is None:
            self.report({'ERROR'}, "No scene")
            return {'CANCELLED'}

        cam = scene.camera
        if cam is None or getattr(cam, "data", None) is None:
            self._notify_no_background(context, "No camera or no background image on active camera")
            return {'CANCELLED'}

        bg_image = None
        for item in cam.data.background_images:
            if item.image is not None:
                bg_image = item.image
                break

        if bg_image is None:
            self._notify_no_background(context, "No background image on active camera")
            return {'CANCELLED'}

        w, h = int(bg_image.size[0]), int(bg_image.size[1])
        if w <= 0 or h <= 0:
            self._notify_no_background(context, "Background image has no valid size")
            return {'CANCELLED'}

        render = scene.render
        render.resolution_x = w
        render.resolution_y = h
        render.resolution_percentage = 100

        iface_ = bpy.app.translations.pgettext_iface
        msg = iface_("Render resolution set to ") + "%dx%d" % (w, h)
        self.report({'INFO'}, msg)
        print(f"[CameraMatch] {msg}")
        return {'FINISHED'}


def register():
    utils.register_class_safe(CMP_OT_MatchCamera)
    utils.register_class_safe(CMP_OT_LineCheckRow)
    utils.register_class_safe(CMP_OT_MatchBackgroundResolution)


def unregister():
    utils.unregister_class_safe(CMP_OT_MatchBackgroundResolution)
    utils.unregister_class_safe(CMP_OT_LineCheckRow)
    utils.unregister_class_safe(CMP_OT_MatchCamera)
