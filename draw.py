import math
import os

import blf
import bpy
import gpu
from gpu.types import GPUShaderCreateInfo
from gpu_extras.batch import batch_for_shader

try:
    from mathutils.geometry import tessellate_polygon
except Exception:
    tessellate_polygon = None

from .constants import THEME as T


_SHADER = None
_IMAGE_SHADER = None
_SDF_SHADER = None
_COLOR_SHADER = None
_PAPER_SHADER = None
_GRADIENT_SHADER = None
_VIEWPORT = (1.0, 1.0)
_SDF_BATCH_CACHE = {}
_ASSET_TEXTURES = {}
_ART_FONT_ID = None
_ART_FONT_TRIED = False
_EMPTY_COLOR = (0.0, 0.0, 0.0, 0.0)


def _shader():
    global _SHADER
    if _SHADER is None:
        _SHADER = gpu.shader.from_builtin("UNIFORM_COLOR")
    return _SHADER


def _image_shader():
    global _IMAGE_SHADER
    if _IMAGE_SHADER is not None:
        return _IMAGE_SHADER
    for name in ("IMAGE_SCENE_LINEAR_TO_REC709_SRGB", "IMAGE", "2D_IMAGE"):
        try:
            _IMAGE_SHADER = gpu.shader.from_builtin(name)
            return _IMAGE_SHADER
        except Exception:
            continue
    return None


def set_viewport(width, height):
    global _VIEWPORT
    viewport = (max(1.0, float(width)), max(1.0, float(height)))
    if viewport != _VIEWPORT:
        _VIEWPORT = viewport
        _SDF_BATCH_CACHE.clear()


def _sdf_shader():
    global _SDF_SHADER
    if _SDF_SHADER is not None:
        return _SDF_SHADER

    vertex = """
    void main()
    {
        vec2 ndc = vec2(
            pos.x / uViewport.x * 2.0 - 1.0,
            pos.y / uViewport.y * 2.0 - 1.0
        );
        gl_Position = vec4(ndc, 0.0, 1.0);
    }
    """
    fragment = """
    float sdRoundBox(vec2 p, vec2 b, float r)
    {
        vec2 q = abs(p) - b + vec2(r);
        return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
    }

    float sdCircle(vec2 p, float r)
    {
        return length(p) - r;
    }

    void main()
    {
        vec2 p = gl_FragCoord.xy - uCenter;
        float d;
        if (uType == 2) {
            d = sdCircle(p, uHalfSize.x);
        } else {
            d = sdRoundBox(p, uHalfSize, uRadius);
        }
        float aa = max(0.72, fwidth(d) * 1.25);
        float bw = max(0.0, uBorderW);
        bool hasBorder = uBorder.a > 0.0 && bw > 0.0;

        if (hasBorder) {
            float outerAlpha = 1.0 - smoothstep(-aa, aa, d);
            float innerAlpha = 1.0 - smoothstep(-aa, aa, d + bw);
            float borderAlpha = max(0.0, outerAlpha - innerAlpha);
            float fillA = uFill.a * innerAlpha;
            float borderA = uBorder.a * borderAlpha;
            float outA = max(fillA, borderA);
            vec3 outRgb = (uFill.rgb * fillA + uBorder.rgb * borderA) / max(outA, 0.0001);
            fragColor = vec4(outRgb, outA);
        } else {
            float alpha = 1.0 - smoothstep(-aa, aa, d);
            fragColor = vec4(uFill.rgb, uFill.a * alpha);
        }
    }
    """

    info = GPUShaderCreateInfo()
    info.vertex_in(0, "VEC2", "pos")
    info.push_constant("VEC2", "uViewport")
    info.push_constant("VEC2", "uCenter")
    info.push_constant("VEC2", "uHalfSize")
    info.push_constant("FLOAT", "uRadius")
    info.push_constant("VEC4", "uFill")
    info.push_constant("VEC4", "uBorder")
    info.push_constant("FLOAT", "uBorderW")
    info.push_constant("INT", "uType")
    info.vertex_source(vertex)
    info.fragment_source(fragment)
    info.fragment_out(0, "VEC4", "fragColor")
    _SDF_SHADER = gpu.shader.create_from_info(info)
    return _SDF_SHADER


def _color_shader():
    global _COLOR_SHADER
    if _COLOR_SHADER is not None:
        return _COLOR_SHADER

    vertex = """
    void main()
    {
        vec2 ndc = vec2(
            pos.x / uViewport.x * 2.0 - 1.0,
            pos.y / uViewport.y * 2.0 - 1.0
        );
        vColor = color;
        gl_Position = vec4(ndc, 0.0, 1.0);
    }
    """
    fragment = """
    void main()
    {
        fragColor = vColor;
    }
    """

    info = GPUShaderCreateInfo()
    info.vertex_in(0, "VEC2", "pos")
    info.vertex_in(1, "VEC4", "color")
    info.push_constant("VEC2", "uViewport")
    iface = gpu.types.GPUStageInterfaceInfo("ra_color_iface")
    iface.smooth("VEC4", "vColor")
    info.vertex_out(iface)
    info.vertex_source(vertex)
    info.fragment_source(fragment)
    info.fragment_out(0, "VEC4", "fragColor")
    _COLOR_SHADER = gpu.shader.create_from_info(info)
    return _COLOR_SHADER


def _paper_shader():
    global _PAPER_SHADER
    if _PAPER_SHADER is not None:
        return _PAPER_SHADER

    vertex = """
    void main()
    {
        vec2 ndc = vec2(
            pos.x / uViewport.x * 2.0 - 1.0,
            pos.y / uViewport.y * 2.0 - 1.0
        );
        gl_Position = vec4(ndc, 0.0, 1.0);
    }
    """
    fragment = """
    float sdBox(vec2 p, vec2 b)
    {
        vec2 q = abs(p) - b;
        return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0);
    }

    float sideY(int index)
    {
        if (index < 4) {
            return uSideY0[index];
        }
        if (index < 8) {
            return uSideY1[index - 4];
        }
        if (index < 12) {
            return uSideY2[index - 8];
        }
        if (index < 16) {
            return uSideY3[index - 12];
        }
        if (index < 20) {
            return uSideY4[index - 16];
        }
        return uSideY5[index - 20];
    }

    float sdPaper(vec2 point)
    {
        vec2 halfSize = uRect.zw * 0.5;
        vec2 center = uRect.xy + halfSize;
        float d = sdBox(point - center, halfSize);

        float sideRadius = uSideData.x;
        for (int i = 0; i < 24; i++) {
            if (i >= uSideCount) {
                break;
            }
            float cy = sideY(i);
            float leftCut = length(point - vec2(uRect.x, cy)) - sideRadius;
            float rightCut = length(point - vec2(uRect.x + uRect.z, cy)) - sideRadius;
            d = max(d, -leftCut);
            d = max(d, -rightCut);
        }

        if (uBottomCount > 0 && uBottomData.z > 0.0001) {
            float nearest = floor((point.x - uBottomData.y) / uBottomData.z + 0.5);
            nearest = clamp(nearest, 0.0, float(uBottomCount - 1));
            float cx = uBottomData.y + nearest * uBottomData.z;
            float bottomCut = length(point - vec2(cx, uRect.y)) - uBottomData.x;
            d = max(d, -bottomCut);
        }
        return d;
    }

    void main()
    {
        float d = sdPaper(gl_FragCoord.xy);
        float aa = max(0.72, fwidth(d) * 1.25);
        float fillA = uFill.a * (1.0 - smoothstep(-aa, aa, d));
        float borderA = 0.0;

        if (uBorder.a > 0.0 && uBorderW > 0.0) {
            borderA = uBorder.a * (1.0 - smoothstep(uBorderW * 0.5 - aa, uBorderW * 0.5 + aa, abs(d)));
        }

        float fillPart = fillA * (1.0 - borderA);
        float outA = fillPart + borderA;
        vec3 outRgb = (uFill.rgb * fillPart + uBorder.rgb * borderA) / max(outA, 0.0001);
        fragColor = vec4(outRgb, outA);
    }
    """

    info = GPUShaderCreateInfo()
    info.vertex_in(0, "VEC2", "pos")
    info.push_constant("VEC2", "uViewport")
    info.push_constant("VEC4", "uRect")
    info.push_constant("VEC4", "uFill")
    info.push_constant("VEC4", "uBorder")
    info.push_constant("FLOAT", "uBorderW")
    info.push_constant("VEC4", "uSideData")
    info.push_constant("VEC4", "uSideY0")
    info.push_constant("VEC4", "uSideY1")
    info.push_constant("VEC4", "uSideY2")
    info.push_constant("VEC4", "uSideY3")
    info.push_constant("VEC4", "uSideY4")
    info.push_constant("VEC4", "uSideY5")
    info.push_constant("INT", "uSideCount")
    info.push_constant("VEC4", "uBottomData")
    info.push_constant("INT", "uBottomCount")
    info.vertex_source(vertex)
    info.fragment_source(fragment)
    info.fragment_out(0, "VEC4", "fragColor")
    _PAPER_SHADER = gpu.shader.create_from_info(info)
    return _PAPER_SHADER


def _gradient_shader():
    global _GRADIENT_SHADER
    if _GRADIENT_SHADER is not None:
        return _GRADIENT_SHADER

    vertex = """
    void main()
    {
        vec2 ndc = vec2(
            pos.x / uViewport.x * 2.0 - 1.0,
            pos.y / uViewport.y * 2.0 - 1.0
        );
        gl_Position = vec4(ndc, 0.0, 1.0);
    }
    """
    fragment = """
    float sdRoundBox(vec2 p, vec2 b, float r)
    {
        vec2 q = abs(p) - b + vec2(r);
        return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
    }

    vec4 sampleGradient(float t)
    {
        if (t <= uStops.y) {
            float span = max(0.0001, uStops.y - uStops.x);
            return mix(uColor0, uColor1, clamp((t - uStops.x) / span, 0.0, 1.0));
        }
        if (t <= uStops.z) {
            float span = max(0.0001, uStops.z - uStops.y);
            return mix(uColor1, uColor2, clamp((t - uStops.y) / span, 0.0, 1.0));
        }
        float span = max(0.0001, uStops.w - uStops.z);
        return mix(uColor2, uColor3, clamp((t - uStops.z) / span, 0.0, 1.0));
    }

    void main()
    {
        vec2 halfSize = uRect.zw * 0.5;
        vec2 center = uRect.xy + halfSize;
        float d = sdRoundBox(gl_FragCoord.xy - center, halfSize, uRadius);
        float aa = max(0.72, fwidth(d) * 1.25);
        float shapeA = 1.0 - smoothstep(-aa, aa, d);
        float t = clamp((gl_FragCoord.y - uRect.y) / max(1.0, uRect.w), 0.0, 1.0);
        vec4 fill = sampleGradient(t);

        float borderA = 0.0;
        if (uBorder.a > 0.0 && uBorderW > 0.0) {
            borderA = uBorder.a * (1.0 - smoothstep(uBorderW * 0.5 - aa, uBorderW * 0.5 + aa, abs(d)));
        }

        float fillA = fill.a * shapeA * (1.0 - borderA);
        float outA = fillA + borderA;
        vec3 outRgb = (fill.rgb * fillA + uBorder.rgb * borderA) / max(outA, 0.0001);
        fragColor = vec4(outRgb, outA);
    }
    """

    info = GPUShaderCreateInfo()
    info.vertex_in(0, "VEC2", "pos")
    info.push_constant("VEC2", "uViewport")
    info.push_constant("VEC4", "uRect")
    info.push_constant("FLOAT", "uRadius")
    info.push_constant("VEC4", "uStops")
    info.push_constant("VEC4", "uColor0")
    info.push_constant("VEC4", "uColor1")
    info.push_constant("VEC4", "uColor2")
    info.push_constant("VEC4", "uColor3")
    info.push_constant("VEC4", "uBorder")
    info.push_constant("FLOAT", "uBorderW")
    info.vertex_source(vertex)
    info.fragment_source(fragment)
    info.fragment_out(0, "VEC4", "fragColor")
    _GRADIENT_SHADER = gpu.shader.create_from_info(info)
    return _GRADIENT_SHADER


def _sdf_rect_batch(x, y, w, h, pad=2.0):
    bx, by, bw, bh = x - pad, y - pad, w + pad * 2.0, h + pad * 2.0
    key = (round(bx, 2), round(by, 2), round(bw, 2), round(bh, 2))
    batch = _SDF_BATCH_CACHE.get(key)
    if batch:
        return batch
    if len(_SDF_BATCH_CACHE) > 512:
        _SDF_BATCH_CACHE.clear()
    points = (
        (bx, by),
        (bx + bw, by),
        (bx + bw, by + bh),
        (bx, by + bh),
    )
    batch = batch_for_shader(_sdf_shader(), "TRI_FAN", {"pos": points})
    _SDF_BATCH_CACHE[key] = batch
    return batch


def sdf_rounded_rect(x, y, w, h, radius, fill, border=None, border_width=0.0):
    fill = fill or _EMPTY_COLOR
    if fill[3] <= 0 and not border:
        return
    shader = _sdf_shader()
    batch = _sdf_rect_batch(x, y, w, h)
    gpu.state.blend_set("ALPHA")
    shader.bind()
    shader.uniform_float("uViewport", _VIEWPORT)
    shader.uniform_float("uCenter", (x + w * 0.5, y + h * 0.5))
    shader.uniform_float("uHalfSize", (w * 0.5, h * 0.5))
    shader.uniform_float("uRadius", min(radius, w * 0.5, h * 0.5))
    shader.uniform_float("uFill", fill)
    shader.uniform_float("uBorder", border if border else _EMPTY_COLOR)
    shader.uniform_float("uBorderW", border_width if border else 0.0)
    shader.uniform_int("uType", 1)
    batch.draw(shader)
    gpu.state.blend_set("NONE")


def sdf_circle(cx, cy, radius, fill, border=None, border_width=0.0):
    fill = fill or _EMPTY_COLOR
    if fill[3] <= 0 and not border:
        return
    size = radius * 2.0
    shader = _sdf_shader()
    batch = _sdf_rect_batch(cx - radius, cy - radius, size, size)
    gpu.state.blend_set("ALPHA")
    shader.bind()
    shader.uniform_float("uViewport", _VIEWPORT)
    shader.uniform_float("uCenter", (cx, cy))
    shader.uniform_float("uHalfSize", (radius, radius))
    shader.uniform_float("uRadius", 0.0)
    shader.uniform_float("uFill", fill)
    shader.uniform_float("uBorder", border if border else _EMPTY_COLOR)
    shader.uniform_float("uBorderW", border_width if border else 0.0)
    shader.uniform_int("uType", 2)
    batch.draw(shader)
    gpu.state.blend_set("NONE")


def _art_font():
    global _ART_FONT_ID, _ART_FONT_TRIED
    if _ART_FONT_TRIED:
        return _ART_FONT_ID or 0
    _ART_FONT_TRIED = True
    for path in (
        "C:/Windows/Fonts/STXINGKA.TTF",
        "C:/Windows/Fonts/STXINWEI.TTF",
        "C:/Windows/Fonts/SIMLI.TTF",
        "C:/Windows/Fonts/SIMKAI.TTF",
    ):
        if not os.path.exists(path):
            continue
        try:
            _ART_FONT_ID = blf.load(path)
            break
        except Exception:
            continue
    return _ART_FONT_ID or 0


def _batch(kind, points, color):
    shader = _shader()
    gpu.state.blend_set("ALPHA")
    batch = batch_for_shader(shader, kind, {"pos": points})
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)
    gpu.state.blend_set("NONE")


def rect(x, y, w, h, color):
    _batch("TRI_FAN", ((x, y), (x + w, y), (x + w, y + h), (x, y + h)), color)


def poly(points, color):
    if len(points) < 3:
        return
    _batch("TRI_FAN", points, color)


def tessellated(contours, color):
    if not contours:
        return
    if tessellate_polygon is None:
        poly(contours[0], color)
        return
    vertices = [point for contour in contours for point in contour]
    triangles = tessellate_polygon(contours)
    if not triangles:
        poly(contours[0], color)
        return
    shader = _shader()
    gpu.state.blend_set("ALPHA")
    batch = batch_for_shader(shader, "TRIS", {"pos": vertices}, indices=triangles)
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)
    gpu.state.blend_set("NONE")


def _polygon_area(points):
    area = 0.0
    count = len(points)
    for index, point in enumerate(points):
        next_point = points[(index + 1) % count]
        area += point[0] * next_point[1] - next_point[0] * point[1]
    return area * 0.5


def _edge_normal(p0, p1, clockwise=False):
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    length = math.hypot(dx, dy)
    if length <= 0.0001:
        return 0.0, 0.0
    if clockwise:
        return -dy / length, dx / length
    return dy / length, -dx / length


def _with_alpha(color, alpha):
    return (color[0], color[1], color[2], color[3] * alpha)


def _colored_triangles(points, colors):
    if len(points) < 3:
        return
    shader = _color_shader()
    gpu.state.blend_set("ALPHA")
    batch = batch_for_shader(shader, "TRIS", {"pos": points, "color": colors})
    shader.bind()
    shader.uniform_float("uViewport", _VIEWPORT)
    batch.draw(shader)
    gpu.state.blend_set("NONE")


def _aa_edge_fringe(points, color, width=1.25):
    if len(points) < 3 or color[3] <= 0:
        return
    clockwise = _polygon_area(points) < 0
    verts = []
    colors = []
    solid = _with_alpha(color, 0.64)
    transparent = _with_alpha(color, 0.0)
    count = len(points)
    for index, p0 in enumerate(points):
        p1 = points[(index + 1) % count]
        nx, ny = _edge_normal(p0, p1, clockwise)
        if nx == 0.0 and ny == 0.0:
            continue
        q0 = (p0[0] + nx * width, p0[1] + ny * width)
        q1 = (p1[0] + nx * width, p1[1] + ny * width)
        verts.extend((p0, p1, q1, p0, q1, q0))
        colors.extend((solid, solid, transparent, solid, transparent, transparent))
    _colored_triangles(verts, colors)


def _aa_outline(points, color, width=1.0, fringe=1.15):
    if len(points) < 3 or color[3] <= 0 or width <= 0:
        return
    clockwise = _polygon_area(points) < 0
    verts = []
    colors = []
    inner_color = _with_alpha(color, 0.0)
    line_color = color
    outer_color = _with_alpha(color, 0.0)
    inner_w = max(0.35, width * 0.48)
    outer_w = max(0.35, width * 0.52)
    count = len(points)
    for index, p0 in enumerate(points):
        p1 = points[(index + 1) % count]
        nx, ny = _edge_normal(p0, p1, clockwise)
        if nx == 0.0 and ny == 0.0:
            continue
        inner0 = (p0[0] - nx * (inner_w + fringe), p0[1] - ny * (inner_w + fringe))
        inner1 = (p1[0] - nx * (inner_w + fringe), p1[1] - ny * (inner_w + fringe))
        edge_in0 = (p0[0] - nx * inner_w, p0[1] - ny * inner_w)
        edge_in1 = (p1[0] - nx * inner_w, p1[1] - ny * inner_w)
        edge_out0 = (p0[0] + nx * outer_w, p0[1] + ny * outer_w)
        edge_out1 = (p1[0] + nx * outer_w, p1[1] + ny * outer_w)
        outer0 = (p0[0] + nx * (outer_w + fringe), p0[1] + ny * (outer_w + fringe))
        outer1 = (p1[0] + nx * (outer_w + fringe), p1[1] + ny * (outer_w + fringe))

        verts.extend((inner0, inner1, edge_in1, inner0, edge_in1, edge_in0))
        colors.extend((inner_color, inner_color, line_color, inner_color, line_color, line_color))
        verts.extend((edge_in0, edge_in1, edge_out1, edge_in0, edge_out1, edge_out0))
        colors.extend((line_color, line_color, line_color, line_color, line_color, line_color))
        verts.extend((edge_out0, edge_out1, outer1, edge_out0, outer1, outer0))
        colors.extend((line_color, line_color, outer_color, line_color, outer_color, outer_color))
    _colored_triangles(verts, colors)


def rounded_rect(x, y, w, h, r, fill, border=None, border_width=1.0):
    sdf_rounded_rect(x, y, w, h, r, fill, border, border_width)


def _mix_color(a, b, t):
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(4))


def _gradient_color(stops, t):
    stops = sorted(stops, key=lambda item: item[0])
    if t <= stops[0][0]:
        return stops[0][1]
    for index in range(1, len(stops)):
        left_t, left_color = stops[index - 1]
        right_t, right_color = stops[index]
        if t <= right_t:
            span = max(0.0001, right_t - left_t)
            return _mix_color(left_color, right_color, (t - left_t) / span)
    return stops[-1][1]


def rounded_rect_gradient_y(x, y, w, h, r, stops, steps=18, border=None, border_width=1.0):
    if w <= 0 or h <= 0 or not stops:
        return
    stops = sorted(stops, key=lambda item: item[0])
    if len(stops) >= 3:
        selected = (stops[0], stops[1], stops[2], stops[-1])
        shader = _gradient_shader()
        batch = _sdf_rect_batch(x, y, w, h)
        gpu.state.blend_set("ALPHA")
        shader.bind()
        shader.uniform_float("uViewport", _VIEWPORT)
        shader.uniform_float("uRect", (x, y, w, h))
        shader.uniform_float("uRadius", min(r, w * 0.5, h * 0.5))
        shader.uniform_float("uStops", tuple(item[0] for item in selected))
        shader.uniform_float("uColor0", selected[0][1])
        shader.uniform_float("uColor1", selected[1][1])
        shader.uniform_float("uColor2", selected[2][1])
        shader.uniform_float("uColor3", selected[3][1])
        shader.uniform_float("uBorder", border if border else _EMPTY_COLOR)
        shader.uniform_float("uBorderW", border_width if border else 0.0)
        batch.draw(shader)
        gpu.state.blend_set("NONE")
        return
    steps = max(2, min(48, int(steps)))
    try:
        gpu.state.scissor_test_set(True)
        for index in range(steps):
            t0 = index / steps
            t1 = (index + 1) / steps
            band_y = y + h * t0
            band_h = max(1.0, h * (t1 - t0))
            gpu.state.scissor_set(
                int(round(x - 2)),
                int(round(band_y)),
                int(round(w + 4)),
                int(round(band_h + 1)),
            )
            sdf_rounded_rect(x, y, w, h, r, _gradient_color(stops, (t0 + t1) * 0.5), None, 0.0)
    finally:
        gpu.state.scissor_test_set(False)
    if border:
        sdf_rounded_rect(x, y, w, h, r, _EMPTY_COLOR, border, border_width)


def line(points, color, width=1.0):
    if len(points) < 2:
        return
    gpu.state.line_width_set(width)
    _batch("LINE_STRIP", points, color)
    gpu.state.line_width_set(1.0)


def dashed_line(x1, y, x2, color, dash=3, gap=4):
    x = x1
    while x < x2:
        line(((x, y), (min(x + dash, x2), y)), color, 1.0)
        x += dash + gap


def begin_scissor(x, y, w, h):
    try:
        gpu.state.scissor_set(int(round(x)), int(round(y)), int(round(w)), int(round(h)))
        gpu.state.scissor_test_set(True)
        return True
    except Exception:
        return False


def end_scissor():
    try:
        gpu.state.scissor_test_set(False)
    except Exception:
        pass


def circle_points(cx, cy, radius, segments=28, reverse=False):
    points = []
    order = range(segments - 1, -1, -1) if reverse else range(segments)
    for i in order:
        angle = math.tau * i / segments
        points.append((cx + math.cos(angle) * radius, cy + math.sin(angle) * radius))
    return points


def _arc_points(cx, cy, radius, start_angle, end_angle, steps=14):
    points = []
    for i in range(1, steps + 1):
        t = i / steps
        angle = start_angle + (end_angle - start_angle) * t
        points.append((cx + math.cos(angle) * radius, cy + math.sin(angle) * radius))
    return points


def receipt_outline_points(x, y, w, h, side_notches=None, bottom_notches=None):
    top = y + h
    bottom = y
    side_notches = sorted(side_notches or [], key=lambda item: item[0])
    bottom_notches = sorted(bottom_notches or [], key=lambda item: item[0])
    points = [(x, bottom)]

    for cx, radius in bottom_notches:
        if cx - radius <= x or cx + radius >= x + w:
            continue
        points.append((cx - radius, bottom))
        points.extend(_arc_points(cx, bottom, radius, math.pi, 0.0, 24))
    points.append((x + w, bottom))

    for cy, radius in side_notches:
        if cy - radius <= bottom or cy + radius >= top:
            continue
        points.append((x + w, cy - radius))
        points.extend(_arc_points(x + w, cy, radius, -math.pi * 0.5, -math.pi * 1.5, 36))
    points.append((x + w, top))
    points.append((x, top))

    for cy, radius in reversed(side_notches):
        if cy - radius <= bottom or cy + radius >= top:
            continue
        points.append((x, cy + radius))
        points.extend(_arc_points(x, cy, radius, math.pi * 0.5, -math.pi * 0.5, 36))
    points.append((x, bottom))
    return points


def _pack_side_notches(side_notches):
    values = [notch[0] for notch in sorted(side_notches or [], key=lambda item: item[0])[:24]]
    count = len(values)
    values.extend([0.0] * (24 - len(values)))
    return count, tuple(tuple(values[index:index + 4]) for index in range(0, 24, 4))


def _bottom_notch_data(bottom_notches):
    bottom_notches = sorted(bottom_notches or [], key=lambda item: item[0])
    if not bottom_notches:
        return 0, (0.0, 0.0, 1.0, 0.0)
    radius = bottom_notches[0][1]
    first = bottom_notches[0][0]
    if len(bottom_notches) > 1:
        spacing = bottom_notches[1][0] - bottom_notches[0][0]
    else:
        spacing = 1.0
    return len(bottom_notches), (radius, first, max(0.0001, spacing), 0.0)


def sdf_receipt_body(x, y, w, h, paper, border, border_width=1.1, side_notches=None, bottom_notches=None):
    if w <= 0 or h <= 0:
        return
    shader = _paper_shader()
    side_radius = (side_notches or [(0.0, 0.0)])[0][1] if side_notches else 0.0
    side_count, side_vectors = _pack_side_notches(side_notches)
    bottom_count, bottom_data = _bottom_notch_data(bottom_notches)
    pad = max(side_radius, bottom_data[0], border_width) + 2.0
    batch = _sdf_rect_batch(x, y, w, h, pad)

    gpu.state.blend_set("ALPHA")
    shader.bind()
    shader.uniform_float("uViewport", _VIEWPORT)
    shader.uniform_float("uRect", (x, y, w, h))
    shader.uniform_float("uFill", paper if paper else _EMPTY_COLOR)
    shader.uniform_float("uBorder", border if border else _EMPTY_COLOR)
    shader.uniform_float("uBorderW", border_width if border else 0.0)
    shader.uniform_float("uSideData", (side_radius, 0.0, 0.0, 0.0))
    for index, values in enumerate(side_vectors):
        shader.uniform_float(f"uSideY{index}", values)
    shader.uniform_int("uSideCount", side_count)
    shader.uniform_float("uBottomData", bottom_data)
    shader.uniform_int("uBottomCount", bottom_count)
    batch.draw(shader)
    gpu.state.blend_set("NONE")


def receipt_body(x, y, w, h, paper, border, border_width=1.1, holes=None, side_notches=None, bottom_notches=None):
    if not holes:
        sdf_receipt_body(x, y, w, h, paper, border, border_width, side_notches, bottom_notches)
        return
    outline = receipt_outline_points(x, y, w, h, side_notches, bottom_notches)
    if paper[3] > 0:
        tessellated([outline] + (holes or []), paper)
        _aa_edge_fringe(outline, paper, 1.25)
    _aa_outline(outline, border, border_width, 1.1)


def text(label, x, y, size=12, color=None, align="LEFT", bold=False, font_id=None):
    color = color or T["ink"]
    font_id = 0 if font_id is None else font_id
    blf.size(font_id, int(size))
    width, _height = blf.dimensions(font_id, label)
    if align == "CENTER":
        x -= width * 0.5
    elif align == "RIGHT":
        x -= width
    blf.color(font_id, *color)
    offsets = ((0.0, 0.0),)
    if bold:
        offsets = ((0.0, 0.0), (0.55, 0.0), (0.0, 0.5), (0.48, 0.42))
    for ox, oy in offsets:
        blf.position(font_id, x + ox, y + oy, 0)
        blf.draw(font_id, label)


def art_text(label, x, y, size=12, color=None, align="CENTER", bold=False):
    text(label, x, y, size, color, align, bold, _art_font())


def checker(x, y, size):
    cell = max(3, int(size / 4))
    colors = ((0.56, 0.56, 0.56, 1.0), (0.76, 0.76, 0.76, 1.0))
    for row in range(4):
        for col in range(4):
            rect(x + col * cell, y + row * cell, cell, cell, colors[(row + col) % 2])


def _asset_texture(path):
    cached = _ASSET_TEXTURES.get(path)
    if cached is not None:
        texture, _size = cached
        return texture or None
    try:
        image = bpy.data.images.load(path, check_existing=True)
        texture = gpu.texture.from_image(image)
    except Exception:
        texture = False
        image = None
    size = tuple(image.size) if image else (0, 0)
    _ASSET_TEXTURES[path] = (texture, size)
    return texture or None


def image_size(path):
    _asset_texture(path)
    cached = _ASSET_TEXTURES.get(path)
    if not cached:
        return None
    texture, size = cached
    if not texture or not size or size[0] <= 0 or size[1] <= 0:
        return None
    return size


def textured_rect(x, y, w, h, texture):
    shader = _image_shader()
    if not shader:
        return False
    points = ((x, y), (x + w, y), (x + w, y + h), (x, y + h))
    uvs = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    gpu.state.blend_set("ALPHA")
    batch = batch_for_shader(shader, "TRI_FAN", {"pos": points, "texCoord": uvs})
    shader.bind()
    shader.uniform_sampler("image", texture)
    batch.draw(shader)
    gpu.state.blend_set("NONE")
    return True


def image_rect(path, x, y, w, h):
    texture = _asset_texture(path)
    if not texture:
        return False
    return textured_rect(x, y, w, h, texture)


def soft_circle(cx, cy, radius, color, border=None, border_width=1.0):
    sdf_circle(cx, cy, radius, color, border, border_width)


def swatch(x, y, size, color, hover=False):
    x = round(x)
    y = round(y)
    size = round(size)
    radius = max(8, int(size * 0.38))
    border_width = 1 if hover else 0
    if len(color) > 3 and color[3] < 0.995:
        checker(x + border_width, y + border_width, size - border_width * 2)
    sdf_rounded_rect(
        x,
        y,
        size,
        size,
        radius,
        color,
        T["accent"] if hover else None,
        border_width,
    )
