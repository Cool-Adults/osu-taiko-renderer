"""Minimal moderngl sprite batch for the catch renderer.

Owns a standalone EGL context and an offscreen RGBA framebuffer. Draws
textured/solid quads with straight-alpha blending in painter's order, then
reads back tightly-packed RGB24 for the ffmpeg pipe. Deliberately tiny and
self-contained so it can be discarded when the VRender branch takes over.
"""
from __future__ import annotations

import numpy as np

try:
    import moderngl
except Exception as e:  # noqa: BLE001
    raise RuntimeError("moderngl is required for the catch renderer") from e

from .models import Sprite

_VERT = """
#version 330
in vec2 in_pos;      // unit quad corner [-0.5,0.5]
in vec2 in_uv;
uniform vec2 u_screen;   // (w, h) in px
uniform vec2 u_center;   // sprite center in px (origin top-left)
uniform vec2 u_size;     // sprite w,h in px
uniform float u_rot;     // radians
out vec2 v_uv;
void main() {
    vec2 p = in_pos * u_size;
    float c = cos(u_rot), s = sin(u_rot);
    p = vec2(p.x * c - p.y * s, p.x * s + p.y * c);
    vec2 px = u_center + p;
    // px -> clip, with y flipped (top-left origin)
    vec2 ndc = vec2(px.x / u_screen.x * 2.0 - 1.0,
                    1.0 - px.y / u_screen.y * 2.0);
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_uv = in_uv;
}
"""

_FRAG = """
#version 330
in vec2 v_uv;
uniform sampler2D u_tex;
uniform vec4 u_color;
out vec4 f_color;
void main() {
    vec4 t = texture(u_tex, v_uv);
    f_color = t * u_color;
}
"""


class SpriteRenderer:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        # Honor R3D_EGL_DEVICE_INDEX so renders pin to the right GPU (pool
        # isolation: e.g. 1070=index 1 for Pool B). EGL ignores
        # CUDA_VISIBLE_DEVICES, so the device must be selected explicitly.
        import os
        dev = os.environ.get("R3D_EGL_DEVICE_INDEX", "").strip()
        if dev.isdigit():
            self.ctx = moderngl.create_context(
                standalone=True, backend="egl", device_index=int(dev))
        else:
            self.ctx = moderngl.create_context(standalone=True, backend="egl")
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        self.prog = self.ctx.program(vertex_shader=_VERT, fragment_shader=_FRAG)
        # unit quad centered at origin, uv 0..1
        # in_pos.y=-0.5 renders at screen-top -> texture-top (v=0); in_pos.y=+0.5
        # renders at screen-bottom -> texture-bottom (v=1). (Matters for
        # vertically-asymmetric sprites like the catcher.)
        quad = np.array([
            -0.5, -0.5, 0.0, 0.0,
             0.5, -0.5, 1.0, 0.0,
            -0.5,  0.5, 0.0, 1.0,
             0.5,  0.5, 1.0, 1.0,
        ], dtype="f4")
        self.vbo = self.ctx.buffer(quad.tobytes())
        self.vao = self.ctx.vertex_array(
            self.prog, [(self.vbo, "2f 2f", "in_pos", "in_uv")],
        )
        self.prog["u_screen"].value = (float(width), float(height))

        rb = self.ctx.renderbuffer((width, height))
        self.fbo = self.ctx.framebuffer(color_attachments=[rb])
        self._textures: dict[str, moderngl.Texture] = {}
        self._white = self._make_texture_rgba(np.full((1, 1, 4), 255, dtype="u1"))

        # async PBO ring state (read_rgb_async / read_drain) — ported from
        # the std renderer's proven pipeline (osu_std_renderer/render/gl.py)
        self._pbos: list["moderngl.Buffer"] | None = None
        self._pbo_head = 0
        self._pbo_tail = 0

    # --- texture management ---------------------------------------------------

    def upload_texture(self, key: str, rgba: np.ndarray) -> None:
        """rgba: HxWx4 uint8 array (top-left origin)."""
        if rgba.dtype != np.uint8:
            rgba = rgba.astype("u1")
        if rgba.shape[2] == 3:
            a = np.full(rgba.shape[:2] + (1,), 255, dtype="u1")
            rgba = np.concatenate([rgba, a], axis=2)
        self._textures[key] = self._make_texture_rgba(rgba)

    def has_texture(self, key: str) -> bool:
        return key in self._textures

    def _make_texture_rgba(self, rgba: np.ndarray) -> "moderngl.Texture":
        h, w = rgba.shape[:2]
        tex = self.ctx.texture((w, h), 4, rgba.tobytes())
        tex.build_mipmaps()
        tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        return tex

    # --- drawing --------------------------------------------------------------

    def begin(self, clear=(0.04, 0.04, 0.06)) -> None:
        self.fbo.use()
        self.ctx.clear(*clear)

    def draw(self, sprites: list[Sprite]) -> None:
        for sp in sprites:
            tex = self._textures.get(sp.texture_key) if sp.texture_key else self._white
            if tex is None:
                tex = self._white
            tex.use(location=0)
            self.prog["u_tex"].value = 0
            self.prog["u_color"].value = sp.color
            self.prog["u_center"].value = (sp.x, sp.y)
            self.prog["u_size"].value = (sp.w, sp.h)
            self.prog["u_rot"].value = sp.rotation
            self.vao.render(moderngl.TRIANGLE_STRIP)

    _PBO_RING = 3

    def read_rgb_async(self) -> "np.ndarray | None":
        """Queue an async readback of the current fbo into a small PBO
        ring and return the OLDEST completed frame (top-left origin), or
        None while the ring is still filling. Frames come back in strict
        submission order — the render loop pushes them straight to ffmpeg,
        so the byte stream is identical to the synchronous read_rgb path,
        just ~RING-1 frames late. read_drain() flushes the tail. (Ported
        from the std renderer's proven osu_std_renderer/render/gl.py.)"""
        if self._pbos is None:
            size = self.width * self.height * 3
            self._pbos = [self.ctx.buffer(reserve=size)
                          for _ in range(self._PBO_RING)]
        buf = self._pbos[self._pbo_head % len(self._pbos)]
        self.fbo.read_into(buf, components=3, alignment=1)
        self._pbo_head += 1
        if self._pbo_head - self._pbo_tail < len(self._pbos):
            return None
        return self._pop_pbo()

    def _pop_pbo(self) -> np.ndarray:
        buf = self._pbos[self._pbo_tail % len(self._pbos)]
        self._pbo_tail += 1
        data = buf.read()
        arr = np.frombuffer(data, dtype="u1").reshape(
            (self.height, self.width, 3))
        return np.flipud(arr)  # same orientation contract as read_rgb

    def read_drain(self) -> list:
        """Return every frame still in flight, oldest first (map end or
        the gameplay->outro boundary)."""
        out = []
        while self._pbos is not None and self._pbo_tail < self._pbo_head:
            out.append(self._pop_pbo())
        return out

    def read_rgb(self) -> np.ndarray:
        """Return HxWx3 uint8, top-left origin (ready for ffmpeg rgb24).

        Note: a 3-component read is faster end-to-end than reading RGBA and
        dropping alpha — the channel-drop forces a strided copy that costs more
        than the faster aligned transfer saves."""
        data = self.fbo.read(components=3, alignment=1)
        arr = np.frombuffer(data, dtype="u1").reshape((self.height, self.width, 3))
        # moderngl reads bottom-left origin; flip to top-left (view; copied once
        # downstream where the frame is made contiguous).
        return np.flipud(arr)

    def release(self) -> None:
        try:
            self.ctx.release()
        except Exception:  # noqa: BLE001
            pass
