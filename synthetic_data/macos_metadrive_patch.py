"""Make MetaDrive's RGB camera observation work headlessly on Apple Silicon macOS.

SYNTHETIC SOFTWARE TEST FIXTURE INFRASTRUCTURE -- see synthetic_data/README.md.
This module patches third-party simulator internals, not accident_risk code,
and the resulting imagery/telemetry is never evidence of model performance
(AGENTS.md: "Synthetic fixtures are software tests, never evidence of model
performance").

Root causes found by direct inspection of metadrive-simulator 0.4.3 on this
machine (arm64 macOS, Panda3D 1.10.13 Cocoa backend):

1. ``EngineCore`` forces ``RENDER_MODE_ONSCREEN`` whenever ``is_mac()`` is
   true, because upstream Panda3D offscreen buffers were historically
   unreliable on macOS. On this machine opening an actual onscreen window
   fails in a non-interactive process (no window-server session), while a
   plain ``window-type offscreen`` Panda3D buffer opens fine. So we force
   the offscreen path by stubbing ``is_mac``.
2. With ``main_camera`` disabled (image-observation-only configs collapse
   the main window to a 1x1 buffer) and the default 8x multisample request,
   Cocoa's ``choosePixelFormat`` cannot find a usable pixel format at all
   ("Could not find a usable pixel format") -- an 8x-multisampled 1x1
   offscreen buffer is not a combination the macOS Cocoa GL driver on this
   machine will create. We disable multisampling via PRC overrides.
3. Both ``metadrive.third_party.simplepbr.init`` (the global post-process
   pipeline) and ``RGBCamera._setup_effect`` (the per-sensor tonemap
   buffer) hard-code ``msaa_samples=16`` in Python, bypassing the PRC
   overrides from (2) entirely, so the same pixel-format failure recurs
   one level deeper and crashes with ``AttributeError: 'NoneType' object
   has no attribute 'set_shader'`` once the (silently failed) render target
   is used. We patch both call sites to request 0 samples.

None of this touches accident_risk itself; it only needs to run once,
before the first ``MetaDriveEnv`` is constructed, in the process generating
synthetic imagery.
"""

from __future__ import annotations

import panda3d.core as p3d
from direct.filter.FilterManager import FilterManager
from panda3d.core import loadPrcFileData

_PATCHED = False


def apply() -> None:
    """Idempotently patch MetaDrive/Panda3D so headless RGB rendering works here."""
    global _PATCHED
    if _PATCHED:
        return

    import metadrive.engine.core.engine_core as engine_core

    # (1) Force the offscreen code path instead of macOS's onscreen fallback.
    engine_core.is_mac = lambda: False

    # (2) Disable multisampling globally before any window/buffer is created.
    loadPrcFileData("", "framebuffer-multisample 0")
    loadPrcFileData("", "multisamples 0")

    # (3a) metadrive.third_party.simplepbr.init(msaa_samples=16, ...) for the
    # main post-process pipeline -- force msaa_samples=0 regardless of the
    # caller's request.
    _original_pbr_init = engine_core.init

    def _pbr_init_no_msaa(*args: object, **kwargs: object) -> object:
        kwargs["msaa_samples"] = 0
        return _original_pbr_init(*args, **kwargs)

    engine_core.init = _pbr_init_no_msaa

    # (3b) RGBCamera._setup_effect hard-codes fbprops.set_multisamples(16)
    # for its own per-sensor tonemap buffer. Replace with an identical copy
    # that requests 0 samples; FrameBufferProperties is an immutable
    # extension type, so the call argument must be patched, not the method.
    from metadrive.component.sensors.rgb_camera import RGBCamera
    from metadrive.constants import CameraTagStateKey, Semantics
    from metadrive.engine.core.terrain import Terrain
    from metadrive.third_party.simplepbr import _load_shader_str

    def _setup_effect_no_msaa(self: RGBCamera) -> None:
        cam = self.get_cam().node()
        cam.setTagStateKey(CameraTagStateKey.RGB)
        cam.setTagState(
            Semantics.TERRAIN.label,
            Terrain.make_render_state(self.engine, "terrain.vert.glsl", "terrain.frag.glsl"),
        )
        self.scene_tex = None
        self.manager = FilterManager(self.buffer, self.cam)
        fbprops = p3d.FrameBufferProperties()
        fbprops.float_color = True
        fbprops.set_rgba_bits(16, 16, 16, 16)
        fbprops.set_depth_bits(24)
        fbprops.set_multisamples(0)
        self.scene_tex = p3d.Texture()
        self.scene_tex.set_format(p3d.Texture.F_rgba16)
        self.scene_tex.set_component_type(p3d.Texture.T_float)
        self.tonemap_quad = self.manager.render_scene_into(colortex=self.scene_tex, fbprops=fbprops)
        post_vert_str = _load_shader_str("post.vert", {})
        post_frag_str = _load_shader_str("tonemap.frag", {})
        tonemap_shader = p3d.Shader.make(p3d.Shader.SL_GLSL, vertex=post_vert_str, fragment=post_frag_str)
        self.tonemap_quad.set_shader(tonemap_shader)
        self.tonemap_quad.set_shader_input("tex", self.scene_tex)
        self.tonemap_quad.set_shader_input("exposure", 1.0)

    RGBCamera._setup_effect = _setup_effect_no_msaa

    _PATCHED = True
