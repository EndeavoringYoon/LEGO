"""Process-separated MuJoCo viewer.

The viewer window runs in its own process and keeps rendering the latest state at a
fixed rate, so a slow computation loop in the main process (decoding, IK, retargeting)
no longer drops frames or freezes camera control. The main process only sends the
state (qpos, mocap) and the markers of each frame, and returns immediately.

RenderProcessViewer exposes the parts of the MuJoCoMinimalViewer interface that
MuJoCoParserClass uses (cam, vopt, scn.flags, add_marker, add_overlay, render, close,
is_alive, key states), so it is enabled with init_viewer(render_process=True) and the
existing render loops work unchanged.

grab_image() asks the render process for the current frame and waits for it, and
viewer_pause() / SPACE pause both processes as before. Not supported in this mode: RGB
overlays, figures, mouse picking and perturbation (init_viewer falls back to the
in-process viewer when RGB overlays or figures are requested).
"""
import time
import queue
import pickle
import threading
import traceback
import multiprocessing as mp
import numpy as np
import mujoco


def _to_plain(value):
    """Convert MuJoCo enums to int so that marker dicts can be pickled."""
    if isinstance(value, (bool, int, float, str, type(None))):
        return value
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, (list, tuple)):
        return np.asarray(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


class RenderProcessViewer:
    """Main-process proxy of a MuJoCo viewer that renders in a separate process."""

    def __init__(
            self,
            model,
            data,
            title      = 'MuJoCo Viewer',
            width      = 1400,
            height     = 1000,
            hide_menus = True,
            maxgeom    = 50000,
            fontscale  = mujoco.mjtFontScale.mjFONTSCALE_200.value,
            render_hz  = 60,
            throttle   = True,
        ):
        """
        Parameters:
            model (mujoco.MjModel): Model to show. Reassigning viewer.model later sends the new model.
            data (mujoco.MjData): Data whose qpos / mocap are sent on every render().
            title, width, height, hide_menus, maxgeom, fontscale: Same as MuJoCoMinimalViewer.
            render_hz (float): Frame rate of the render process.
            throttle (bool): If True, render() sleeps so that the caller's loop runs at most at
                render_hz, which keeps motion playback at a watchable speed (as with the in-process
                viewer). If False, render() never waits: calls arriving faster than render_hz only
                skip sending, so a fast computation loop is not slowed down by the viewer.
        """
        self.model = model
        self.data  = data
        self.cfg   = dict(title=title, width=int(width), height=int(height),
                          hide_menus=hide_menus, maxgeom=maxgeom, fontscale=fontscale,
                          render_hz=float(render_hz))
        self.throttle = throttle

        # Local copies of the viewer settings, edited by MuJoCoParserClass.set_viewer()
        self.cam  = mujoco.MjvCamera()
        self.vopt = mujoco.MjvOption()
        self.scn  = mujoco.MjvScene(model, maxgeom=1) # only scn.flags is used
        self.pert = mujoco.MjvPerturb()
        self.ctx      = None # no GL context in the main process
        self.viewport = None

        # Per-frame items, sent with the next render()
        self._markers = []
        self._overlay = []

        # Key states mirrored from the render process
        self._paused_flag      = False
        self._key_pressed      = None
        self._is_key_pressed   = False
        self._key_pressed_set  = set()
        self._key_repeated_set = set()
        self._left_double_click_pressed  = False
        self._right_double_click_pressed = False
        self._last_mouse_x = 0
        self._last_mouse_y = 0

        self._proc       = None
        self._conn       = None
        self._alive      = None
        self._cam_shared = None
        self._n_frames   = None
        self._sent_model = None
        self._last_send  = 0.0
        self._outbox     = None

    # ------------------------------------------------------------------ lifecycle
    def start(self):
        """Spawn the render process with the current model and settings."""
        ctx = mp.get_context('spawn') # fork is unsafe with GL / torch state
        self._conn, child_conn = ctx.Pipe()
        self._alive      = ctx.Value('b', 1)
        self._cam_shared = ctx.Array('d', 6) # azimuth, distance, elevation, lookat[3]
        self._n_frames   = ctx.Value('l', 0)
        self._proc = ctx.Process(
            target = _render_main,
            args   = (child_conn, self._alive, self._cam_shared, self._n_frames, self.cfg),
            daemon = True,
        )
        self._proc.start()
        # Messages go through a sender thread, so a large model or a busy render process
        # never blocks the caller
        self._outbox = queue.Queue()
        threading.Thread(target=self._sender, daemon=True).start()
        self._send_model()
        return self

    def _sender(self):
        while True:
            msg = self._outbox.get()
            if msg is None:
                return
            try:
                self._conn.send(msg)
            except (BrokenPipeError, OSError):
                self._alive.value = 0
                return

    @property
    def is_alive(self):
        """True while the render process and its window are open."""
        return (self._proc is not None and self._proc.is_alive()
                and bool(self._alive.value))

    @property
    def n_frames(self):
        """Number of frames drawn by the render process so far."""
        return 0 if self._n_frames is None else self._n_frames.value

    def close(self):
        """Close the window and stop the render process."""
        if self._proc is None:
            return
        self._send(('close',))
        self._outbox.put(None)
        self._proc.join(timeout=0.5)
        if self._proc.is_alive(): # e.g. still starting up
            self._proc.terminate()
            self._proc.join(timeout=1.0)
        self._proc = None

    @property
    def _paused(self):
        return self._paused_flag

    @_paused.setter
    def _paused(self, value):
        # Set by MuJoCoParserClass.viewer_pause(); the render process shows the pause and
        # the next render() waits until SPACE is pressed in the window, as in the in-process viewer
        if value and not self._paused_flag and self._conn is not None:
            self._send(('pause',))
        self._paused_flag = bool(value)

    # ------------------------------------------------------------------ drawing
    def add_marker(self, **marker_params):
        """Queue a marker (same keyword arguments as MuJoCoMinimalViewer.add_marker)."""
        self._markers.append({k: _to_plain(v) for k, v in marker_params.items()})

    def add_overlay(self, loc='bottom left', gridpos=None, text1='', text2=''):
        """Queue a text overlay for the next frame."""
        self._overlay.append((loc, text1, text2))

    def plot_rgb_overlay(self, rgb=None, loc='top right'):
        raise NotImplementedError("RGB overlays are not supported with render_process=True.")

    def push_settings(self):
        """Send camera / visualization options to the render process."""
        self._send(('settings', self._settings()))

    def render(self):
        """Send the current state and markers; returns without waiting for the frame."""
        if not self.is_alive:
            return
        if self.model is not self._sent_model:
            self._send_model()

        # At most render_hz updates; a slower loop is never delayed
        period = 1.0 / self.cfg['render_hz']
        wait   = self._last_send + period - time.time()
        if wait > 0:
            if not self.throttle:
                self._markers = []
                self._overlay = []
                return
            time.sleep(wait)
        self._last_send = time.time()

        if self._outbox.qsize() > 4: # render process is behind (e.g. paused): keep only newer states
            self._markers = []
            self._overlay = []
            self._receive_events()
            return
        d = self.data
        self._send(('state', d.qpos.copy(), d.mocap_pos.copy(), d.mocap_quat.copy(),
                    self._markers, self._overlay))
        self._markers = []
        self._overlay = []
        self._receive_events()
        while self._paused_flag and self.is_alive: # wait for SPACE in the window
            self._receive_events(block=True)

    def grab_image(self):
        """Return the RGB image (H x W x 3) of the current state, rendered by the render process."""
        if not self.is_alive:
            raise RuntimeError("The render process is not running.")
        if self.model is not self._sent_model:
            self._send_model()
        d = self.data
        self._send(('state', d.qpos.copy(), d.mocap_pos.copy(), d.mocap_quat.copy(),
                    self._markers, self._overlay))
        self._send(('grab',))
        self._image = None
        while self._image is None and self.is_alive:
            self._receive_events(block=True)
        return self._image

    # ------------------------------------------------------------------ internals
    def _settings(self):
        return dict(
            azimuth   = float(self.cam.azimuth),
            distance  = float(self.cam.distance),
            elevation = float(self.cam.elevation),
            lookat    = np.array(self.cam.lookat),
            vopt_flags = np.array(self.vopt.flags),
            geomgroup  = np.array(self.vopt.geomgroup),
            scn_flags  = np.array(self.scn.flags),
        )

    def _send_model(self):
        self._send(('model', pickle.dumps(self.model), self._settings()))
        self._sent_model = self.model

    def _send(self, msg):
        self._outbox.put(msg)

    def _receive_events(self, block=False):
        """Mirror key events, grabbed images and camera pose from the render process."""
        try:
            while self._conn.poll(0.05 if block else 0):
                kind, payload = self._conn.recv()
                block = False
                if kind == 'image':
                    self._image = payload
                elif kind == 'key':
                    key, action, paused = payload
                    self._paused_flag = paused
                    if action == 1:   # glfw.PRESS
                        self._key_pressed_set.add(key)
                    elif action == 2: # glfw.REPEAT
                        self._key_repeated_set.add(key)
                    elif action == 0: # glfw.RELEASE
                        self._key_pressed_set.discard(key)
                        self._key_repeated_set.discard(key)
                    self._key_pressed    = key
                    self._is_key_pressed = True
        except (EOFError, OSError):
            self._alive.value = 0
        c = self._cam_shared
        self.cam.azimuth, self.cam.distance, self.cam.elevation = c[0], c[1], c[2]
        self.cam.lookat[:] = c[3:6]


def _apply_settings(viewer, settings):
    viewer.cam.azimuth   = settings['azimuth']
    viewer.cam.distance  = settings['distance']
    viewer.cam.elevation = settings['elevation']
    viewer.cam.lookat[:] = settings['lookat']
    viewer.vopt.flags[:]     = settings['vopt_flags']
    viewer.vopt.geomgroup[:] = settings['geomgroup']
    viewer.scn.flags[:]      = settings['scn_flags']


def _render_main(conn, alive, cam_shared, n_frames, cfg):
    """Entry point of the render process: render the latest received state at render_hz."""
    try:
        import glfw
        from mujoco_parser import MuJoCoMinimalViewer

        kind, model_bytes, settings = conn.recv()
        model = pickle.loads(model_bytes)
        data  = mujoco.MjData(model)
        viewer = MuJoCoMinimalViewer(
            model, data,
            title        = cfg['title'],
            width        = cfg['width'],
            height       = cfg['height'],
            hide_menus   = cfg['hide_menus'],
            maxgeom      = cfg['maxgeom'],
            perturbation = False,
            n_fig        = 0,
            use_rgb_overlay = False,
        )
        viewer.ctx = mujoco.MjrContext(model, cfg['fontscale'])
        _apply_settings(viewer, settings)

        # Forward key events to the main process
        def key_callback(window, key, scancode, action, mods):
            viewer._key_callback(window, key, scancode, action, mods)
            try:
                conn.send(('key', (key, action, viewer._paused)))
            except (BrokenPipeError, OSError):
                pass
        glfw.set_key_callback(viewer.window, key_callback)

        markers, overlay = [], []
        grab   = False
        period = 1.0 / cfg['render_hz']
        while viewer.is_alive:
            t_start = time.time()
            # Apply every pending message; only the latest state matters
            while conn.poll():
                msg = conn.recv()
                if msg[0] == 'close':
                    viewer.close()
                    break
                elif msg[0] == 'model':
                    model = pickle.loads(msg[1])
                    data  = mujoco.MjData(model)
                    viewer.ctx.free()
                    viewer.model, viewer.data = model, data
                    viewer.scn = mujoco.MjvScene(model, maxgeom=cfg['maxgeom'])
                    viewer.ctx = mujoco.MjrContext(model, cfg['fontscale'])
                    _apply_settings(viewer, msg[2])
                    markers, overlay = [], []
                elif msg[0] == 'settings':
                    _apply_settings(viewer, msg[1])
                elif msg[0] == 'pause':
                    viewer._paused = True
                elif msg[0] == 'grab':
                    grab = True
                    break # render the state received just before the request
                elif msg[0] == 'state':
                    _, qpos, mocap_pos, mocap_quat, markers, overlay = msg
                    if qpos.shape == data.qpos.shape:
                        data.qpos[:] = qpos
                    if mocap_pos.shape == data.mocap_pos.shape:
                        data.mocap_pos[:]  = mocap_pos
                        data.mocap_quat[:] = mocap_quat
            if not viewer.is_alive:
                break

            mujoco.mj_forward(model, data)
            viewer._markers[:] = list(markers) # markers stay until the next state arrives
            for loc, text1, text2 in overlay:
                viewer.add_overlay(loc=loc, text1=text1, text2=text2)
            if viewer._paused: # tell the main process; render() below blocks until SPACE
                conn.send(('key', (glfw.KEY_SPACE, -1, True)))
            viewer.render()
            n_frames.value += 1
            if grab and viewer.is_alive:
                img = np.zeros((viewer.viewport.height, viewer.viewport.width, 3), dtype=np.uint8)
                mujoco.mjr_render(viewer.viewport, viewer.scn, viewer.ctx)
                mujoco.mjr_readPixels(img, None, viewer.viewport, viewer.ctx)
                conn.send(('image', np.flipud(img).copy()))
                grab = False

            cam = viewer.cam
            cam_shared[0], cam_shared[1], cam_shared[2] = cam.azimuth, cam.distance, cam.elevation
            cam_shared[3:6] = list(cam.lookat)

            wait = period - (time.time() - t_start)
            if wait > 0:
                time.sleep(wait)
    except Exception:
        traceback.print_exc()
    finally:
        alive.value = 0
