"""Small native MuJoCo/GLFW viewer, retaining its window across collision swaps.

Only public GLFW and MuJoCo APIs. Rendering uses every supplied mesh face;
physics still uses its own unchanged integration timestep.
"""
from contextlib import nullcontext
import time
import math
import numpy as np
import glfw
import mujoco


class Viewer:
    def __init__(self, model, data, key_callback=None, title="MuJoCo : go2"):
        if not glfw.init():raise RuntimeError('GLFW initialization failed')
        self.window=glfw.create_window(1280,720,title,None,None)
        if not self.window:
            glfw.terminate();raise RuntimeError('Cannot create MuJoCo window')
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)  # Frame pacing belongs to the demo, not physics vsync.
        self.cam=mujoco.MjvCamera();mujoco.mjv_defaultCamera(self.cam)
        self.opt=mujoco.MjvOption();mujoco.mjv_defaultOption(self.opt)
        self.opt.geomgroup[3]=0  # Collision primitives are not the original visual mesh.
        self.key_callback=key_callback
        self.last_cursor=glfw.get_cursor_pos(self.window)
        self.context=None;self.scene=None;self.closed=False;self.reload_ms=0.
        self.replace_model(model,data)
        glfw.set_key_callback(self.window,self._key)
        glfw.set_cursor_pos_callback(self.window,self._cursor)
        glfw.set_mouse_button_callback(self.window,self._button)
        glfw.set_scroll_callback(self.window,self._scroll)

    @property
    def viewport(self):
        width,height=glfw.get_framebuffer_size(self.window)
        return mujoco.MjrRect(0,0,width,height)

    def lock(self):
        # Physics, input and rendering execute sequentially in this process.
        return nullcontext()

    def _key(self, window, key, scancode, action, mods):
        if action!=glfw.PRESS:return
        if key==glfw.KEY_ESCAPE:glfw.set_window_should_close(window,True)
        elif self.key_callback and key<128:self.key_callback(key)

    def _button(self, *_):
        self.last_cursor=glfw.get_cursor_pos(self.window)

    def _cursor(self, window, x, y):
        old_x,old_y=self.last_cursor;self.last_cursor=(x,y)
        left=glfw.get_mouse_button(window,glfw.MOUSE_BUTTON_LEFT)==glfw.PRESS
        right=glfw.get_mouse_button(window,glfw.MOUSE_BUTTON_RIGHT)==glfw.PRESS
        middle=glfw.get_mouse_button(window,glfw.MOUSE_BUTTON_MIDDLE)==glfw.PRESS
        if not(left or right or middle):return
        shift=glfw.get_key(window,glfw.KEY_LEFT_SHIFT)==glfw.PRESS or glfw.get_key(window,glfw.KEY_RIGHT_SHIFT)==glfw.PRESS
        if right:action=mujoco.mjtMouse.mjMOUSE_MOVE_H if shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif left:action=mujoco.mjtMouse.mjMOUSE_ROTATE_H if shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        else:action=mujoco.mjtMouse.mjMOUSE_ZOOM
        height=max(1,glfw.get_window_size(window)[1])
        mujoco.mjv_moveCamera(self.model,action,(x-old_x)/height,(y-old_y)/height,self.cam)

    def _scroll(self, window, x, y):
        mujoco.mjv_moveCamera(self.model,mujoco.mjtMouse.mjMOUSE_ZOOM,0,-.05*y,self.cam)

    def replace_model(self, model, data):
        started=time.monotonic();glfw.make_context_current(self.window)
        if self.context:self.context.free()
        self.model=model;self.data=data
        self.scene=mujoco.MjvScene(model,maxgeom=max(1000,model.ngeom*2))
        self.context=mujoco.MjrContext(model,mujoco.mjtFontScale.mjFONTSCALE_100)
        self.terrain=[]
        for i in range(model.ngeom):
            if model.geom_type[i]!=mujoco.mjtGeom.mjGEOM_HFIELD:continue
            h=int(model.geom_dataid[i]);a=int(model.hfield_adr[h]);n=int(model.hfield_nrow[h]*model.hfield_ncol[h])
            lo=float(model.hfield_data[a:a+n].min())*model.hfield_size[h,2]
            hi=float(model.hfield_data[a:a+n].max())*model.hfield_size[h,2]
            center=model.geom_pos[i].copy();center[2]+=(lo+hi)/2
            radius=float(np.linalg.norm([*model.hfield_size[h,:2],(hi-lo)/2]))
            self.terrain.append((i,center,radius))
        self.visible_terrain_tiles=len(self.terrain)
        self.reload_ms=(time.monotonic()-started)*1000

    def sync(self):
        glfw.make_context_current(self.window)
        # Conservatively cull whole out-of-view tiles, preserving all visible cells.
        hidden=[];previous_group5=self.opt.geomgroup[5]
        if len(self.terrain)>1:
            mujoco.mjv_updateCamera(self.model,self.data,self.cam,self.scene)
            a,b=self.scene.camera;eye=(a.pos+b.pos)*.5;forward=(a.forward+b.forward)*.5;up=(a.up+b.up)*.5
            right=np.cross(forward,up);rect=self.viewport
            tv=math.tan(math.radians(self.model.vis.global_.fovy)/2)
            th=tv*rect.width/max(1,rect.height)
            for i,center,radius in self.terrain:
                delta=center-eye;depth=float(delta@forward)
                if depth+radius<0 or abs(delta@up)>max(0,depth)*tv+radius*math.sqrt(1+tv*tv) or abs(delta@right)>max(0,depth)*th+radius*math.sqrt(1+th*th):
                    hidden.append((i,int(self.model.geom_group[i])));self.model.geom_group[i]=5
            self.opt.geomgroup[5]=0
        self.visible_terrain_tiles=len(self.terrain)-len(hidden)
        try:mujoco.mjv_updateScene(self.model,self.data,self.opt,None,self.cam,mujoco.mjtCatBit.mjCAT_ALL,self.scene)
        finally:
            for i,group in hidden:self.model.geom_group[i]=group
            self.opt.geomgroup[5]=previous_group5
        if len(self.terrain)>1:
            # Avoid shadow-map acne on centimetre terrain; geometry is unchanged.
            self.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=0
        rect=self.viewport
        if rect.width and rect.height:
            mujoco.mjr_render(rect,self.scene,self.context)
            glfw.swap_buffers(self.window)
        glfw.poll_events()

    def is_running(self):
        return not self.closed and not glfw.window_should_close(self.window)

    def close(self):
        if self.closed:return
        glfw.make_context_current(self.window)
        self.context.free();self.scene=None
        glfw.destroy_window(self.window);glfw.terminate();self.closed=True
