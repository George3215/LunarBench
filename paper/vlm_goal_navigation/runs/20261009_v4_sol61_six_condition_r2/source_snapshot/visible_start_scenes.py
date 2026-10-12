"""Controlled visible-start, descent-occlusion, finite-bypass navigation scenes.
These replace the incompatible legacy 24; they are not the original terrains.
"""
import math
from scenes import Scene
from slabs import Slab, SlabTerrain


class PhysicalRamp(Slab):
    def geom_xml(self):
        # MuJoCo positive Y rotation slopes downward along positive X.
        length=math.hypot(self.x1-self.x0,self.z1-self.z0)
        theta=-self.pitch
        half_t=self.thickness_m/2
        cx=(self.x0+self.x1)/2-half_t*math.sin(theta)
        cz=(self.z0+self.z1)/2-half_t*math.cos(theta)
        return (f'<geom name="{self.name}" type="box" size="{length/2} {(self.y1-self.y0)/2} {half_t}" '
                f'pos="{cx} {(self.y0+self.y1)/2} {cz}" euler="0 {theta} 0" friction="1 .005 .0001" rgba="{self.rgba}" />\n')


def catalogue():
    result = {}
    for family in range(8):
        for variant in range(3):
            name = f'descent_{family+1:02d}_{variant+1}'
            h = 1.0 + .05 * (family % 3)
            wall_half = .55 + .10 * (family % 4)
            tx = 2.8 + .15 * (family // 4)
            sy = (-.25, 0., .25)[variant]
            ty = (.12, 0., -.12)[variant]
            yaw = math.atan2(ty-sy, tx+4.2)
            scene = Scene(name=name, terrain_profile='visible_start_descent',
                target_xy=(tx, ty), target_height_m=.60, target_z_absolute=.60,
                approach_xy=(-4.2, sy), lost_xy=(-4.2, sy), yaw_rad=yaw,
                terrain_params={'platform_height_m': h, 'wall_half_width_m': wall_half,
                                'wall_top_m': .68, 'ramp_start_x': -3.5, 'ramp_end_x': -.8},
                mechanism='visible from elevated start; descent occludes; finite wall permits bypass',
                notes=('Initial visible pose, no teleport after reset.',
                       'Eight geometry variants times three start/target offsets.'))
            terrain = SlabTerrain(name, [
                Slab(-7, 7, -4, 4, 0, 0, name='task3_floor'),
                Slab(-7, -3.5, -4, 4, h, h, name='task3_start_platform'),
                PhysicalRamp(-3.5, -.8, -4, 4, h, 0, name='task3_descent_ramp'),
                Slab(.0, .25, -wall_half, wall_half, .68, .68,
                     name='task3_occluder_wall', rgba='0.42 0.43 0.45 1')],
                {'size_m': [14, 8], 'profile': 'visible_start_descent',
                 'scene': name, **scene.terrain_params})
            result[name] = (scene, terrain)
    return result
