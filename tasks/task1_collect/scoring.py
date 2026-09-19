"""Physical collection events. Implementation tolerances are Python constants, in SI units."""
import itertools
import numpy as np
import mujoco

SETTLE_SECONDS = 0.5
MAX_LINEAR_SPEED = 0.08
MAX_ANGULAR_SPEED = 1.0
LIFT_CLEARANCE = 0.03
GRASP_SECONDS = 0.04
GEOMETRY_TOLERANCE = 0.002


class CollectionScore:
    def __init__(self, env):
        self.env = env
        m = env.model
        self.geoms = np.array([m.geom(f'sample_{i}').id for i in env.meta['rocks']], dtype=int)
        self.indices = {int(g): i for i, g in enumerate(self.geoms)}
        self.robot_geoms = set(env.robot.robot_geoms) if hasattr(env.robot, 'robot_geoms') else set()
        root = m.body(env.robot.base_body).id
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            while b and b != root:
                b = int(m.body_parentid[b])
            if b == root:
                self.robot_geoms.add(g)
        # Identify coupled sliding fingers by joint equality, independent of actuator order.
        self.fingers = {}
        for i in range(m.neq):
            if m.eq_type[i] != mujoco.mjtEq.mjEQ_JOINT:
                continue
            a, b = int(m.eq_obj1id[i]), int(m.eq_obj2id[i])
            if a < 0 or b < 0 or any(m.jnt_type[j] != mujoco.mjtJoint.mjJNT_SLIDE for j in (a, b)):
                continue
            for side, j in enumerate((a, b)):
                for g in self.robot_geoms:
                    if m.geom_bodyid[g] == m.jnt_bodyid[j]:
                        self.fingers[g] = 1 << side
        self.vertices = []
        for g in self.geoms:
            mesh = m.geom_dataid[g]
            start, count = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
            self.vertices.append(m.mesh_vert[start:start + count].copy())
        zone = env.config['collection_zone']
        self.inner_half = (env.zone_size - zone['wall_thickness_m']) / 2
        # Actual wall top, including its adaptation to uneven terrain.
        walls = [g for g in range(m.ngeom) if m.geom(g).name.startswith('zone_wall_')]
        if not walls:
            raise ValueError('Collection scoring requires physical box walls')
        self.top = min(float(m.geom_pos[g, 2] + m.geom_size[g, 2]) for g in walls)
        self.reset()

    def reset(self):
        n = len(self.geoms)
        self.grasp_time = np.zeros(n)
        self.lifted = np.zeros(n, dtype=bool)
        self.dwell = np.zeros(n)
        self.fall_charged = False
        self.terms = {'collection': 0., 'fall': 0.}

    def bounds(self, i):
        d = self.env.data
        g = self.geoms[i]
        vertices = self.vertices[i] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        return vertices.min(axis=0), vertices.max(axis=0)

    def update(self):
        e = self.env
        d, m = e.data, e.model
        dt = float(m.opt.timestep)
        masks = np.zeros(len(self.geoms), dtype=np.uint8)
        touching = np.zeros(len(self.geoms), dtype=bool)
        # Only inspect contacts involving the robot: static rock/terrain pairs are irrelevant.
        for c in d.contact:
            if c.dist > 0:
                continue
            a, b = int(c.geom1), int(c.geom2)
            if a in self.robot_geoms and b in self.indices:
                a, b = b, a
            if a in self.indices and b in self.robot_geoms:
                i = self.indices[a]
                touching[i] = True
                masks[i] |= self.fingers.get(b, 0)
        held = masks == 3
        self.grasp_time[:] = np.where(held, self.grasp_time + dt, 0.)
        for i in np.flatnonzero(held & ~self.lifted & (self.grasp_time >= GRASP_SECONDS)):
            low, high = self.bounds(i)
            # Highest terrain sample under the bounding footprint (not just its centre).
            xy = np.array(list(itertools.product((low[0], high[0]), (low[1], high[1]))))
            ground = max(float(e.field.height_local(*d.geom_xpos[self.geoms[i], :2])),
                         float(np.max(e.field.height_local(xy[:, 0], xy[:, 1]))))
            if low[2] - ground >= LIFT_CLEARANCE:
                self.lifted[i] = True
        positions = d.geom_xpos[self.geoms]
        candidates = self.lifted & ~touching & np.all(np.abs(positions[:, :2] - e.zone_center) < self.inner_half, axis=1)
        valid = np.zeros(len(self.geoms), dtype=bool)
        for i in np.flatnonzero(candidates):
            if i in e.collected:
                continue
            low, high = self.bounds(i)
            velocity = np.empty(6)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, e.rock_bodies[i], velocity, 0)
            ground = float(e.field.height_local(*positions[i, :2]))
            valid[i] = (np.all(low[:2] >= e.zone_center - self.inner_half) and
                        np.all(high[:2] <= e.zone_center + self.inner_half) and
                        low[2] >= ground - GEOMETRY_TOLERANCE and high[2] <= self.top and
                        np.linalg.norm(velocity[3:]) <= MAX_LINEAR_SPEED and
                        np.linalg.norm(velocity[:3]) <= MAX_ANGULAR_SPEED)
        self.dwell[:] = np.where(valid, self.dwell + dt, 0.)
        new = [int(i) for i in np.flatnonzero(self.dwell >= SETTLE_SECONDS) if i not in e.collected]
        e.collected.update(new)
        fall = bool(e.fallen and not self.fall_charged)
        self.fall_charged |= fall
        self.terms = {'collection': len(new) * float(e.config['reward']['per_rock']),
                      'fall': -float(e.config['reward']['fall_penalty']) if fall else 0.}
        return sum(self.terms.values())

    def state(self):
        return {'grasp_time': self.grasp_time.copy(), 'lifted': self.lifted.copy(),
                'dwell': self.dwell.copy(), 'fall_charged': self.fall_charged, 'terms': dict(self.terms)}

    def restore(self, state):
        for key in ('grasp_time', 'lifted', 'dwell'):
            value = np.asarray(state[key])
            if value.shape != getattr(self, key).shape or not np.isfinite(value).all():
                raise ValueError('Invalid collection state: ' + key)
            getattr(self, key)[:] = value
        self.fall_charged = bool(state['fall_charged'])
        self.terms = dict(state['terms'])
