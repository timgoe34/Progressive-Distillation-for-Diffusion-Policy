#@markdown ### **Imports**
# diffusion policy import
from typing import Tuple, Sequence, Callable
import numpy as np
import torch
import torch.nn as nn
import collections
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import torchvision

from copy import deepcopy       # for progressive destillation
import time                     # for timing stuff
import torch.nn.functional as F

# env import
import gym                  # for gyms
from gym import spaces  
import pygame               # for games
import pymunk               # for games
import pymunk.pygame_util
from pymunk.space_debug_draw_options import SpaceDebugColor
from pymunk.vec2d import Vec2d
import shapely.geometry as sg   # for shapes in games
import cv2                      # for generating videos
import skimage.transform as st  # for transforming images for video preparation
import gdown                    # for downloading from google drive
import os                       # for loading files, checking paths..
import zarr                     # for loading zarr zipped objects (used for loading in large training data)

# import new network
from utils.network import * 


#from huggingface_hub.utils import IGNORE_GIT_FOLDER_PATTERNS

#@markdown ### **Environment**
#@markdown Defines a PyMunk-based Push-T environment `PushTEnv`.
#@markdown
#@markdown **Goal**: push the gray T-block into the green area.
#@markdown
#@markdown Adapted from [Implicit Behavior Cloning](https://implicitbc.github.io/)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

positive_y_is_up: bool = False
"""Make increasing values of y point upwards.

When True::

    y
    ^
    |      . (3, 3)
    |
    |   . (2, 2)
    |
    +------ > x

When False::

    +------ > x
    |
    |   . (2, 2)
    |
    |      . (3, 3)
    v
    y

"""

def to_pygame(p: Tuple[float, float], surface: pygame.Surface) -> Tuple[int, int]:
    """Convenience method to convert pymunk coordinates to pygame surface
    local coordinates.

    Note that in case positive_y_is_up is False, this function wont actually do
    anything except converting the point to integers.
    """
    if positive_y_is_up:
        return round(p[0]), surface.get_height() - round(p[1])
    else:
        return round(p[0]), round(p[1])


def light_color(color: SpaceDebugColor):
    color = np.minimum(1.2 * np.float32([color.r, color.g, color.b, color.a]), np.float32([255]))
    color = SpaceDebugColor(r=color[0], g=color[1], b=color[2], a=color[3])
    return color

class DrawOptions(pymunk.SpaceDebugDrawOptions):
    def __init__(self, surface: pygame.Surface) -> None:
        """Draw a pymunk.Space on a pygame.Surface object.

        Typical usage::

        >>> import pymunk
        >>> surface = pygame.Surface((10,10))
        >>> space = pymunk.Space()
        >>> options = pymunk.pygame_util.DrawOptions(surface)
        >>> space.debug_draw(options)

        You can control the color of a shape by setting shape.color to the color
        you want it drawn in::

        >>> c = pymunk.Circle(None, 10)
        >>> c.color = pygame.Color("pink")

        See pygame_util.demo.py for a full example

        Since pygame uses a coordiante system where y points down (in contrast
        to many other cases), you either have to make the physics simulation
        with Pymunk also behave in that way, or flip everything when you draw.

        The easiest is probably to just make the simulation behave the same
        way as Pygame does. In that way all coordinates used are in the same
        orientation and easy to reason about::

        >>> space = pymunk.Space()
        >>> space.gravity = (0, -1000)
        >>> body = pymunk.Body()
        >>> body.position = (0, 0) # will be positioned in the top left corner
        >>> space.debug_draw(options)

        To flip the drawing its possible to set the module property
        :py:data:`positive_y_is_up` to True. Then the pygame drawing will flip
        the simulation upside down before drawing::

        >>> positive_y_is_up = True
        >>> body = pymunk.Body()
        >>> body.position = (0, 0)
        >>> # Body will be position in bottom left corner

        :Parameters:
                surface : pygame.Surface
                    Surface that the objects will be drawn on
        """
        self.surface = surface
        super(DrawOptions, self).__init__()

    def draw_circle(
        self,
        pos: Vec2d,
        angle: float,
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        p = to_pygame(pos, self.surface)

        pygame.draw.circle(self.surface, fill_color.as_int(), p, round(radius), 0)
        pygame.draw.circle(self.surface, light_color(fill_color).as_int(), p, round(radius-4), 0)

        circle_edge = pos + Vec2d(radius, 0).rotated(angle)
        p2 = to_pygame(circle_edge, self.surface)
        line_r = 2 if radius > 20 else 1
        # pygame.draw.lines(self.surface, outline_color.as_int(), False, [p, p2], line_r)

    def draw_segment(self, a: Vec2d, b: Vec2d, color: SpaceDebugColor) -> None:
        p1 = to_pygame(a, self.surface)
        p2 = to_pygame(b, self.surface)

        pygame.draw.aalines(self.surface, color.as_int(), False, [p1, p2])

    def draw_fat_segment(
        self,
        a: Tuple[float, float],
        b: Tuple[float, float],
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        p1 = to_pygame(a, self.surface)
        p2 = to_pygame(b, self.surface)

        r = round(max(1, radius * 2))
        pygame.draw.lines(self.surface, fill_color.as_int(), False, [p1, p2], r)
        if r > 2:
            orthog = [abs(p2[1] - p1[1]), abs(p2[0] - p1[0])]
            if orthog[0] == 0 and orthog[1] == 0:
                return
            scale = radius / (orthog[0] * orthog[0] + orthog[1] * orthog[1]) ** 0.5
            orthog[0] = round(orthog[0] * scale)
            orthog[1] = round(orthog[1] * scale)
            points = [
                (p1[0] - orthog[0], p1[1] - orthog[1]),
                (p1[0] + orthog[0], p1[1] + orthog[1]),
                (p2[0] + orthog[0], p2[1] + orthog[1]),
                (p2[0] - orthog[0], p2[1] - orthog[1]),
            ]
            pygame.draw.polygon(self.surface, fill_color.as_int(), points)
            pygame.draw.circle(
                self.surface,
                fill_color.as_int(),
                (round(p1[0]), round(p1[1])),
                round(radius),
            )
            pygame.draw.circle(
                self.surface,
                fill_color.as_int(),
                (round(p2[0]), round(p2[1])),
                round(radius),
            )

    def draw_polygon(
        self,
        verts: Sequence[Tuple[float, float]],
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        ps = [to_pygame(v, self.surface) for v in verts]
        ps += [ps[0]]

        radius = 2
        pygame.draw.polygon(self.surface, light_color(fill_color).as_int(), ps)

        if radius > 0:
            for i in range(len(verts)):
                a = verts[i]
                b = verts[(i + 1) % len(verts)]
                self.draw_fat_segment(a, b, radius, fill_color, fill_color)

    def draw_dot(
        self, size: float, pos: Tuple[float, float], color: SpaceDebugColor
    ) -> None:
        p = to_pygame(pos, self.surface)
        pygame.draw.circle(self.surface, color.as_int(), p, round(size), 0)


def pymunk_to_shapely(body, shapes):
    geoms = list()
    for shape in shapes:
        if isinstance(shape, pymunk.shapes.Poly):
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts += [verts[0]]
            geoms.append(sg.Polygon(verts))
        else:
            raise RuntimeError(f'Unsupported shape type {type(shape)}')
    geom = sg.MultiPolygon(geoms)
    return geom

# env
class PushTEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}
    reward_range = (0., 1.)

    def __init__(self,
            legacy=False,
            block_cog=None, damping=None,
            render_action=True,
            render_size=96,
            reset_to_state=None
        ):
        self._seed = None
        self.seed()
        self.window_size = ws = 512  # The size of the PyGame window
        self.render_size = render_size
        self.sim_hz = 100
        # Local controller params.
        self.k_p, self.k_v = 100, 20    # PD control.z
        self.control_hz = self.metadata['video.frames_per_second']
        # legcay set_state for data compatiblity
        self.legacy = legacy

        # agent_pos, block_pos, block_angle
        self.observation_space = spaces.Box(
            low=np.array([0,0,0,0,0], dtype=np.float64),
            high=np.array([ws,ws,ws,ws,np.pi*2], dtype=np.float64),
            shape=(5,),
            dtype=np.float64
        )

        # positional goal for agent
        self.action_space = spaces.Box(
            low=np.array([0,0], dtype=np.float64),
            high=np.array([ws,ws], dtype=np.float64),
            shape=(2,),
            dtype=np.float64
        )

        self.block_cog = block_cog
        self.damping = damping
        self.render_action = render_action

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.screen = None

        self.space = None
        self.teleop = None
        self.render_buffer = None
        self.latest_action = None
        self.reset_to_state = reset_to_state

    def reset(self):
        seed = self._seed
        self._setup()
        if self.block_cog is not None:
            self.block.center_of_gravity = self.block_cog
        if self.damping is not None:
            self.space.damping = self.damping

        # use legacy RandomState for compatiblity
        state = self.reset_to_state
        if state is None:
            rs = np.random.RandomState(seed=seed)
            state = np.array([
                rs.randint(50, 450), rs.randint(50, 450),
                rs.randint(100, 400), rs.randint(100, 400),
                rs.randn() * 2 * np.pi - np.pi
                ])
        self._set_state(state)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        dt = 1.0 / self.sim_hz
        self.n_contact_points = 0
        n_steps = self.sim_hz // self.control_hz
        if action is not None:
            self.latest_action = action
            for i in range(n_steps):
                # Step PD control.
                # self.agent.velocity = self.k_p * (act - self.agent.position)    # P control works too.
                acceleration = self.k_p * (action - self.agent.position) + self.k_v * (Vec2d(0, 0) - self.agent.velocity)
                self.agent.velocity += acceleration * dt

                # Step physics.
                self.space.step(dt)

        # compute reward
        goal_body = self._get_goal_pose_body(self.goal_pose)
        goal_geom = pymunk_to_shapely(goal_body, self.block.shapes)
        block_geom = pymunk_to_shapely(self.block, self.block.shapes)

        intersection_area = goal_geom.intersection(block_geom).area
        goal_area = goal_geom.area
        coverage = intersection_area / goal_area
        reward = np.clip(coverage / self.success_threshold, 0, 1)
        done = coverage > self.success_threshold
        terminated = done
        truncated = done

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self, mode):
        return self._render_frame(mode)

    def teleop_agent(self):
        TeleopAgent = collections.namedtuple('TeleopAgent', ['act'])
        def act(obs):
            act = None
            mouse_position = pymunk.pygame_util.from_pygame(Vec2d(*pygame.mouse.get_pos()), self.screen)
            if self.teleop or (mouse_position - self.agent.position).length < 30:
                self.teleop = True
                act = mouse_position
            return act
        return TeleopAgent(act)

    def _get_obs(self):
        obs = np.array(
            tuple(self.agent.position) \
            + tuple(self.block.position) \
            + (self.block.angle % (2 * np.pi),))
        return obs

    def _get_goal_pose_body(self, pose):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (50, 100))
        body = pymunk.Body(mass, inertia)
        # preserving the legacy assignment order for compatibility
        # the order here dosn't matter somehow, maybe because CoM is aligned with body origin
        body.position = pose[:2].tolist()
        body.angle = pose[2]
        return body

    def _get_info(self):
        n_steps = self.sim_hz // self.control_hz
        n_contact_points_per_step = int(np.ceil(self.n_contact_points / n_steps))
        info = {
            'pos_agent': np.array(self.agent.position),
            'vel_agent': np.array(self.agent.velocity),
            'block_pose': np.array(list(self.block.position) + [self.block.angle]),
            'goal_pose': self.goal_pose,
            'n_contacts': n_contact_points_per_step}
        return info

    def _render_frame(self, mode):

        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        self.screen = canvas

        draw_options = DrawOptions(canvas)

        # Draw goal pose.
        goal_body = self._get_goal_pose_body(self.goal_pose)
        for shape in self.block.shapes:
            goal_points = [pymunk.pygame_util.to_pygame(goal_body.local_to_world(v), draw_options.surface) for v in shape.get_vertices()]
            goal_points += [goal_points[0]]
            pygame.draw.polygon(canvas, self.goal_color, goal_points)

        # Draw agent and block.
        self.space.debug_draw(draw_options)

        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # the clock is aleady ticked during in step for "human"


        img = np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        img = cv2.resize(img, (self.render_size, self.render_size))
        if self.render_action:
            if self.render_action and (self.latest_action is not None):
                action = np.array(self.latest_action)
                coord = (action / 512 * 96).astype(np.int32)
                marker_size = int(8/96*self.render_size)
                thickness = int(1/96*self.render_size)
                cv2.drawMarker(img, coord,
                    color=(255,0,0), markerType=cv2.MARKER_CROSS,
                    markerSize=marker_size, thickness=thickness)
        return img


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0,25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

    def _handle_collision(self, arbiter, space, data):
        self.n_contact_points += len(arbiter.contact_point_set.points)

    def _set_state(self, state):
        if isinstance(state, np.ndarray):
            state = state.tolist()
        pos_agent = state[:2]
        pos_block = state[2:4]
        rot_block = state[4]
        self.agent.position = pos_agent
        # setting angle rotates with respect to center of mass
        # therefore will modify the geometric position
        # if not the same as CoM
        # therefore should be modified first.
        if self.legacy:
            # for compatiblity with legacy data
            self.block.position = pos_block
            self.block.angle = rot_block
        else:
            self.block.angle = rot_block
            self.block.position = pos_block

        # Run physics to take effect
        self.space.step(1.0 / self.sim_hz)

    def _set_state_local(self, state_local):
        agent_pos_local = state_local[:2]
        block_pose_local = state_local[2:]
        tf_img_obj = st.AffineTransform(
            translation=self.goal_pose[:2],
            rotation=self.goal_pose[2])
        tf_obj_new = st.AffineTransform(
            translation=block_pose_local[:2],
            rotation=block_pose_local[2]
        )
        tf_img_new = st.AffineTransform(
            matrix=tf_img_obj.params @ tf_obj_new.params
        )
        agent_pos_new = tf_img_new(agent_pos_local)
        new_state = np.array(
            list(agent_pos_new[0]) + list(tf_img_new.translation) \
                + [tf_img_new.rotation])
        self._set_state(new_state)
        return new_state

    def _setup(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = 0
        self.teleop = False
        self.render_buffer = list()

        # Add walls.
        walls = [
            self._add_segment((5, 506), (5, 5), 2),
            self._add_segment((5, 5), (506, 5), 2),
            self._add_segment((506, 5), (506, 506), 2),
            self._add_segment((5, 506), (506, 506), 2)
        ]
        self.space.add(*walls)

        # Add agent, block, and goal zone.
        self.agent = self.add_circle((256, 400), 15)
        self.block = self.add_tee((256, 300), 0)
        self.goal_color = pygame.Color('LightGreen')
        self.goal_pose = np.array([256,256,np.pi/4])  # x, y, theta (in radians)

        # Add collision handeling
        self.collision_handeler = self.space.add_collision_handler(0, 0)
        self.collision_handeler.post_solve = self._handle_collision
        self.n_contact_points = 0

        self.max_score = 50 * 100
        self.success_threshold = 0.95    # 95% coverage.

    def _add_segment(self, a, b, radius):
        shape = pymunk.Segment(self.space.static_body, a, b, radius)
        shape.color = pygame.Color('LightGray')    # https://htmlcolorcodes.com/color-names
        return shape

    def add_circle(self, position, radius):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color('RoyalBlue')
        self.space.add(body, shape)
        return body

    def add_box(self, position, height, width):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (height, width))
        body = pymunk.Body(mass, inertia)
        body.position = position
        shape = pymunk.Poly.create_box(body, (height, width))
        shape.color = pygame.Color('LightSlateGray')
        self.space.add(body, shape)
        return body

    def add_tee(self, position, angle, scale=30, color='LightSlateGray', mask=pymunk.ShapeFilter.ALL_MASKS()):
        mass = 1
        length = 4
        vertices1 = [(-length*scale/2, scale),
                                 ( length*scale/2, scale),
                                 ( length*scale/2, 0),
                                 (-length*scale/2, 0)]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        vertices2 = [(-scale/2, scale),
                                 (-scale/2, length*scale),
                                 ( scale/2, length*scale),
                                 ( scale/2, scale)]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices1)
        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (shape1.center_of_gravity + shape2.center_of_gravity) / 2
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1, shape2)
        return body


class PushTImageEnv(PushTEnv):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self,
            legacy=False,
            block_cog=None,
            damping=None,
            render_size=96):
        super().__init__(
            legacy=legacy,
            block_cog=block_cog,
            damping=damping,
            render_size=render_size,
            render_action=False)
        ws = self.window_size
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=1,
                shape=(3,render_size,render_size),
                dtype=np.float32
            ),
            'agent_pos': spaces.Box(
                low=0,
                high=ws,
                shape=(2,),
                dtype=np.float32
            )
        })
        self.render_cache = None

    def _get_obs(self):
        img = super()._render_frame(mode='rgb_array')

        agent_pos = np.array(self.agent.position)
        img_obs = np.moveaxis(img.astype(np.float32) / 255, -1, 0)
        obs = {
            'image': img_obs,
            'agent_pos': agent_pos
        }

        # draw action
        if self.latest_action is not None:
            action = np.array(self.latest_action)
            coord = (action / 512 * 96).astype(np.int32)
            marker_size = int(8/96*self.render_size)
            thickness = int(1/96*self.render_size)
            cv2.drawMarker(img, coord,
                color=(255,0,0), markerType=cv2.MARKER_CROSS,
                markerSize=marker_size, thickness=thickness)
        self.render_cache = img

        return obs

    def render(self, mode):
        assert mode == 'rgb_array'

        if self.render_cache is None:
            self._get_obs()

        return self.render_cache

#@markdown ### **Env Demo**
#@markdown Standard Gym Env (0.21.0 API)

# 0. create env object
env = PushTImageEnv()

# 1. seed env for initial state.
# Seed 0-200 are used for the demonstration dataset.
env.seed(1000)

# 2. must reset before use
obs, info = env.reset()

# 3. 2D positional action space [0,512]
action = env.action_space.sample()

# 4. Standard gym step method
obs, reward, terminated, truncated, info = env.step(action)

# prints and explains each dimension of the observation and action vectors
with np.printoptions(precision=4, suppress=True, threshold=5):
    print("obs['image'].shape:", obs['image'].shape, "float32, [0,1]")
    print("obs['agent_pos'].shape:", obs['agent_pos'].shape, "float32, [0,512]")
    print("action.shape: ", action.shape, "float32, [0,512]")



#@markdown ### **Dataset**
#@markdown
#@markdown Defines `PushTImageDataset` and helper functions
#@markdown
#@markdown The dataset class
#@markdown - Load data ((image, agent_pos), action) from a zarr storage
#@markdown - Normalizes each dimension of agent_pos and action to [-1,1]
#@markdown - Returns
#@markdown  - All possible segments with length `pred_horizon`
#@markdown  - Pads the beginning and the end of each episode with repetition
#@markdown  - key `image`: shape (obs_hoirzon, 3, 96, 96)
#@markdown  - key `agent_pos`: shape (obs_hoirzon, 2)
#@markdown  - key `action`: shape (pred_horizon, 2)

def create_sample_indices(
        episode_ends:np.ndarray, sequence_length:int,
        pad_before: int=0, pad_after: int=0):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result

# normalize data
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

# dataset
class PushTImageDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str,
                 pred_horizon: int,
                 obs_horizon: int,
                 action_horizon: int):

        # read from zarr dataset
        dataset_root = zarr.open(dataset_path, 'r')

        # float32, [0,1], (N,96,96,3)
        train_image_data = dataset_root['data']['img'][:]
        train_image_data = np.moveaxis(train_image_data, -1,1)
        # (N,3,96,96)

        # (N, D)
        train_data = {
            # first two dims of state vector are agent (i.e. gripper) locations
            'agent_pos': dataset_root['data']['state'][:,:2],
            'action': dataset_root['data']['action'][:]
        }
        episode_ends = dataset_root['meta']['episode_ends'][:]

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon-1,
            pad_after=action_horizon-1)

        # compute statistics and normalized data to [-1,1]
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])

        # images are already normalized
        normalized_train_data['image'] = train_image_data

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, \
            sample_start_idx, sample_end_idx = self.indices[idx]

        # get nomralized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        # discard unused observations
        nsample['image'] = nsample['image'][:self.obs_horizon,:]
        nsample['agent_pos'] = nsample['agent_pos'][:self.obs_horizon,:]
        return nsample
    
#@markdown ### **Dataset**


# download demonstration data from Google Drive
dataset_path = "pusht_cchi_v7_replay.zarr.zip"
if not os.path.isfile(dataset_path):
    id = "1KY1InLurpMvJDRb14L9NlXT_fEsCvVUq&confirm=t"
    gdown.download(id=id, output=dataset_path, quiet=False)

# parameters, DONT CHANGE!
pred_horizon = 16
obs_horizon = 2
action_horizon = 8
#|o|o|                             observations: 2
#| |a|a|a|a|a|a|a|a|               actions executed: 8
#|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

# create dataset from file
dataset = PushTImageDataset(
    dataset_path=dataset_path,
    pred_horizon=pred_horizon,
    obs_horizon=obs_horizon,
    action_horizon=action_horizon
)
# save training data statistics (min, max) for each dim
stats = dataset.stats

# create dataloader
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=64,
    num_workers=0,
    shuffle=True,
    # accelerate cpu-gpu transfer
    pin_memory=True,
    # don't kill worker process afte each epoch
    persistent_workers=False
)


#@markdown ### **Vision Encoder**
#@markdown
#@markdown Defines helper functions:
#@markdown - `get_resnet` to initialize standard ResNet vision encoder
#@markdown - `replace_bn_with_gn` to replace all BatchNorm layers with GroupNorm

def get_resnet(name:str, weights=None, **kwargs) -> nn.Module:
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", None
    """
    # Use standard ResNet implementation from torchvision
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)

    # remove the final fully connected layer
    # for resnet18, the output dim should be 512
    resnet.fc = torch.nn.Identity()
    return resnet


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module


# leaves these as is
# ResNet18 has output dim of 512
vision_feature_dim = 512
# agent_pos is 2 dimensional
lowdim_obs_dim = 2
# observation feature has 514 dims in total per step
obs_dim = vision_feature_dim + lowdim_obs_dim
action_dim = 2

############################
### IMPORTANT Parameters ###
############################

# change parameters here
infSteps = 1                # steps used during inference
infRuns = 1000              # how many inference runs?
load_model = True           # False will go ahead and train the model, if TRUE you should make sure, that the model_name exist in the model_folder
useNoisePred = False         # use noise prediction or sample prediction. IMPORTANT for training and inference

model_folder = "models"         # folder with models
model_name = 'pt_ema_student_1_stepspush_t.pth'  # model name in folder model_folder, if pretrained == False, a model with this name will be trained and saved into the models folder


###############################'
### IMPORTANT Parameters END ###
###############################'

model_path = f"{model_folder}/{model_name}"


# Download pretrained models

# Check if the folder exists
if not os.path.exists(model_folder):
    # Create the folder
    os.makedirs(model_folder)
    print(f"Folder '{model_folder}' created.")
else:
    print(f"Folder '{model_folder}' already exists.")



pt_noise_model_path = "models/pt_noise_model.pth"
if not os.path.isfile(pt_noise_model_path):
    print("Downloading pretrained Noise Model")
    id = "1EjTvhFJ9CQgs1DH0ShIHQ3MP8Ub3bd0Z&confirm=t"
    gdown.download(id=id, output=pt_noise_model_path, quiet=False)

pt_ema_student_1_stepspush_t_model_path = "models/pt_ema_student_1_stepspush_t.pth"
if not os.path.isfile(pt_ema_student_1_stepspush_t_model_path):
    print("Downloading pretrained destilled Model")
    id = "173aH8mG5gznfuPcFN8ST0jQbontfqQHD&confirm=t"
    gdown.download(id=id, output=pt_ema_student_1_stepspush_t_model_path, quiet=False)




def train(
    device = device,
    model_path = model_path,
    num_epochs = 100,
    useNoisePred = useNoisePred
    ):

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=100,
        # the choise of beta schedule has big impact on performance
        # we found squared cosine works the best
        beta_schedule='squaredcos_cap_v2',
        # clip output to [-1,1] to improve stability
        clip_sample=True,
        # our network predicts noise (instead of denoised action)
        prediction_type = 'epsilon' if useNoisePred else 'sample'
    )

    print("Training model for noise pred." if useNoisePred else "Training model for samples.")
    
    pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon
    ) if useNoisePred else ConditionalUnet1D_sample(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon
    ) 

    vision_encoder = replace_bn_with_gn(get_resnet('resnet18'))
    # the final arch has 2 parts
    nets = nn.ModuleDict({
        'vision_encoder': vision_encoder,
        'noise_pred_net': pred_net
    })     

    # device transfer
    
    _ = nets.to(device)

    #@markdown ### **Training**

    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(
        parameters=nets.parameters(),
        power=0.75)

    # Standard ADAM optimizer
    # Note that EMA parametesr are not optimized
    optimizer = torch.optim.AdamW(
        params=nets.parameters(),
        lr=1e-4, weight_decay=1e-6)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * num_epochs
    )

    final_loss = 0
    # Training loop
    with tqdm(range(num_epochs), desc='Epoch') as tglobal:
        for epoch_idx in tglobal:
            epoch_loss = list()
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    # Device transfer
                    nimage = nbatch['image'][:, :obs_horizon].to(device)
                    nagent_pos = nbatch['agent_pos'][:, :obs_horizon].to(device)
                    naction = nbatch['action'].to(device)
                    B = nagent_pos.shape[0]

                    # Normalize images
                    nimage = nimage.float() / 255.0
                    image_features = nets['vision_encoder'](nimage.flatten(end_dim=1))
                    image_features = image_features.reshape(*nimage.shape[:2], -1)

                    # Concatenate vision feature and low-dim observations
                    obs_features = torch.cat([image_features, nagent_pos], dim=-1)
                    obs_cond = obs_features.flatten(start_dim=1)

                    # Sample noise and timesteps
                    noise = torch.randn(naction.shape, device=device)
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (B,), device=device
                    ).long()

                    # Add noise to actions (forward diffusion)
                    noisy_actions = noise_scheduler.add_noise(  naction, noise, timesteps)

                    # Predict the noise sample
                    noisy = pred_net(noisy_actions, timesteps, global_cond=obs_cond)

                    # Compute L2 loss
                    loss = nn.functional.mse_loss(noisy, naction)

                    # Backpropagate and optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    # Step lr scheduler
                    lr_scheduler.step()

                    # Update EMA model
                    ema.step(nets.parameters())

                    # Logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)

            tglobal.set_postfix(loss=np.mean(epoch_loss))
            final_loss = np.mean(epoch_loss)

    # Copy EMA weights to the model
    ema_nets = nets
    ema.copy_to(ema_nets.parameters())

    # Save the trained model
    print(f"Saving model...\n"            
                f'  current_mean_loss: {final_loss}\n')
            
    #torch.save(ema_nets.state_dict(), )
    
    torch.save({
        'model_state_dict': ema_nets.state_dict(),
        'training_info': {        
            'current_mean_loss': final_loss,
            'prediction_type': "noise" if useNoisePred else "sample" 
        }
    }, model_path)
    print(f"Model saved to {model_path}.")
    return(ema_nets)


def load_model_with_info(model_path,            # file_path: Path to the .pth file containing the model and training info.
                         device,                # The device to load the model onto.
                         lowdim_obs_dim,        # low dim obs of the model
                         vision_feature_dim,    # Vision feature dimension
                         obs_horizon,           # Observation horizon
                         action_dim,            # dim of actions
                         useNoisePred           # use noise Prediction Network or Sample Prediction Network?
                         ):
    """
    Load a model and optionally display training information.
            
    Returns:
        nets: Loaded model.
    """
    # construct ResNet18 encoder
   
    vision_encoder = replace_bn_with_gn(get_resnet('resnet18'))

    # observation feature has 514 dims in total per step
    obs_dim = vision_feature_dim + lowdim_obs_dim

    
    
    # the final arch has 2 parts
    nets = nn.ModuleDict({
        'vision_encoder': vision_encoder,
        'noise_pred_net': ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon
    )
    }) if useNoisePred else nn.ModuleDict({
        'vision_encoder': vision_encoder,
        'noise_pred_net': ConditionalUnet1D_sample(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon
    )
    })
          
    # Load model
    if os.path.exists(model_path):
        print(f"model_path exists!")
        #model_checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        checkpoint = torch.load(model_path, weights_only=False, map_location=device)        
    
        # Check if training info exists
        if 'training_info' in checkpoint:
            training_info = checkpoint['training_info']
            print("Training Information:")
            for key, value in training_info.items():
                print(f"  {key}: {value}")

            nets.load_state_dict(checkpoint['model_state_dict'], strict=False)
            nets = nets.to(device) # map nets to correct device!   
            return nets
        else:
            print("No training information found in checkpoint.")

            print("Loading nets from modelpath...")
            nets.load_state_dict(checkpoint, strict=False)
            nets = nets.to(device) # map nets to correct device!    
                        
            print("Model loaded successfully.")
            
            return nets
    else:
        print("model_path invalid!")
        return 0

#@markdown ### **Loading Models if load_model = True, else it will be trained **

nets = load_model_with_info(model_path,            # file_path: Path to the .pth file containing the model and training info.
                            device,                # The device to load the model onto.
                            lowdim_obs_dim,        # low dim obs of the model
                            vision_feature_dim,    # Vision feature dimension
                            obs_horizon,           # Observation horizon
                            action_dim,            # dim of actions
                            useNoisePred           # use noise Prediction Network or Sample Prediction Network?
                            ) if load_model else train(
            device = device,
            model_path = model_path,
            num_epochs = 100,
            useNoisePred = useNoisePred
        )

#@markdown ### **Inference**
def inference(                             
        loaded_ema_nets,
        useNoisePred,
        model_name, 
        max_steps = 200,                    # limit enviornment interaction to 200 steps before termination
        seed = 100000,                      # use a seed >200 to avoid initial states seen in the training dataset
        set_num_diffusion_iters = 100,
        saveVideo = False,
        verbose = False):

    
    """
    inference Loop used for evaluating the models performance on the push t task
    
    Args:
    - max_steps: limit enviornment interaction to 200 steps before termination
    - seed: set env seed
    - loaded_ema_nets: ema_nets used for predictions
    - model_name: name of the used model nets -> determines the video output name / score output name
    """
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=100,
        # the choise of beta schedule has big impact on performance
        # we found squared cosine works the best
        beta_schedule='squaredcos_cap_v2',
        # clip output to [-1,1] to improve stability
        clip_sample=True,
        # our network predicts noise (instead of denoised action)
        prediction_type = 'epsilon' if useNoisePred else 'sample'
    )

    env = PushTImageEnv()    
    env.seed(seed)

    # get first observation
    obs, info = env.reset()

    # keep a queue of last 2 steps of observations
    obs_deque = collections.deque(
        [obs] * obs_horizon, maxlen=obs_horizon)
    # save visualization and rewards
    imgs = [env.render(mode='rgb_array')]
    rewards = list()
    done = False
    step_idx = 0 
    start_time = time.time()
    

    with tqdm(total=max_steps, desc="Eval PushTImageEnv") as pbar:
        while not done:
            B = 1
            # stack the last obs_horizon number of observations
            images = np.stack([x['image'] for x in obs_deque])
            agent_poses = np.stack([x['agent_pos'] for x in obs_deque])

            # normalize observation
            nagent_poses = normalize_data(agent_poses, stats=stats['agent_pos'])
            # images are already normalized to [0,1]
            nimages = images

            # device transfer
            nimages = torch.from_numpy(nimages).to(device, dtype=torch.float32)
            # (2,3,96,96)
            nagent_poses = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32)
            # (2,2)
            
            # infer action
            with torch.no_grad():                                                 
                # get image features
                image_features = loaded_ema_nets['vision_encoder'](nimages)
                # (2,512)

                # concat with low-dim observations
                obs_features = torch.cat([image_features, nagent_poses], dim=-1)

                # reshape observation to (B,obs_horizon*obs_dim)
                obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

                # initialize action from Guassian noise
                noisy_action = torch.randn( (B, pred_horizon, action_dim), device=device)

                # init scheduler
                noise_scheduler.set_timesteps(set_num_diffusion_iters)

                for k in noise_scheduler.timesteps:
                    # predict
                   
                    noise_pred = loaded_ema_nets['noise_pred_net'](
                        sample=  noisy_action,
                        timestep= k,
                        global_cond=obs_cond
                    ) if useNoisePred else loaded_ema_nets['noise_pred_net'](
                        noisy_sample=  noisy_action,
                        timestep= k,
                        global_cond=obs_cond
                    ) 
                            
                    noisy_action = noise_scheduler.step(
                        model_output=noise_pred,
                        timestep= k,
                        sample=noisy_action
                    ).prev_sample   

            # unnormalize action
            noisy_action = noisy_action.detach().to('cpu').numpy()
            # (B, pred_horizon, action_dim)
            noisy_action = noisy_action[0]
            action_pred = unnormalize_data(noisy_action, stats=stats['action'])

            # only take action_horizon number of actions
            start = obs_horizon - 1
            end = start + action_horizon
            action = action_pred[start:end,:]
            # (action_horizon, action_dim)

            # execute action_horizon number of steps
            # without replanning
            for i in range(len(action)):
                # stepping env
                obs, reward, done, _, info = env.step(action[i])
                # save observations
                obs_deque.append(obs)
                # and reward/vis
                rewards.append(reward)
                imgs.append(env.render(mode='rgb_array'))

                # update progress bar
                step_idx += 1
                pbar.update(1)
                pbar.set_postfix(reward=reward)
                if step_idx > max_steps:
                    done = True
                if done:
                    break

    # print out the maximum target coverage
    
    score = max(rewards)
    inferenceTime = (time.time() - start_time)
    if verbose:
        print(f'{model_name}')
        print(f' - Inference Time: {inferenceTime:.4f}')
        print(f' - Score: {score:.4f}')

    if saveVideo:

        # Check image format and convert if necessary
        def prepare_image_for_video(img):
            # If image is in float or normalized format
            if img.dtype == np.float32 or img.dtype == np.float64:
                img = (img * 255).astype(np.uint8)
            
            # Ensure correct color channels
            if img.shape[-1] == 1:  # grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[-1] == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            
            # Resize if necessary
            img = cv2.resize(img, (256, 256))
            
            return img

        # Modified video writing
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f'vis_{model_name}.mp4', fourcc, 20.0, (256, 256))

        for img in imgs:
            processed_img = prepare_image_for_video(img)
            out.write(processed_img)

        out.release()
    
    return([score, inferenceTime])


def run_multiple_inference(n_runs, set_num_diffusion_iters, max_steps=200, seed=100000, loaded_ema_nets=None, model_name=None, useNoisePred = useNoisePred):
    """
    Runs the inference function multiple times and stores results in numpy arrays.
    
    Parameters:
    n_runs (int): Number of inference runs to perform
    set_num_diffusion_iters (int): Number of diffusion iterations to use
    max_steps (int): Maximum number of environment steps (default: 200)
    seed (int): Random seed (default: 100000)
    loaded_ema_nets: Loaded network models
    model_name (str): Path to the model
    noise_scheduler: Noise scheduler for inference
    
    
    saves [scores_array, times_array] 
    """
    print(f"Running inference on model: {model_name} for: {n_runs} runs...") 
    # Initialize arrays to store results
    infData = []
    
    # Run inference n times
    for i in range(n_runs):
        # Increment seed for each run to ensure different initial states
        current_seed = seed + i
        
        # Run inference and store results
        score, inf_time = inference(
            max_steps=max_steps,
            seed=current_seed,
            loaded_ema_nets=loaded_ema_nets,
            model_name=model_name,
            set_num_diffusion_iters=set_num_diffusion_iters,
            useNoisePred = useNoisePred
        )
        
        infData.append((score,inf_time))
        # Calculate and print the mean score so far
        mean_score = np.mean([data[0] for data in infData])
        mean_time = np.mean([data[1] for data in infData])
        print(f"Mean score after {i+1}/{n_runs} runs: {mean_score:.4f} - Mean EpochTime [s]: {mean_time:.4f}")
    # save infData
    dt = np.dtype([
        ('score', np.float32),
        ('inf_time', np.float32)  
    ]) 
    inference_array = np.array(infData, dtype=dt)  
            
    # save inference data                
    np.save(f'infData/{"".join(e for e in model_name if e.isalnum())}_infData_runs_{n_runs}_steps_{set_num_diffusion_iters}.npy', inference_array) 
    
# RUN INFERENCE MULTIPLE TIMES

inference(
            loaded_ema_nets=nets,
            model_name=model_name,
            set_num_diffusion_iters=1,
            useNoisePred = useNoisePred,
            max_steps=200,
            seed=100000,            
            saveVideo = True,
            verbose = True
        )
# or run multiple times :)
'''
run_multiple_inference(
    n_runs=infRuns,
    set_num_diffusion_iters=infSteps,
    loaded_ema_nets=nets,
    model_name=model_path,
    useNoisePred = useNoisePred
)
'''

# Progressive Destillation
# if needed, run below code to progressively destill the model into faster submodels

def progressive_distillation_training(
    dataloader, 
    trained_nets, 
    checkpoint_saving_dir, 
    num_epochs, 
    initial_steps, 
    device, 
    obs_horizon = 2, 
    epoch_loss_threshold = 0, 
    useEma = False, 
    optimizer_weight_decay = 1e-6,
    loss_weight_balance = 0.5, # how are the loss terms weighted? default to 50_50
    model_name = ""
):
    """
    Progressive Distillation Algorithm for training a student model with fewer diffusion steps.

    Args:
        dataloader: PyTorch DataLoader for training data.
        nets: Dictionary of model components (e.g., vision_encoder, noise_pred_net).
        noise_scheduler: Noise scheduler (e.g., DDPMScheduler).
        useEma: save ema model or normal model?
        checkpoint_saving_dir: Directory to save checkpoints.
        num_epochs: Number of epochs per stage.
        initial_steps: Initial number of diffusion steps for the teacher model.
        device: Torch device for training.
        epoch_loss_threshold: if this loss is reached during one distillation epoch, then the next epoch starts! if left at 0, it wont stop early
        optimizer_weight_decay: weight_decay of optimizer # 1e-6 or maybe 0.1?
    """
    # Note that EMA parameter are not optimized
    

    os.makedirs(checkpoint_saving_dir, exist_ok=True)
    teacher_nets = trained_nets # assign pretrained nets
    teacher_steps = initial_steps
    student_steps = teacher_steps // 2 #  1
    # ema model
    ema_model = EMAModel(
        parameters=trained_nets.parameters(),
        power=0.75)
    # for this demo, we use DDPMScheduler with 100 diffusion iterations
    num_diffusion_iters = 100
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        # the choise of beta schedule has big impact on performance
        # we found squared cosine works the best
        beta_schedule='squaredcos_cap_v2',
        # clip output to [-1,1] to improve stability
        clip_sample=True,
        # our network predicts noise (instead of denoised action)
        prediction_type= 'sample' # sample for denoised sample
    )
    start_time = time.time()
    while student_steps >= 1:  # Progressive halving
        print(f"Distillation Stage: Teacher {teacher_steps} steps -> Student {student_steps} steps")
        # init student from teacher
        student_nets = deepcopy(teacher_nets) # added deepcopy
        opt = torch.optim.AdamW(
            params=student_nets.parameters(),
            lr=1e-4, 
            weight_decay = optimizer_weight_decay
        ) 
        
        distill_mean_loss = 0
        with tqdm(range(num_epochs), desc=f'Epoch (Steps {student_steps})') as tglobal:
            for epoch_idx in tglobal:
                epoch_loss = list()
                with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                    for nbatch in tepoch:
                        # Device transfer
                        nimage = nbatch['image'][:, :obs_horizon].to(device)
                        nagent_pos = nbatch['agent_pos'][:, :obs_horizon].to(device)
                        naction = nbatch['action'].to(device)
                        B = nagent_pos.shape[0]

                        # Normalize images
                        nimage = nimage.float() / 255.0
                        image_features = teacher_nets['vision_encoder'](nimage.flatten(end_dim=1))
                        image_features = image_features.reshape(*nimage.shape[:2], -1)

                        # Concatenate vision feature and low-dim observations
                        obs_features = torch.cat([image_features, nagent_pos], dim=-1)
                        obs_cond = obs_features.flatten(start_dim=1)
                        
                        # Sample noise and timesteps
                        noise = torch.randn(naction.shape, device=device)
                        timesteps = torch.randint(
                            0, teacher_steps,
                            (B,), device=device
                        ).long()
                        # Add noise to actions (forward diffusion)
                        noisy_actions = noise_scheduler.add_noise(
                            naction, noise, timesteps)

                        # Predict the noise sample
                        student_noisy_sample = student_nets['noise_pred_net']( noisy_actions, timesteps//2, global_cond=obs_cond)
                        teacher_noisy_sample = teacher_nets['noise_pred_net']( noisy_actions, timesteps, global_cond=obs_cond)

                        # Compute L2 loss
                        loss = loss_weight_balance * F.mse_loss(student_noisy_sample, naction) + (1-loss_weight_balance) * F.mse_loss(student_noisy_sample, teacher_noisy_sample)

                        # Backpropagate and optimize
                        loss.backward()
                        opt.step()
                        opt.zero_grad()
                        # lr_scheduler.step() maybe
                        # try with kl divergence loss                       
                        #kl_loss = nn.KLDivLoss(reduction='batchmean')   
                        ema_model.step(student_nets.parameters())

                        # Log and track progress
                        loss_cpu = loss.item()
                        epoch_loss.append(loss_cpu)
                        tepoch.set_postfix(loss=loss_cpu)
                    # early stop if loss is below threshold
                    distill_mean_loss = np.mean(epoch_loss)
                    if np.mean(epoch_loss) <= epoch_loss_threshold and np.mean(epoch_loss) != 0 and epoch_loss_threshold != 0:
                        print(f"\nStopping early at epoch {epoch_idx} with loss: {np.mean(epoch_loss):.5f} <= {epoch_loss_threshold}")
                        break  

                tglobal.set_postfix(loss=np.mean(epoch_loss))
                

        # Save checkpoints
        if useEma:            
            # move to EMA weights
            ema_nets = student_nets
            ema_model.copy_to(ema_nets.parameters())
            print(f"Saving ema model...\n"
                f'  training_time: {time.time() - start_time}\n'
                f'  current_mean_loss: {distill_mean_loss}\n',
                f'  current_inf_steps: {student_steps}\n')
            
            
            save_path = os.path.join(checkpoint_saving_dir, f'ema_student_{student_steps}_steps{model_name}.pth')
            torch.save({
                'model_state_dict': ema_nets.state_dict(),
                'training_info': {
                    'training_time': time.time() - start_time,
                    'current_mean_loss': distill_mean_loss,
                    'current_inf_steps': student_steps,
                }
            }, save_path)
        else:
            
            print(f"Saving normal model...\n"
                f'  training_time: {time.time() - start_time}\n'
                f'  current_mean_loss: {distill_mean_loss}\n',
                f'  current_inf_steps: {student_steps}\n')
            
            
            save_path = os.path.join(checkpoint_saving_dir, f'student_{student_steps}_steps{model_name}.pth')
            torch.save({
                'model_state_dict': student_nets.state_dict(),
                'training_info': {
                    'training_time': time.time() - start_time,
                    'current_mean_loss': distill_mean_loss,
                    'current_inf_steps': student_steps,
                }
            }, save_path)
        # assign teacher from student
        teacher_nets = deepcopy(student_nets) 
        # Halve steps for the next stage
        teacher_steps = student_steps
        student_steps //= 2


'''
progressive_distillation_training(
    trained_nets=nets,
    dataloader=dataloader,
    checkpoint_saving_dir=os.path.join(os.getcwd(), "models"), 
    num_epochs=15,
    initial_steps=100,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    epoch_loss_threshold=0, # 0.005
    useEma = True,
    optimizer_weight_decay = 1e-6, # 0.1 oder 1e-6, geringer für mehr Generalisierbarkeit
    loss_weight_balance = 0.2,
    model_name="push_t"
) 
'''
