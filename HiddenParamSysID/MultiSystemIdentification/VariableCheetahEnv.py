import os
from datetime import datetime
from typing import Dict, Optional
import numpy as np
import gymnasium as gym

# default values specified as constants
DEFAULT_FRICTION = 0.4
DEFAULT_TORSO_LENGTH = 1
DEFAULT_BTHIGH_LENGTH = .145
DEFAULT_BSHIN_LENGTH = .15
DEFAULT_BFOOT_LENGTH = .094
DEFAULT_FTHIGH_LENGTH = .133
DEFAULT_FSHIN_LENGTH = .106
DEFAULT_FFOOT_LENGTH = .07
DEFAULT_BTHIGH_GEAR = 120
DEFAULT_BSHIN_GEAR = 90
DEFAULT_BFOOT_GEAR = 60
DEFAULT_FTHIGH_GEAR = 120
DEFAULT_FSHIN_GEAR = 60
DEFAULT_FFOOT_GEAR = 30


class VariableCheetahEnv(gym.Env):
    def __init__(self, dynamics_variable_ranges:Dict, *env_args, **env_kwargs):
        '''

        :param dynamics_variable_ranges: A dictionary of ranges for the dynamics variables. The following
        keys are optional. The default values are used if not specified. Provide as a tuple of (lower, upper).
        - friction
        - torso_length
        - bthigh_length
        - bshin_length
        - bfoot_length
        - fthigh_length
        - fshin_length
        - ffoot_length
        - bthigh_gear
        - bshin_gear
        - bfoot_gear
        - fthigh_gear
        - fshin_gear
        - ffoot_gear
        '''
        super().__init__()
        self.env_args = env_args
        self.env_kwargs = env_kwargs
        self.render_mode = env_kwargs.get('render_mode', None)
        self.dynamics_variable_ranges = dynamics_variable_ranges

        # append defaults if not specified
        if 'friction' not in self.dynamics_variable_ranges:
            self.dynamics_variable_ranges['friction'] = (DEFAULT_FRICTION, DEFAULT_FRICTION)
        if 'torso_length' not in self.dynamics_variable_ranges:
            self.dynamics_variable_ranges['torso_length'] = (DEFAULT_TORSO_LENGTH, DEFAULT_TORSO_LENGTH)
        if 'bthigh_length' not in self.dynamics_variable_ranges:
            self.dynamics_variable_ranges['bthigh_length'] = (DEFAULT_BTHIGH_LENGTH, DEFAULT_BTHIGH_LENGTH)
        if 'bshin_length' not in self.dynamics_variable_ranges:
            self.dynamics_variable_ranges['bshin_length'] = (DEFAULT_BSHIN_LENGTH, DEFAULT_BSHIN_LENGTH)
        if 'bfoot_length' not in self.dynamics_variable_ranges:
            self.dynamics_variable_ranges['bfoot_length'] = (DEFAULT_BFOOT_LENGTH, DEFAULT_BFOOT_LENGTH)
        if 'fthigh_length' not in self.dynamics_variable_ranges:
            self.dynamics_variable_ranges['fthigh_length'] = (DEFAULT_FTHIGH_LENGTH, DEFAULT_FTHIGH_LENGTH)
        if 'fshin_length' not in self.dynamics_variable_ranges:
            self.dynamics_variable_ranges['fshin_length'] = (DEFAULT_FSHIN_LENGTH, DEFAULT_FSHIN_LENGTH)
        if 'ffoot_length' not in self.dynamics_variable_ranges:
            self.dynamics_variable_ranges['ffoot_length'] = (DEFAULT_FFOOT_LENGTH, DEFAULT_FFOOT_LENGTH)
        if 'bthigh_gear' not in self.dynamics_variable_ranges:
            self.dynamics_variable_ranges['bthigh_gear'] = (DEFAULT_BTHIGH_GEAR, DEFAULT_BTHIGH_GEAR)
        if 'bshin_gear' not in self.dynamics_variable_ranges:
            self.dynamics_variable_ranges['bshin_gear'] = (DEFAULT_BSHIN_GEAR, DEFAULT_BSHIN_GEAR)
        if 'bfoot_gear' not in self.dynamics_variable_ranges:
            self.dynamics_variable_ranges['bfoot_gear'] = (DEFAULT_BFOOT_GEAR, DEFAULT_BFOOT_GEAR)
        if 'fthigh_gear' not in self.dynamics_variable_ranges:
            self.dynamics_variable_ranges['fthigh_gear'] = (DEFAULT_FTHIGH_GEAR, DEFAULT_FTHIGH_GEAR)
        if 'fshin_gear' not in self.dynamics_variable_ranges:
            self.dynamics_variable_ranges['fshin_gear'] = (DEFAULT_FSHIN_GEAR, DEFAULT_FSHIN_GEAR)
        if 'ffoot_gear' not in self.dynamics_variable_ranges:
            self.dynamics_variable_ranges['ffoot_gear'] = (DEFAULT_FFOOT_GEAR, DEFAULT_FFOOT_GEAR)

        # placeholder variable
        self.env =  gym.make('HalfCheetah-v3', *self.env_args, **self.env_kwargs)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        # path to write xml to
        current_date_time = datetime.now().strftime("%Y%m%d%H%M%S")
        self.xml_path = '/tmp/half_cheetahs'
        self.xml_name =  f'{current_date_time}.xml'
        os.makedirs(self.xml_path, exist_ok=True)

    def reset(self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        # sample env parameters
        friction = np.random.uniform(self.dynamics_variable_ranges['friction'][0], self.dynamics_variable_ranges['friction'][1])
        torso_length = np.random.uniform(self.dynamics_variable_ranges['torso_length'][0], self.dynamics_variable_ranges['torso_length'][1])
        bthigh_length = np.random.uniform(self.dynamics_variable_ranges['bthigh_length'][0], self.dynamics_variable_ranges['bthigh_length'][1])
        bshin_length = np.random.uniform(self.dynamics_variable_ranges['bshin_length'][0], self.dynamics_variable_ranges['bshin_length'][1])
        bfoot_length = np.random.uniform(self.dynamics_variable_ranges['bfoot_length'][0], self.dynamics_variable_ranges['bfoot_length'][1])
        fthigh_length = np.random.uniform(self.dynamics_variable_ranges['fthigh_length'][0], self.dynamics_variable_ranges['fthigh_length'][1])
        fshin_length = np.random.uniform(self.dynamics_variable_ranges['fshin_length'][0], self.dynamics_variable_ranges['fshin_length'][1])
        ffoot_length = np.random.uniform(self.dynamics_variable_ranges['ffoot_length'][0], self.dynamics_variable_ranges['ffoot_length'][1])
        bthigh_gear = np.random.uniform(self.dynamics_variable_ranges['bthigh_gear'][0], self.dynamics_variable_ranges['bthigh_gear'][1])
        bshin_gear = np.random.uniform(self.dynamics_variable_ranges['bshin_gear'][0], self.dynamics_variable_ranges['bshin_gear'][1])
        bfoot_gear = np.random.uniform(self.dynamics_variable_ranges['bfoot_gear'][0], self.dynamics_variable_ranges['bfoot_gear'][1])
        fthigh_gear = np.random.uniform(self.dynamics_variable_ranges['fthigh_gear'][0], self.dynamics_variable_ranges['fthigh_gear'][1])
        fshin_gear = np.random.uniform(self.dynamics_variable_ranges['fshin_gear'][0], self.dynamics_variable_ranges['fshin_gear'][1])
        ffoot_gear = np.random.uniform(self.dynamics_variable_ranges['ffoot_gear'][0], self.dynamics_variable_ranges['ffoot_gear'][1])

        # create xml file for these parameters
        path = self.create_xml_file(friction, torso_length, bthigh_length, bshin_length, bfoot_length, fthigh_length, fshin_length, ffoot_length, bthigh_gear, bshin_gear, bfoot_gear, fthigh_gear, fshin_gear, ffoot_gear)

        # load env with this xml file
        self.env = gym.make('HalfCheetah-v3', xml_file=path, *self.env_args, **self.env_kwargs)

        # return observation
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    # generates a custom file for these parameters and writes it to tmp. Returns a path.
    # Note that the constants in this file are the defaults, which  I have rescaled based on the lengths specified.
    def create_xml_file(self, friction, torso_length, bthigh_length, bshin_length, bfoot_length, fthigh_length, fshin_length, ffoot_length, bthigh_gear, bshin_gear, bfoot_gear, fthigh_gear, fshin_gear, ffoot_gear):
        file_string = f'''<!-- Cheetah Model

    The state space is populated with joints in the order that they are
    defined in this file. The actuators also operate on joints.

    State-Space (name/joint/parameter):
        - rootx     slider      position (m)
        - rootz     slider      position (m)
        - rooty     hinge       angle (rad)
        - bthigh    hinge       angle (rad)
        - bshin     hinge       angle (rad)
        - bfoot     hinge       angle (rad)
        - fthigh    hinge       angle (rad)
        - fshin     hinge       angle (rad)
        - ffoot     hinge       angle (rad)
        - rootx     slider      velocity (m/s)
        - rootz     slider      velocity (m/s)
        - rooty     hinge       angular velocity (rad/s)
        - bthigh    hinge       angular velocity (rad/s)
        - bshin     hinge       angular velocity (rad/s)
        - bfoot     hinge       angular velocity (rad/s)
        - fthigh    hinge       angular velocity (rad/s)
        - fshin     hinge       angular velocity (rad/s)
        - ffoot     hinge       angular velocity (rad/s)

    Actuators (name/actuator/parameter):
        - bthigh    hinge       torque (N m)
        - bshin     hinge       torque (N m)
        - bfoot     hinge       torque (N m)
        - fthigh    hinge       torque (N m)
        - fshin     hinge       torque (N m)
        - ffoot     hinge       torque (N m)

-->
<mujoco model="cheetah">
  <compiler angle="radian" coordinate="local" inertiafromgeom="true" settotalmass="14"/>
  <default>
    <joint armature=".1" damping=".01" limited="true" solimplimit="0 .8 .03" solreflimit=".02 1" stiffness="8"/>
    <geom conaffinity="0" condim="3" contype="1" friction="{friction} .1 .1" rgba="0.8 0.6 .4 1" solimp="0.0 0.8 0.01" solref="0.02 1"/>
    <motor ctrllimited="true" ctrlrange="-1 1"/>
  </default>
  <size nstack="300000" nuser_geom="1"/>
  <option gravity="0 0 -9.81" timestep="0.01"/>
  <asset>
    <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="skybox" width="100"/>
    <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
    <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
    <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
    <material name="geom" texture="texgeom" texuniform="true"/>
  </asset>
  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom conaffinity="1" condim="3" material="MatPlane" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="40 40 40" type="plane"/>
    <body name="torso" pos="0 0 .7">
      <camera name="track" mode="trackcom" pos="0 -3 0.3" xyaxes="1 0 0 0 0 1"/>
      <joint armature="0" axis="1 0 0" damping="0" limited="false" name="rootx" pos="0 0 0" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 0 1" damping="0" limited="false" name="rootz" pos="0 0 0" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 1 0" damping="0" limited="false" name="rooty" pos="0 0 0" stiffness="0" type="hinge"/>
      <geom fromto="{-0.5 * torso_length/DEFAULT_TORSO_LENGTH} 0 0 {0.5 * torso_length/DEFAULT_TORSO_LENGTH} 0 0" name="torso" size="0.046" type="capsule"/>
      <geom axisangle="0 1 0 .87" name="head" pos="{0.5 * torso_length/DEFAULT_TORSO_LENGTH + 0.1} 0 .1" size="0.046 0.15" type="capsule"/>
      <!-- <site name='tip'  pos='.15 0 .11'/>-->
      <body name="bthigh" pos="{-0.5 * torso_length/DEFAULT_TORSO_LENGTH} 0 0">
        <joint axis="0 1 0" damping="6" name="bthigh" pos="0 0 0" range="-.52 1.05" stiffness="240" type="hinge"/>
        <geom axisangle="0 1 0 -3.8" name="bthigh" pos="{.1 * (bthigh_length/DEFAULT_BTHIGH_LENGTH)} 0 {-.13* (bthigh_length/DEFAULT_BTHIGH_LENGTH)}" size="0.046 {bthigh_length}" type="capsule"/>
        <body name="bshin" pos="{.16 * (bthigh_length/DEFAULT_BTHIGH_LENGTH)} 0 {-.25  * (bthigh_length/DEFAULT_BTHIGH_LENGTH)}">
          <joint axis="0 1 0" damping="4.5" name="bshin" pos="0 0 0" range="-.785 .785" stiffness="180" type="hinge"/>
          <geom axisangle="0 1 0 -2.03" name="bshin" pos="{-.14 * bshin_length/DEFAULT_BSHIN_LENGTH} 0 {-.07*bshin_length/DEFAULT_BSHIN_LENGTH}" rgba="0.9 0.6 0.6 1" size="0.046 {bshin_length}" type="capsule"/>
          <body name="bfoot" pos="{-.28 * bshin_length/DEFAULT_BSHIN_LENGTH} 0 {-.14 * bshin_length/DEFAULT_BSHIN_LENGTH}">
            <joint axis="0 1 0" damping="3" name="bfoot" pos="0 0 0" range="-.4 .785" stiffness="120" type="hinge"/>
            <geom axisangle="0 1 0 -.27" name="bfoot" pos="{.03 * bfoot_length / DEFAULT_BFOOT_LENGTH} 0 {-.097 * bfoot_length / DEFAULT_BFOOT_LENGTH}" rgba="0.9 0.6 0.6 1" size="0.046 {bfoot_length}" type="capsule"/>
          </body>
        </body>
      </body>
      <body name="fthigh" pos="{0.5 * torso_length/DEFAULT_TORSO_LENGTH} 0 0">
        <joint axis="0 1 0" damping="4.5" name="fthigh" pos="0 0 0" range="-1 .7" stiffness="180" type="hinge"/>
        <geom axisangle="0 1 0 .52" name="fthigh" pos="{-.07 * fthigh_length/DEFAULT_FTHIGH_LENGTH} 0 {-.12 * fthigh_length/DEFAULT_FTHIGH_LENGTH}" size="0.046 {fthigh_length}" type="capsule"/>
        <body name="fshin" pos="{-.14 * fthigh_length/DEFAULT_FTHIGH_LENGTH} 0 {-.24 * fthigh_length/DEFAULT_FTHIGH_LENGTH}">
          <joint axis="0 1 0" damping="3" name="fshin" pos="0 0 0" range="-1.2 .87" stiffness="120" type="hinge"/>
          <geom axisangle="0 1 0 -.6" name="fshin" pos="{.065 * fshin_length/DEFAULT_FSHIN_LENGTH} 0 {-.09 * fshin_length/DEFAULT_FSHIN_LENGTH}" rgba="0.9 0.6 0.6 1" size="0.046 {fshin_length}" type="capsule"/>
          <body name="ffoot" pos="{.13 * fshin_length/DEFAULT_FSHIN_LENGTH} 0 {-.18 * fshin_length/DEFAULT_FSHIN_LENGTH}">
            <joint axis="0 1 0" damping="1.5" name="ffoot" pos="0 0 0" range="-.5 .5" stiffness="60" type="hinge"/>
            <geom axisangle="0 1 0 -.6" name="ffoot" pos="{.045 * ffoot_length/DEFAULT_FFOOT_LENGTH} 0 {-.07 * ffoot_length/DEFAULT_FFOOT_LENGTH}" rgba="0.9 0.6 0.6 1" size="0.046 {ffoot_length}" type="capsule"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor gear="{bthigh_gear}" joint="bthigh" name="bthigh"/>
    <motor gear="{bshin_gear}" joint="bshin" name="bshin"/>
    <motor gear="{bfoot_gear}" joint="bfoot" name="bfoot"/>
    <motor gear="{fthigh_gear}" joint="fthigh" name="fthigh"/>
    <motor gear="{fshin_gear}" joint="fshin" name="fshin"/>
    <motor gear="{ffoot_gear}" joint="ffoot" name="ffoot"/>
  </actuator>
</mujoco>'''
        with open(os.path.join(self.xml_path, self.xml_name), 'w') as f:
            f.write(file_string)
        return os.path.join(self.xml_path, self.xml_name)






