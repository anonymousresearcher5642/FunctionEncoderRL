# import my local env
from MultiSystemIdentification.VariableCheetahEnv import *

env = VariableCheetahEnv(dynamics_variable_ranges={ 'friction':(DEFAULT_FRICTION*0.5, DEFAULT_FRICTION*2),
                                                    'torso_length':(DEFAULT_TORSO_LENGTH * 0.5, DEFAULT_TORSO_LENGTH * 1.5),
                                                    'bthigh_length':(DEFAULT_BTHIGH_LENGTH * 0.5, DEFAULT_BTHIGH_LENGTH * 1.5),
                                                    'bshin_length':(DEFAULT_BSHIN_LENGTH * 0.5, DEFAULT_BSHIN_LENGTH * 1.5),
                                                    'bfoot_length':(DEFAULT_BFOOT_LENGTH * 0.5, DEFAULT_BFOOT_LENGTH * 1.5),
                                                    'fthigh_length':(DEFAULT_FTHIGH_LENGTH * 0.5, DEFAULT_FTHIGH_LENGTH * 1.5),
                                                    'fshin_length':(DEFAULT_FSHIN_LENGTH * 0.5, DEFAULT_FSHIN_LENGTH * 1.5),
                                                    'ffoot_length':(DEFAULT_FFOOT_LENGTH * 0.5, DEFAULT_FFOOT_LENGTH * 1.5),
                                                    'bthigh_gear':(DEFAULT_BTHIGH_GEAR * 0.0, DEFAULT_BTHIGH_GEAR * 2.0),
                                                    'bshin_gear':(DEFAULT_BSHIN_GEAR * 0.0, DEFAULT_BSHIN_GEAR * 2.0),
                                                    'bfoot_gear':(DEFAULT_BFOOT_GEAR * 0.0, DEFAULT_BFOOT_GEAR * 2.0),
                                                    'fthigh_gear':(DEFAULT_FTHIGH_GEAR * 0.0, DEFAULT_FTHIGH_GEAR * 2.0),
                                                    'fshin_gear':(DEFAULT_FSHIN_GEAR * 0.0, DEFAULT_FSHIN_GEAR * 2.0),
                                                    'ffoot_gear':(DEFAULT_FFOOT_GEAR * 0.0, DEFAULT_FFOOT_GEAR * 2.0),
                                                    },
                         render_mode='human')

for i in range(10):
    env.reset()
    for j in range(100):
        env.render()
        env.step(env.action_space.sample())
    env.close()

