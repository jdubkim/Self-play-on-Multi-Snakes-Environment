from gym.envs.registration import register
from gym.scoreboard.registration import add_group
from gym.scoreboard.registration import add_task

'''
Environment Settings
'''

def set_env():
    register(
        id='SnakeEnv-v0',
        entry_point='gym.envs.gym_snake:SnakeEnv',
    )

    add_group(
        id='gym_snake',
        name='gym_snake',
        description='snake'
    )

    add_task(
        id='SnakeEnv-v0',
        group='gym_snake',
        summary="Multi snakes environment"
    )

