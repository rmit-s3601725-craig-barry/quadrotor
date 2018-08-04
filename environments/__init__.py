from gym.envs.registration import register

register(
    id='Hover-v0',
    entry_point='environments.hover:Environment',
)