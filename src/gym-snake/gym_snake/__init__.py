from gym.envs.registration import register

register(
  id='snake-single-v0',
  entry_point='gym_snake.envs:SingleSnake',
)
register(
  id='snake-multiple-v0',
  entry_point='gym_snake.envs:MultipleSnakes',
)
register(
  id='snake-competitive-v0',
  entry_point='gym_snake.envs:CompetitiveSnakes',
)
register(
  id='snake-multiple-test-v0',
  entry_point='gym_snake.envs:SnakeEnv',
)
