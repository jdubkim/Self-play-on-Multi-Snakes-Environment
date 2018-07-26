class Config(object):
    USE_ATARI_SIZE = True
    MODEL_DIR = 'saved_models/'
    EXPR_NAME = 'ppo/'
    PRIMARY_MODEL_SCOPE = 'primary_model'
    OPPONENT_MODEL_SCOPE = 'opponent_model'
    OPPONENT_MODEL2_SCOPE = 'opponent_model2'

    OPPONENT_SAVE_INTERVAL = 50  # how frequently to save the trained agent, for later play
    MAX_SAVED_OPPONENTS = 1000  # the max number of agents in the pool of opponents

    def set_num_snakes(num_snakes):
        Config.NUM_SNAKES = num_snakes

    def set_directory(expr_dir):
        Config.EXPR_NAME = expr_dir