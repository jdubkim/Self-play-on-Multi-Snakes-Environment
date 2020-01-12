import os

class Config(object):
    USE_ATARI_SIZE = True
    MODEL_DIR = 'saved_models/'
    EXPR_NAME = 'ppo/'
    EXPR_DIR = 'ppo/'
    PRIMARY_MODEL_SCOPE = 'primary_model'
    OPPONENT_MODEL_SCOPE = 'opponent_model' # TODO: Change it to list
    OPPONENT_MODEL2_SCOPE = 'opponent_model2'

    SCREEN_RESOLUTION = 300
    OPPONENT_SAVE_INTERVAL = 50  # how frequently to save the trained agent, for later play
    MAX_SAVED_OPPONENTS = 1000  # the max number of agents in the pool of opponents

    def set_num_snakes(num_snakes):
        Config.NUM_SNAKES = num_snakes

    def set_directory(expr_dir:str):
        if expr_dir is None:
            return

        Config.EXPR_NAME = expr_dir
        Config.EXPR_DIR = expr_dir + '/'
        child_dir = os.path.dirname(Config.MODEL_DIR + Config.EXPR_DIR)
        parent_dir = os.path.dirname(Config.MODEL_DIR)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        if not os.path.exists(child_dir):
            os.makedirs(child_dir)