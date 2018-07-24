import gym
import numpy as np

class SnakeColor:

    def __init__(self, head_color, body_color):
        self.head_color = head_color
        self.body_color = body_color

class RGBifier:
    def __init__(self, size, zoom_factor=1, players_colors={}):

        self.p_colors = {1: SnakeColor((191, 242, 191), (0, 204, 0)),  # Green
                         2: SnakeColor((188, 128, 230), (119, 0, 204)),  # Violet
                         3: SnakeColor((128, 154, 230), (0, 51, 204)),  # Blue
                         4: SnakeColor((230, 128, 188), (204, 0, 119)),  # Magenta
                         10: SnakeColor((255, 255, 189), (255, 255, 0))}  # Opponent, orange
        self.zoom_factor = zoom_factor
        self.size = size
        self.height = size[0]
        self.width = size[1]

    def get_color(self, state):
        # VOID -> BLACK
        if state == 0:
            return [0, 0, 0]
        elif state == 255:
            return (255, 0, 0)
        elif state == 10:   # Wall
            return (0, 255, 0)
        else:
            print("state is ", state)
            snake_id = state // 2
            is_head = state % 2

            if snake_id not in self.p_colors.keys():
                snake_id = 0
            if is_head == 0:
                return self.p_colors[snake_id].body_color
            else:
                print("snake id is ", snake_id)
                return self.p_colors[snake_id].head_color

    def get_img_array(self, state):
        COLOR_CHANNELS = 3

        color_lu = np.vectorize(lambda x: self.get_color(x), otypes=[np.uint8, np.uint8, np.uint8])
        img = np.array(color_lu(state))
        print("img is ", img.shape)
        # Transpose to get channels as last
        img_zoomed = np.transpose(img, [1, 2, 0])
        return img_zoomed

    def get_image(self, state):
        screen_dim = 600
        cell_dim = (int) (screen_dim / self.size[0])
        # Transform to RGB image with 3 channels
        COLOR_CHANNELS = 3
        color_lu = np.vectorize(lambda x: self.get_color(x), otypes=[np.uint8, np.uint8, np.uint8])
        img = np.array(color_lu(state))
        # Zoom every channel
        img_zoomed = np.zeros((3, self.height * cell_dim, self.width * cell_dim), dtype=np.uint8)

        for c in range(COLOR_CHANNELS):
            for i in range(img.shape[1]):
                for j in range(img.shape[2]):
                    img_zoomed[c, i * cell_dim: (i + 1) * cell_dim,
                    j * cell_dim: (j + 1) * cell_dim] = np.full(
                        (cell_dim, cell_dim), img[c, i, j])
        # Transpose to get channels as last
        img_zoomed = np.transpose(img_zoomed, [1, 2, 0])
        return img_zoomed

class Renderer:

    def __init__(self, size, zoom_factor=1, players_colors = {}):
        self.rgb = RGBifier(size, zoom_factor, players_colors)
        self.viewer = None

    def render(self, state, mode='human', close=False):
        if close:
            self.close()
            return

        img = self.rgb.get_image(state)

        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None