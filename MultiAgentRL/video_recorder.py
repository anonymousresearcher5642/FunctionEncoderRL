from typing import Any

import numpy as np
from pettingzoo.utils import BaseWrapper
import cv2

# create a wrapper to record the video of a pettingzoo environment
class VideoRecorder(BaseWrapper):

    def __init__(self, env, path: str, fps: int = 30, width=700, height=700) -> None:
        # initialize the base class
        super(VideoRecorder, self).__init__(env)
        # initialize the video recorder
        self.frames = []
        self.path = path
        self.fps = fps
        self.episode = 0
        self.width = width
        self.height = height

    def reset(self) -> None | np.ndarray | str | list:
        # reset the environment
        # write the video
        # Create a VideoWriter object
        if len(self.frames):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
            path = f"{self.path}/episode_{self.episode}.mp4"
            out = cv2.VideoWriter(path, fourcc, self.fps, (self.width, self.height)) # 700x700 is the size of the frames from simpleenv

            # Write frames to the video
            for frame in self.frames:
                out.write(frame)

            # Release the VideoWriter
            out.release()
            print(f"Wrote episode {self.episode} to {path}")
            self.frames = []
            self.episode += 1
        return self.env.reset()


    def render(self) -> None | np.ndarray | str | list:
        # render the environment
        frame = self.env.render()
        # save the frame
        self.frames.append(frame)
        return None

    def __getattr__(self, name: str) -> Any:
        """Returns an attribute with ``name``, unless ``name`` starts with an underscore."""
        # if name.startswith("_"):
        #     raise AttributeError(f"accessing private attribute '{name}' is prohibited")
        return getattr(self.env, name)