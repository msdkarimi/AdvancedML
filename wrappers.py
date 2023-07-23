import os

import gym
import numpy as np
import cv2

from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback

from gym import spaces
from gym.spaces import Box

from collections import deque
from gym import ObservationWrapper


class PixelObservationWrapper(gym.ObservationWrapper):
    """Augment observations by pixel values."""

    def __init__(
        self, env, render_kwargs=None
    ):
        """Initializes a new pixel Wrapper.

        Args:
            env: The environment to wrap.
            render_kwargs: Optional `dict` containing keyword arguments passed
                to the `self.render` method.

        Raises:
            ValueError: If `env`'s observation spec is not compatible with the
                wrapper. Supported formats are a single array, or a dict of
                arrays.

        """


        super(PixelObservationWrapper, self).__init__(env)
        
        if render_kwargs is None:
            render_kwargs = {}

        render_kwargs.setdefault('pixels', {})
    
        render_mode = render_kwargs['pixels'].pop("mode", "rgb_array")
        assert render_mode == "rgb_array", render_mode
        render_kwargs['pixels']["mode"] = "rgb_array"
    
        pixels = self.env.render(**render_kwargs['pixels'])
        
        if np.issubdtype(pixels.dtype, np.integer):
            low, high = (0, 255)
        elif np.issubdtype(pixels.dtype, np.float):
            low, high = (-float("inf"), float("inf"))
        else:
            raise TypeError(pixels.dtype)
        
        
        self.observation_space = spaces.Box(
            shape=pixels.shape, low=low, high=high, dtype=pixels.dtype
        )

        self._env = env
        self._render_kwargs = render_kwargs

    def observation(self, observation):
        observation =self.env.render(**self._render_kwargs["pixels"])
        return observation


class GrayScaleWrapper(gym.ObservationWrapper):
    """Convert the image observation from RGB to gray scale.
    """

    def __init__(self, env: gym.Env, preprocessed: bool, keep_dim: bool = False, smooth: bool = False):
        """Convert the image observation from RGB to gray scale.

        Args:
            env (Env): The environment to apply the wrapper
            preprocessed : Whether we used preprocessing or not
            smooth : Whether to use smoothing filter or not
            keep_dim (bool): If `True`, a singleton dimension will be added, i.e. observations are of the shape AxBx1.
                Otherwise, they are of shape AxB.
        """
        super().__init__(env)
        self.keep_dim = keep_dim
        self.smooth = smooth
        self.preprocessed = preprocessed
        assert (
            isinstance(self.observation_space, Box)
            and len(self.observation_space.shape) == 3
            and self.observation_space.shape[-1] == 3
        )

        obs_shape = self.observation_space.shape[:2]
        
        if self.preprocessed:
            obs_shape = (obs_shape[0]-50, obs_shape[1]-50)
            
        if self.keep_dim:

            self.observation_space = Box(
                low=0, high=255, shape=(1, obs_shape[0], obs_shape[1]), dtype=np.uint8
            )
            
        else:
            self.observation_space = Box(
                low=0, high=255, shape=obs_shape, dtype=np.uint8
            )

    def observation(self, observation):
        """Converts the colour observation to greyscale.

        Args:
            observation: Color observations

        Returns:
            Grayscale observations
        """
        
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        if self.smooth:
            observation[ observation < 30 ] = 255
            kernel = np.ones((5,5),np.float32)/30
            observation = cv2.filter2D(observation,-1,kernel)

        

        if self.keep_dim:
            observation = np.expand_dims(observation, 0)
        
        return observation


class ResizeWrapper(gym.ObservationWrapper):
    """
    Resizes the gray observation. It should be 2-dimensional.

    """

    def __init__(self, env: gym.Env, shape=None):
        """Convert the image observation from RGB to gray scale.

        Args:
            env (Env): The environment to apply the wrapper
            shape : The shape we want the current image turns to.
        """
        super().__init__(env)
        self.shape = tuple(shape)
        assert (
            isinstance(self.observation_space, Box)
            and len(self.observation_space.shape) == len(self.shape)
        )


        self.observation_space = Box(
            low=0, high=255, shape=self.shape, dtype=np.uint8
        )


    def observation(self, observation):
        """
        Converts the colour observation to greyscale.

        Args:
            observation: gray observations

        Returns:
            Resized observation
        """


        observation = cv2.resize(
            observation, self.shape, interpolation=cv2.INTER_AREA
        )

        return observation

class PreprocessWrapper(gym.ObservationWrapper):
    """
    Preprocess the RGB observations.

    """

    def __init__(self, env: gym.Env, cut: int= 50):
        """Convert the image observation from RGB to gray scale.

        Args:
            env (Env): The environment to apply the wrapper
            cut: We wanna cut the observation to remove unnecessary features.
            keep_dim (bool): If `True`, a singleton dimension will be added, i.e. observations are of the shape AxBx1.
                Otherwise, they are of shape AxB.
        """
        super().__init__(env)
        assert (
            isinstance(self.observation_space, Box)
            and len(self.observation_space.shape) == 3
            and self.observation_space.shape[-1] == 3
        )


        self.observation_space = Box(
            low=0, high=255, shape=self.observation_space.shape, dtype=np.uint8
        )
 

    def observation(self, observation):
        """
        Preprocesses the observation.

        Args:
            observation: RGB observations

        Returns:
            preproceesed RGB observation
        """

        a = np.logical_or(observation[:,:,0] < 20 , observation[:,:,0] > 80)
        b = np.logical_or(observation[:,:,1] < 20 , observation[:,:,1] > 80)
        c = np.logical_or(observation[:,:,2] < 20 , observation[:,:,2] > 80)
        
        d = np.logical_and(a, b)
        e = np.logical_and(d, c)

        observation[e,0] = 255
        observation[e,1] = 255
        observation[e,2] = 255
        
        for i in range(120):
            a = np.logical_and(observation[:,:,0] >  i , observation[:,:,0] < 15 + i)
            b = np.logical_and(observation[:,:,1] >  i , observation[:,:,1] < 15 + i)
            c = np.logical_and(observation[:,:,2] >  i , observation[:,:,2] < 15 + i)
            
            d = np.logical_and(a, b)
            e = np.logical_and(d, c)

            observation[e,0] = 255
            observation[e,1] = 255
            observation[e,2] = 255
        
        
        a = np.logical_and(observation[:,:,0] >  47 , observation[:,:,0] < 69)
        b = np.logical_and(observation[:,:,1] >  60 , observation[:,:,1] < 86)
        c = np.logical_and(observation[:,:,2] >  72 , observation[:,:,2] < 103)
        
        d = np.logical_and(a, b)
        e = np.logical_and(d, c)
        
        observation[e,0] = 255
        observation[e,1] = 255
        observation[e,2] = 255
        
        
        a = np.logical_and(observation[:,:,0] >  47 , observation[:,:,0] < 69)
        b = np.logical_and(observation[:,:,1] >  60 , observation[:,:,1] < 86)
        c = np.logical_and(observation[:,:,2] >  72 , observation[:,:,2] < 103)
        
        d = np.logical_and(a, b)
        e = np.logical_and(d, c)
        
        observation[e,0] = 255
        observation[e,1] = 255
        observation[e,2] = 255

        observation = observation[:450, 50: ,:]

        # cv2.imshow("Input", observation)
        # cv2.imwrite('sup.png', observation)
        # cv2.waitKey(1000)
        # cv2.destroyAllWindows()
        return observation

class LazyFrames(object):
    r"""Ensures common frames are only stored once to optimize memory use.

    To further reduce the memory use, it is optionally to turn on lz4 to
    compress the observations.

    .. note::

        This object should only be converted to numpy array just before forward pass.

    Args:
        lz4_compress (bool): use lz4 to compress the frames internally

    """
    __slots__ = ("frame_shape", "dtype", "shape", "lz4_compress", "_frames")

    def __init__(self, frames, lz4_compress=False):
        self.frame_shape = tuple(frames[0].shape)
        self.shape = (len(frames),) + self.frame_shape
        self.dtype = frames[0].dtype
        if lz4_compress:
            from lz4.block import compress

            frames = [compress(frame) for frame in frames]
        self._frames = frames
        self.lz4_compress = lz4_compress

    def __array__(self, dtype=None):
        arr = self[:]
        if dtype is not None:
            return arr.astype(dtype)
        return arr

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, int_or_slice):
        if isinstance(int_or_slice, int):
            return self._check_decompress(self._frames[int_or_slice])  # single frame
        return np.stack(
            [self._check_decompress(f) for f in self._frames[int_or_slice]], axis=0
        )

    def __eq__(self, other):
        return self.__array__() == other

    def _check_decompress(self, frame):
        if self.lz4_compress:
            from lz4.block import decompress

            return np.frombuffer(decompress(frame), dtype=self.dtype).reshape(
                self.frame_shape
            )
        return frame


class FrameStack(ObservationWrapper):
    r"""Observation wrapper that stacks the observations in a rolling manner.

    For example, if the number of stacks is 4, then the returned observation contains
    the most recent 4 observations. For environment 'Pendulum-v1', the original observation
    is an array with shape [3], so if we stack 4 observations, the processed observation
    has shape [4, 3].

    .. note::

        To be memory efficient, the stacked observations are wrapped by :class:`LazyFrame`.

    .. note::

        The observation space must be `Box` type. If one uses `Dict`
        as observation space, it should apply `FlattenDictWrapper` at first.

    Example::

        >>> import gym
        >>> env = gym.make('PongNoFrameskip-v0')
        >>> env = FrameStack(env, 4)
        >>> env.observation_space
        Box(4, 210, 160, 3)

    Args:
        env (Env): environment object
        num_stack (int): number of stacks
        lz4_compress (bool): use lz4 to compress the frames internally

    """

    def __init__(self, env, num_stack, lz4_compress=False):
        super(FrameStack, self).__init__(env)
        self.num_stack = num_stack
        self.lz4_compress = lz4_compress

        self.frames = deque(maxlen=self.num_stack)

        low = np.repeat(self.observation_space.low[np.newaxis, ...], self.num_stack, axis=0)
        high = np.repeat(
            self.observation_space.high[np.newaxis, ...], self.num_stack, axis=0
        )
        self.observation_space = Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )

    def observation(self):
        assert len(self.frames) == self.num_stack, (len(self.frames), self.num_stack)


        return LazyFrames(list(self.frames), self.lz4_compress)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.frames.append(observation)
        
        
        
        return self.observation(), reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        [self.frames.append(observation) for _ in range(self.num_stack)]
        return self.observation()
    
    
    
