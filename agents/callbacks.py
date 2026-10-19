import os
import warnings

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecMonitor, VecVideoRecorder


class SaveCallback(CheckpointCallback):
    """Callback for saving a model every ``save_freq`` calls
    to ``env.step()`` and updating the latest saved model.
    By default, it only saves model checkpoints,
    you need to pass ``save_replay_buffer=True``,
    and ``save_vecnormalize=True`` to also save replay buffer checkpoints
    and normalization statistics checkpoints.

    Parameters
    ----------
    save_freq : int
        Save checkpoints every ``save_freq`` call of the callback.
    save_path: str
        Path to the folder where the model will be saved.
    name_prefix : str, default=""
        The seed used for random number generation to initialize the environment.
    save_replay_buffer : bool, default=False
        Save the model replay buffer.
    save_vecnormalize : bool, default=False
        Save the ``VecNormalize`` statistics.
    verbose: int, default=0
        Verbosity level: 0 for no output, 2 for indicating when saving model checkpoint.

    Notes
    -----
    When using multiple environments, each call to  ``env.step()``
    will effectively correspond to ``n_envs`` steps.
    To account for that, you can use ``save_freq = max(save_freq // n_envs, 1)``
    """

    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str = "",
        save_replay_buffer: bool = False,
        save_vecnormalize: bool = False,
        verbose: int = 0,
    ):
        super().__init__(save_freq, save_path, name_prefix, save_replay_buffer, save_vecnormalize, verbose)

    def _checkpoint_path(self, checkpoint_type: str = "", extension: str = "") -> str:
        """Helper to get checkpoint path for each type of checkpoint.

        Parameters
        ----------
        checkpoint_type : str, default=""
            empty for the model, "replay_buffer_"
            or "vecnormalize_" for the other checkpoints.
        extension: str, default=""
            Checkpoint file extension (zip for model, pkl for others)

        Returns
        -------
        str
            Path to the checkpoint.
        """
        return os.path.join(self.save_path, f"{checkpoint_type}{self.num_timesteps}.{extension}")

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            super()._on_step()

            model_path = os.path.join(self.save_path, "latest_model.zip")
            self.model.save(model_path)
            if self.verbose >= 2:
                print(f"Saving latest model checkpoint to {model_path}")

            if (
                self.save_replay_buffer
                and hasattr(self.model, "replay_buffer")
                and self.model.replay_buffer is not None
            ):
                # If model has a replay buffer, save it too
                replay_buffer_path = os.path.join(self.save_path, "replay_buffer_latest_model.pkl")
                self.model.save_replay_buffer(replay_buffer_path)
                if self.verbose > 1:
                    print(f"Saving latest model replay buffer checkpoint to {replay_buffer_path}")

            if self.save_vecnormalize and self.model.get_vec_normalize_env() is not None:
                # Save the VecNormalize statistics
                vec_normalize_path = os.path.join(self.save_path, "vecnormalize_latest_model.pkl")
                self.model.get_vec_normalize_env().save(vec_normalize_path)
                if self.verbose >= 2:
                    print(f"Saving latest model VecNormalize to {vec_normalize_path}")

        return True


class SaveBestNormalizeCallback(BaseCallback):
    """Callback for saving the vecnormalize and replay buffer when
    a new best model is found.

    Parameters
    ----------
    save_path: str
        Path to the folder where the model will be saved.
    verbose: int, default=0
        Verbosity level: 0 for no output, 2 for indicating when saving model checkpoint.
    """

    def __init__(self, save_path: str, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.save_path = save_path

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if hasattr(self.model, "replay_buffer") and self.model.replay_buffer is not None:
            # If model has a replay buffer, save it too
            replay_buffer_path = os.path.join(self.save_path, "replay_buffer_best_model.pkl")
            self.model.save_replay_buffer(replay_buffer_path)
            if self.verbose > 1:
                print(f"Saving best model replay buffer checkpoint to {replay_buffer_path}")

        if self.model.get_vec_normalize_env() is not None:
            # Save the VecNormalize statistics
            vec_normalize_path = os.path.join(self.save_path, "vecnormalize_best_model.pkl")
            self.model.get_vec_normalize_env().save(vec_normalize_path)
            if self.verbose >= 2:
                print(f"Saving best model VecNormalize to {vec_normalize_path}")

        return True


class VideoRecordCallback(BaseCallback):
    """Callback for saving a video of the model's evaluation.

    Parameters
    ----------
    save_path: str
        Path to the folder where the video will be saved.
    video_length: int
        Length of recorded video.
    log_dir: str
        Path of the directory where log info is saved.
    verbose: int, default=0
        Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages.
    """

    def __init__(self, save_path: str, video_length: int, log_dir: str, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.save_path = save_path
        self.video_length = video_length
        self.log_dir = log_dir

    def _init_callback(self) -> None:
        assert self.parent is not None, "``VideoRecordCallback`` callback must be used with an ``EvalCallback``"

        self.recorder = VecVideoRecorder(
            self.parent.eval_env,
            self.save_path,
            record_video_trigger=lambda x: x == 0,
            video_length=self.video_length,
            name_prefix="eval_callback_video",
        )
        self.eval_env = VecMonitor(self.recorder, self.log_dir)

        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

        # Create folders if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # The parent EvalCallback has synchronized the normalization of its eval env, which this one wraps.
        episode_rewards, episode_lengths = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=1,
            render=True,
            deterministic=True,
            return_episode_rewards=True,
            warn=True,
        )

        # SB3 >= 2 has no public way to save a video shorter than video_length without closing the env.
        if self.recorder.recording:
            self.recorder._stop_recording()

        return True

    def _on_training_end(self) -> None:
        self.eval_env.close()
