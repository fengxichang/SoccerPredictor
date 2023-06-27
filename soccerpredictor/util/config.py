from argparse import Namespace
from typing import Any, Dict
from soccerpredictor.util.constants import UNITS


class SPSingleton(type):
    """
    Python singleton implementation:
    https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python

    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)

        return cls._instances[cls]


class SPConfig(metaclass=SPSingleton):
    """
    Stores arguments passed by user when running program.
    Arguments are set during first run and restored when resuming the program to avoid
    changing important parameters between runs. If conflicting arguments are specified
    during consecutive runs, the config just reloads previous ones and ignores the new ones.
    Fixed arguments are restored by `restore_args` method, other arguments can be changed.

    存储用户在运行程序时传递的参数。
    参数在第一次运行时被设置，在恢复程序时被恢复，以避免在两次运行之间改变重要的参数。
    在运行期间改变重要参数。如果在连续运行期间指定了相互冲突的参数
    的时候，配置只是重新加载以前的参数，而忽略新的参数。
    固定的参数由`restore_args`方法恢复，其他参数可以被改变。
    """

    def __init__(self) -> None:
        self.epochs = None
        self.lrdecay = None
        self.lrpatience = None
        self.ntest = None
        self.ndiscard = None
        self.predict = None
        self.printfreq = None
        self.resume = None
        self.savefreq = None
        self.seed = None
        self.timesteps = None
        self.verbose = None
        self.units = UNITS

    def restore_args(self, model_settings: Dict[str, Any]) -> None:
        """
        Restores configuration from previously saved model settings.

        :param model_settings: Loaded model settings.
        """
        self.lrpatience = model_settings["config"]["lrpatience"]
        self.lrdecay = model_settings["config"]["lrdecay"]
        self.ntest = model_settings["config"]["ntest"]
        self.ndiscard = model_settings["config"]["ndiscard"]
        self.seed = model_settings["config"]["seed"]
        self.timesteps = model_settings["config"]["timesteps"]

        if self.verbose > 0:
            print("Restoring previous params: ("
                  f"lrpatience: {self.lrpatience},",
                  f"lrdecay: {self.lrdecay},",
                  f"ntest: {self.ntest},",
                  f"ndiscard: {self.ndiscard},",
                  f"seed: {self.seed},",
                  f"timesteps: {self.timesteps})")

    def set_args(self, args: Namespace) -> None:
        """
        Sets arguments passed by user/by default.

        :param args: Argparse arguments.
        """
        self.epochs = args.epochs
        self.lrdecay = args.lrdecay
        self.lrpatience = args.lrpatience
        self.ntest = args.ntest
        self.ndiscard = args.ndiscard
        self.predict = args.predict
        self.printfreq = args.printfreq
        self.savefreq = args.savefreq
        self.resume = args.resume
        self.seed = args.seed
        self.timesteps = args.timesteps
        self.verbose = args.verbose
