import importlib


def make_env(config):
    module = config["module"]
    module = importlib.import_module(f"tfmpc.envs.{module}")
    cls_name = config["cls_name"]
    return getattr(module, cls_name).load(config["config"])
