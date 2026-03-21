# Lazy imports to avoid pulling in heavy deps (omegaconf, ray, etc.)
# when only lightweight submodules (prompts, action_parser) are needed.


def __getattr__(name):
    if name == "BaseWebEnvManager":
        from poisonclaw.envs.base_web_env import BaseWebEnvManager
        return BaseWebEnvManager
    if name == "MiniWoBEnv":
        from poisonclaw.envs.miniwob_env import MiniWoBEnv
        return MiniWoBEnv
    if name == "MiniWoBEnvManager":
        from poisonclaw.envs.miniwob_env import MiniWoBEnvManager
        return MiniWoBEnvManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
