import enum
import functools
import sys
from pathlib import Path

from environs import env


class AppEnv(enum.StrEnum):
    DEVELOPMENT = enum.auto()
    PRODUCTION = enum.auto()


def app_env() -> AppEnv:
    load_env()
    return env.enum("APP_ENV", AppEnv.DEVELOPMENT, enum=AppEnv, by_value=True)


def get_base_url(name: str, prod: str, testnet: str) -> str:
    if base_path := env.str(name, None):
        return base_path
    match app_env():
        case AppEnv.DEVELOPMENT:
            return testnet
        case AppEnv.PRODUCTION:
            return prod
        case _:
            raise ValueError


def is_dev() -> bool:
    return app_env() is AppEnv.DEVELOPMENT


def is_prod() -> bool:
    return app_env() is AppEnv.PRODUCTION


@functools.cache
def load_env() -> None:
    env.read_env(Path(sys.argv[0]).with_name(".env"))
