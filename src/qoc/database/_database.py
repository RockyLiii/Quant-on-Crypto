import arcticdb as adb

from ._library import Library


class Database:
    db: adb.Arctic

    def __init__(self, uri: str = "mem://") -> None:
        self.db = adb.Arctic(uri=uri)

    def get_library(
        self, name: str, *, create_if_missing: bool = True, **kwargs
    ) -> Library:
        return Library(
            self.db.get_library(name, create_if_missing=create_if_missing, **kwargs)
        )
