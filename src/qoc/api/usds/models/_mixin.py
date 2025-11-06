from collections.abc import Iterator, Mapping, Sequence
from typing import overload


class DictMixin[KT, VT](Mapping[KT, VT]):
    root: dict[KT, VT]

    def __getitem__(self, key: KT) -> VT:
        return self.root[key]

    def __iter__(self) -> Iterator[KT]:
        return iter(self.root)

    def __len__(self) -> int:
        return len(self.root)


class ListMixin[T](Sequence[T]):
    root: list[T]

    @overload
    def __getitem__(self, index: int) -> T: ...
    @overload
    def __getitem__(self, index: slice) -> list[T]: ...
    def __getitem__(self, index: int | slice) -> T | list[T]:
        return self.root[index]

    def __len__(self) -> int:
        return len(self.root)
