from __future__ import annotations
from typing import (
    Iterable, Iterator, Sequence, SupportsIndex,
    Any, Generic, TypeVar, overload
)


T = TypeVar('T')
U = TypeVar('U')
K = TypeVar('K')
V = TypeVar('V')


class GraphList(Sequence[T], Generic[T]):
    def __init__(self, init: Iterable[T] | None = None):
        self._data: list[T] = []
        if init is not None:
            self._data.extend(init)

    def append(self, x: T) -> None:
        self._data.append(x)

    def __contains__(self, x) -> bool:
        return any(_graph_eq(x, y) for y in self._data)

    def __len__(self) -> int:
        return len(self._data)

    @overload
    def __getitem__(self, index: SupportsIndex) -> T: ...

    @overload
    def __getitem__(self, index: slice) -> GraphList[T]: ...

    def __getitem__(self, index):
        if isinstance(index, slice):
            return GraphList(self._data[index])
        return self._data[index]

    @overload
    def __setitem__(self, index: SupportsIndex, value: T) -> None: ...

    @overload
    def __setitem__(self, index: slice, value: Iterable[T]) -> None: ...

    def __setitem__(self, index, value):
        if isinstance(index, slice):
            if not isinstance(value, Iterable):
                raise TypeError('Must assign an iterable to a slice')
            self._data[index] = list(value)
        else:
            self._data[index] = value

    def __iter__(self) -> Iterator[T]:
        return iter(self._data)

    def __reversed__(self) -> Iterator[T]:
        return reversed(self._data)

    def __bool__(self) -> bool:
        return len(self._data) != 0

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self._data!r})'

    def __add__(self, other: GraphList[U]) -> GraphList[T | U]:
        if not isinstance(other, GraphList):
            raise TypeError(
                f'Unsupported operand type(s) for +: '
                f'\'{type(self).__name__}\' and '
                f'\'{type(other).__name__}\''
            )

        result: GraphList[T | U] = GraphList()
        result.extend(self)
        result.extend(other)
        return result

    def clear(self) -> None:
        self._data.clear()

    def copy(self) -> GraphList[T]:
        copied: GraphList[T] = GraphList()
        copied._data = self._data.copy()
        return copied

    def extend(self, iterable: Iterable[T]) -> None:
        self._data.extend(iterable)

    def sort(self, key):
        self._data.sort(key=key)

    def tolist(self) -> list[T]:
        return self._data


class GraphDict(Generic[K, V]):
    def __init__(self, items: Iterable[tuple[K, V]] | None = None):
        self._buckets: dict[int, list[tuple[K, V]]] = {}
        if items is not None:
            for k, v in items:
                self[k] = v

    def __setitem__(self, key: K, value: V) -> None:
        h = _graph_hash(key)
        bucket = self._buckets.setdefault(h, [])
        for i, (k2, _) in enumerate(bucket):
            if _graph_eq(key, k2):
                bucket[i] = (k2, value)
                return
        bucket.append((key, value))

    def __getitem__(self, key: K) -> V:
        h = _graph_hash(key)
        bucket = self._buckets.get(h, [])
        for k2, v in bucket:
            if _graph_eq(key, k2):
                return v
        raise KeyError(f'Key {key!r} not found')

    def __delitem__(self, key: K) -> None:
        h = _graph_hash(key)
        bucket = self._buckets.get(h)
        if bucket is None:
            raise KeyError(f'Key {key!r} not found')
        for i, (k2, _) in enumerate(bucket):
            if _graph_eq(key, k2):
                bucket.pop(i)
                if not bucket:
                    del self._buckets[h]
                return
        raise KeyError(f'Key {key!r} not found')

    def __contains__(self, key: Any) -> bool:
        h = _graph_hash(key)
        bucket = self._buckets.get(h, [])
        return any(_graph_eq(key, k2) for k2, _ in bucket)

    @overload
    def get(self, key: K, default: None = None) -> V | None: ...

    @overload
    def get(self, key: K, default: V) -> V: ...

    def get(self, key: K, default: V | None = None) -> V | None:
        try:
            return self[key]
        except KeyError:
            return default

    def pop(self, key: K, default: Any = None) -> V:
        try:
            value = self[key]
            del self[key]
            return value
        except KeyError:
            if default is None:
                raise
            return default

    def keys(self) -> Iterator[K]:
        for bucket in self._buckets.values():
            for k, _ in bucket:
                yield k

    def values(self) -> Iterator[V]:
        for bucket in self._buckets.values():
            for _, v in bucket:
                yield v

    def items(self) -> Iterator[tuple[K, V]]:
        for bucket in self._buckets.values():
            for k, v in bucket:
                yield (k, v)

    def clear(self) -> None:
        self._buckets.clear()

    @overload
    def update(self, other: GraphDict[K, V]) -> None: ...
    @overload
    def update(self, other: Iterable[tuple[K, V]]) -> None: ...

    def update(self, other) -> None:
        if isinstance(other, GraphDict):
            for k, v in other.items():
                self[k] = v
        else:
            for k, v in other:
                self[k] = v

    def __len__(self) -> int:
        return sum(len(bucket) for bucket in self._buckets.values())

    def __iter__(self) -> Iterator[K]:
        return self.keys()

    def __repr__(self) -> str:
        items = ', '.join(f'{k!r}: {v!r}' for k, v in self.items())
        return f'GraphDict({{{items}}})'


class GraphSet(Generic[T]):
    def __init__(self, init: Iterable[T] | None = None):
        self._buckets: dict[int, list[T]] = {}
        if init is not None:
            for item in init:
                self.add(item)

    def add(self, item: T) -> None:
        h = _graph_hash(item)
        bucket = self._buckets.setdefault(h, [])
        for x in bucket:
            if _graph_eq(item, x):
                return
        bucket.append(item)

    def discard(self, item: T) -> None:
        h = _graph_hash(item)
        bucket = self._buckets.get(h)
        if not bucket:
            return
        for i, x in enumerate(bucket):
            if _graph_eq(item, x):
                bucket.pop(i)
                if not bucket:
                    del self._buckets[h]
                return

    def remove(self, item: T) -> None:
        h = _graph_hash(item)
        bucket = self._buckets.get(h)
        if not bucket:
            raise KeyError(f'Item {item!r} not found')
        for i, x in enumerate(bucket):
            if _graph_eq(item, x):
                bucket.pop(i)
                if not bucket:
                    del self._buckets[h]
                return
        raise KeyError(f'Item {item!r} not found')

    def pop(self) -> T:
        for h, bucket in list(self._buckets.items()):
            item = bucket.pop()
            if not bucket:
                del self._buckets[h]
            return item
        raise KeyError('pop from an empty set')

    def __contains__(self, item: Any) -> bool:
        h = _graph_hash(item)
        bucket = self._buckets.get(h, [])
        return any(_graph_eq(item, x) for x in bucket)

    def __len__(self) -> int:
        return sum(len(bucket) for bucket in self._buckets.values())

    def __iter__(self) -> Iterator[T]:
        for bucket in self._buckets.values():
            for x in bucket:
                yield x

    def copy(self) -> GraphSet[T]:
        new_set: GraphSet[T] = GraphSet()
        for item in self:
            new_set.add(item)
        return new_set

    def clear(self) -> None:
        self._buckets.clear()

    def union(self, other: Iterable[T]) -> GraphSet[T]:
        result = GraphSet(self)
        for item in other:
            result.add(item)
        return result

    def intersection(self, other: Iterable[T]) -> GraphSet[T]:
        result: GraphSet[T] = GraphSet()
        for item in self:
            if item in other:
                result.add(item)
        return result

    def difference(self, other: Iterable[T]) -> GraphSet[T]:
        result: GraphSet[T] = GraphSet()
        for item in self:
            if item not in other:
                result.add(item)
        return result

    def symmetric_difference(self, other: Iterable[T]) -> GraphSet[T]:
        result: GraphSet[T] = GraphSet()
        for item in self:
            if item not in other:
                result.add(item)
        for item in other:
            if item not in self:
                result.add(item)
        return result

    # Operator overloads
    def __or__(self, other: Iterable[T]) -> GraphSet[T]:
        return self.union(other)

    def __and__(self, other: Iterable[T]) -> GraphSet[T]:
        return self.intersection(other)

    def __xor__(self, other: Iterable[T]) -> GraphSet[T]:
        return self.symmetric_difference(other)

    def __ior__(self, other: Iterable[T]) -> GraphSet[T]:
        for item in other:
            self.add(item)
        return self

    def __iand__(self, other: Iterable[T]) -> GraphSet[T]:
        to_keep: list[T] = []
        for item in self:
            if item in other:
                to_keep.append(item)
        self.clear()
        for item in to_keep:
            self.add(item)
        return self

    def __ixor__(self, other: Iterable[T]) -> GraphSet[T]:
        for item in other:
            if item in self:
                self.discard(item)
            else:
                self.add(item)
        return self

    def __repr__(self) -> str:
        items = ', '.join(repr(x) for x in self)
        return f'GraphSet({{{items}}})'


def _graph_hash(x: Any) -> int:
    if hasattr(x, '__graph_hash__'):
        return x.__graph_hash__()
    return hash(x)


def _graph_eq(x, y) -> bool:
    if hasattr(x, '__graph_eq__'):
        return x.__graph_eq__(y)
    if hasattr(y, '__graph_eq__'):
        return y.__graph_eq__(x)
    return x == y


GList = GraphList
GDict = GraphDict
GSet = GraphSet
