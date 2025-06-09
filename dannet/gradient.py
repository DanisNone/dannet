from __future__ import annotations
from typing import Any, Callable, Generic, TypeVar
import dannet as dt


T = TypeVar('T')


class GradientOp(Generic[T]):
    _is_call: bool = False

    def __init__(
        self,
        fwd: Callable[..., T],
        bwd: Callable[..., tuple[dt.core.Tensor | None, ...]],
        nondiff_argnum: tuple[int, ...] = ()
    ):
        self.fwd = fwd
        self.bwd = bwd
        self.nondiff = tuple(map(int, nondiff_argnum))

    def __call__(self, *args: Any, **kwargs: Any) -> T:
        args = tuple(
            arg if i in self.nondiff else dt.array(arg)
            for i, arg in enumerate(args)
        )
        if self._is_call:
            return self.fwd(*args, **kwargs)

        GradientOp._is_call = True
        try:
            result = self.fwd(*args, **kwargs)
        finally:
            GradientOp._is_call = False

        GradientTape._add_op(self, result, args, kwargs)
        return result


class GradientTape:
    _run_instances: list[GradientTape] = []

    def __init__(self) -> None:
        self.tensors: list[tuple[
            GradientOp, dt.core.Tensor, tuple[Any, ...], dict[str, Any]
        ]] = []
        self.__used = False

    def __enter__(self) -> GradientTape:
        if self.__used:
            # TODO: add message
            raise RuntimeError
        self.__used = True
        self._run_instances.append(self)
        return self

    def __exit__(self, *args: Any) -> None:
        self._run_instances.remove(self)

    @classmethod
    def _add_op(
        cls,
        op: GradientOp, result: Any,
        args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        for instance in cls._run_instances:
            instance.tensors.append((op, result, args, kwargs))

    def gradients(
        self,
        loss: dt.core.Tensor,
        params: list[dt.core.Tensor]
    ) -> list[dt.core.Tensor]:
        gradients = {id(loss): dt.ones_like(loss)}
        for op, result, args, kwargs in self.tensors[::-1]:
            if isinstance(result, tuple):
                grad_out = [gradients.get(id(out), None) for out in result]
                if all(g for g in grad_out if g is None):
                    continue
            else:
                grad_out = gradients.get(id(result))
                if grad_out is None:
                    continue
            grads = op.bwd(grad_out, result, args, kwargs)
            if not isinstance(grads, tuple):
                grads = (grads, )

            diff_args = [
                arg for i, arg in enumerate(args)
                if i not in op.nondiff
            ]
            if len(diff_args) != len(grads):
                # TODO: add message
                raise ValueError("")
            for i, (inp, inp_grad) in enumerate(zip(diff_args, grads)):
                if inp_grad is None:
                    continue
                if id(inp) not in gradients:
                    gradients[id(inp)] = inp_grad
                else:
                    gradients[id(inp)] += inp_grad
        return [gradients.get(id(p), dt.zeros_like(p)) for p in params]
