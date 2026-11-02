import copy
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import List


class InterpMethod(Enum):

    PIECEWISE_CONSTANT_LEFT_CONTINUOUS = 'PIECEWISE_CONSTANT_LEFT_CONTINUOUS'
    LINEAR = 'LINEAR'
    MONOTONE_CONVEX = 'MONOTONE_CONVEX'

    @classmethod
    def from_string(cls, value: str) -> 'InterpMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class ExtrapMethod(Enum):

    FLAT = 'FLAT'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'ExtrapMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class Interpolator1D(ABC):
    """Abstract interface for a 1-D interpolator."""

    def __init__(self,
                 axis1: np.ndarray,
                 values: np.ndarray,
                 interpolation_method: InterpMethod,
                 extrapolation_method: ExtrapMethod) -> None:

        self.axis1_ = axis1
        self.values_ = values
        self.interp_method_ = interpolation_method
        self.extrap_method_ = extrapolation_method
        self.length_ = len(self.axis1_)

    @abstractmethod
    def interpolate(self, x: float) -> float:
        pass

    @abstractmethod
    def integrate(self, start_x: float, end_x: float) -> float:
        pass

    @abstractmethod
    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        pass

    @abstractmethod
    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        pass

    @property
    def axis1(self) -> np.ndarray:
        return self.axis1_

    @property
    def values(self) -> np.ndarray:
        return self.values_

    @property
    def length(self) -> int:
        return self.length_

    @property
    def interp_method(self) -> str:
        return self.interp_method_.to_string()

    @property
    def extrap_method(self) -> str:
        return self.extrap_method_.to_string()


class Interpolator1DPCP(Interpolator1D):
    """Piecewise-constant left-continuous interpolator with FLAT extrapolation.

    With axis1 = [1, 3, 5, 7], values = [3, 4, 5, 6]:
        f(0.5) = 3, f(1) = 3, f(1.5) = 4, f(3) = 4, f(5.5) = 6, f(8) = 6
    """

    def __init__(self, axis1: np.ndarray, values: np.ndarray,
                 extrapolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values,
                         InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS,
                         extrapolation_method)
        assert self.extrap_method_ == ExtrapMethod.FLAT

    def interpolate(self, x: float) -> float:
        idx = min(int(np.searchsorted(self.axis1_, x, side='left')), self.length_ - 1)
        return float(self.values_[idx])

    def integrate(self, start_x: float, end_x: float) -> float:
        edges = np.concatenate(([-np.inf], np.asarray(self.axis1_, dtype=float)[:-1], [np.inf]))
        lo = np.maximum(start_x, edges[:-1])
        hi = np.minimum(end_x, edges[1:])
        overlaps = np.maximum(hi - lo, 0.0)
        return float(np.dot(np.asarray(self.values_, dtype=float), overlaps))

    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        idx = min(int(np.searchsorted(self.axis1_, x, side='left')), self.length_ - 1)
        grad = np.zeros(self.length_)
        grad[idx] = 1.0
        return grad

    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        edges = np.concatenate(([-np.inf], np.asarray(self.axis1_, dtype=float)[:-1], [np.inf]))
        lo = np.maximum(start_x, edges[:-1])
        hi = np.minimum(end_x, edges[1:])
        return np.maximum(hi - lo, 0.0)



class Interpolator1DMonotoneConvex(Interpolator1D):
    """Shape-preserving C1 quadratic spline through the nodes, FLAT outside them.

    As in Hagan and West's monotone convex method, each node interval holds one
    quadratic or two quadratics joined at an extra knot (their breakpoint eta).

    1. Node slopes s_i from the secants d_i: interior slopes use the three-point
       (parabolic) estimate, 0 at a local extremum, capped at 2 * min(|d_i|, |d_{i+1}|);
       end slopes make the end interval a single quadratic, s_0 = 2 d_1 - s_1.
       Each s_i lies between its two neighbouring secants.
    2. On an interval with secant d, the derivative must equal d somewhere (mean
       value theorem). The knot goes there and the derivative is linear on each
       side, s_{i-1} -> d -> s_i: convex if s_{i-1} < d < s_i, concave in the
       mirror case, and monotone because the three slopes share a sign.

    For monotone, strictly convex nodes the curve is continuous, C1, monotone and
    convex on [x_0, x_n]. Other data: a local extremum gets slope 0 (no overshoot,
    the side reaching it is concave); where the secants turn, the knot goes to the
    midpoint and holds the interval's single inflection (the slope cap keeps it
    monotone); collinear nodes give a straight segment with a convex kink at the
    end of the run. One node gives a constant, two nodes a straight line.
    """

    def __init__(self, axis1: np.ndarray, values: np.ndarray,
                 extrapolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values, InterpMethod.MONOTONE_CONVEX,
                         extrapolation_method)
        assert self.extrap_method_ == ExtrapMethod.FLAT
        self.x_ = np.asarray(self.axis1_, dtype=float)
        self.y_ = np.asarray(self.values_, dtype=float)
        if not np.all(np.diff(self.x_) > 0):
            raise ValueError('MONOTONE_CONVEX needs strictly increasing nodes.')
        self.slopes_ = self.node_slopes(self.x_, self.y_)
        self._build_pieces()

    @staticmethod
    def node_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        n = len(x)
        if n == 1:
            return np.zeros(1)
        h = np.diff(x)
        d = np.diff(y) / h
        if n == 2:
            return np.array([d[0], d[0]])
        s = np.zeros(n)
        for i in range(1, n - 1):
            if d[i - 1] * d[i] > 0:
                three_point = (h[i] * d[i - 1] + h[i - 1] * d[i]) / (h[i - 1] + h[i])
                cap = 2.0 * min(abs(d[i - 1]), abs(d[i]))
                s[i] = np.sign(d[i]) * min(abs(three_point), cap)
        # s_0 + s_1 = 2 d_1 makes the end interval one quadratic; the cap keeps s_0 on d_1's side of 0.
        s[0] = 2.0 * d[0] - s[1]
        s[-1] = 2.0 * d[-1] - s[-2]
        return s

    def _build_pieces(self) -> None:
        """Per interval: knot, slope and value at the knot, both curvatures, left-piece integral."""
        m = max(self.length_ - 1, 0)
        self.knots_, self.knot_slopes_, self.knot_values_ = np.zeros(m), np.zeros(m), np.zeros(m)
        self.left_curvatures_, self.right_curvatures_ = np.zeros(m), np.zeros(m)
        self.left_integrals_, interval_integrals = np.zeros(m), np.zeros(m)
        for i in range(m):
            x0, y0, s0, s1 = self.x_[i], self.y_[i], self.slopes_[i], self.slopes_[i + 1]
            h = self.x_[i + 1] - x0
            d = (self.y_[i + 1] - y0) / h
            u, v = d - s0, s1 - d
            tol = 1e-12 * max(abs(s0), abs(s1), abs(d))
            if abs(u) <= tol and abs(v) <= tol:
                share = 0.5                   # linear
            elif abs(u) <= tol:
                share = 1.0                   # collinear on the left: linear, kink at x_{i+1}
            elif abs(v) <= tol:
                share = 0.0                   # collinear on the right: linear, kink at x_i
            elif u * v > 0:
                share = v / (u + v)           # convex or concave: slope d at the knot
            else:
                share = 0.5                   # secants turn: one inflection, at the knot
            alpha, beta = share * h, (1.0 - share) * h
            # The value increments of the two pieces must add up to h * d.
            s_knot = (2.0 * h * d - alpha * s0 - beta * s1) / h
            self.knots_[i] = x0 + alpha
            self.knot_slopes_[i] = s_knot
            self.knot_values_[i] = y0 + 0.5 * alpha * (s0 + s_knot)
            self.left_curvatures_[i] = (s_knot - s0) / alpha if alpha > 0 else 0.0
            self.right_curvatures_[i] = (s1 - s_knot) / beta if beta > 0 else 0.0
            self.left_integrals_[i] = alpha * y0 + alpha ** 2 * (2.0 * s0 + s_knot) / 6.0
            interval_integrals[i] = (self.left_integrals_[i] + beta * self.knot_values_[i]
                                     + beta ** 2 * (2.0 * s_knot + s1) / 6.0)
        # Integral from x_0 to each node.
        self.node_integrals_ = np.concatenate(([0.0], np.cumsum(interval_integrals)))

    def _piece(self, x: float):
        """Interval index, start, value, slope and curvature of the quadratic covering x."""
        i = min(int(np.searchsorted(self.x_, x, side='right')) - 1, self.length_ - 2)
        if x <= self.knots_[i]:
            return i, self.x_[i], self.y_[i], self.slopes_[i], self.left_curvatures_[i]
        return (i, self.knots_[i], self.knot_values_[i], self.knot_slopes_[i],
                self.right_curvatures_[i])

    def interpolate(self, x: float) -> float:
        if self.length_ == 1 or x <= self.x_[0]:
            return float(self.y_[0])
        if x >= self.x_[-1]:
            return float(self.y_[-1])
        _, start, value, slope, curvature = self._piece(x)
        dx = x - start
        return float(value + slope * dx + 0.5 * curvature * dx ** 2)

    def _antiderivative(self, x: float) -> float:
        """Integral from x_0 to x (negative for x < x_0)."""
        if self.length_ == 1 or x <= self.x_[0]:
            return float(self.y_[0] * (x - self.x_[0]))
        if x >= self.x_[-1]:
            return float(self.node_integrals_[-1] + self.y_[-1] * (x - self.x_[-1]))
        i, start, value, slope, curvature = self._piece(x)
        dx = x - start
        partial = value * dx + 0.5 * slope * dx ** 2 + curvature * dx ** 3 / 6.0
        if start > self.x_[i]:
            partial += self.left_integrals_[i]
        return float(self.node_integrals_[i] + partial)

    def integrate(self, start_x: float, end_x: float) -> float:
        return self._antiderivative(end_x) - self._antiderivative(start_x)

    # Optional risk interface
    # The test cells for risk are not prepared but you can do it by your own.
    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        raise NotImplementedError("Optional risk: ordinate gradient is not implemented")

    def gradient_of_integrated_value_wrt_ordinate(
        self, start_x: float, end_x: float,
    ) -> np.ndarray:
        raise NotImplementedError("Optional risk: integrated ordinate gradient is not implemented")


class InterpolatorFactory:

    @staticmethod
    def create_1d_interpolator(axis1: np.ndarray | List,
                               values: np.ndarray | List,
                               interpolation_method: InterpMethod,
                               extrapolation_method: ExtrapMethod):

        axis1_ = copy.deepcopy(axis1)
        values_ = copy.deepcopy(values)
        if isinstance(axis1_, list):
            axis1_ = np.array(axis1_)
        if isinstance(values_, list):
            values_ = np.array(values_)
        assert len(axis1_.shape) == 1 and len(values_.shape) == 1
        assert len(axis1_) == len(values_)
        assert np.all(np.diff(axis1_) >= 0)

        if interpolation_method == InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS:
            return Interpolator1DPCP(axis1_, values_, extrapolation_method)
        elif interpolation_method == InterpMethod.MONOTONE_CONVEX:
            return Interpolator1DMonotoneConvex(axis1_, values_, extrapolation_method)
        else:
            raise ValueError(f'Unsupported interpolation method: {interpolation_method}')
