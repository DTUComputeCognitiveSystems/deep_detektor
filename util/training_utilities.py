import numpy as np


def linear_geometric_curve(n, starting_value, end_value, geometric_component=0.5, linear_end=1.0, geometric_end=1.0):
    """
    Used for creating a linear combination of a linear curve and a geometric curve.
    I think it's nice for learning rates.
    :param int n: Length of sequence.
    :param float starting_value: Initial value of curve.
    :param float end_value: Final values of curve.
    :param float geometric_component: Relative importance of geometric component vs linear component (between 0 and 1).
    :param float linear_end: Relative end of linear component.
    :param float geometric_end: Relative end of geometric component.
    :return: np.ndarray
    """
    assert 0 <= geometric_component <= 1

    # Length of linear component and geometric component
    linear_length = int(n * linear_end)
    geometric_length = int(n * geometric_end)

    # Linear component
    linear_values = np.linspace(start=starting_value * (1 - geometric_component),
                                stop=end_value,
                                num=linear_length)

    # Geometric component
    geometric_values = np.geomspace(start=starting_value * geometric_component,
                                    stop=end_value * 1e-3,
                                    num=geometric_length)

    # Overall array
    curve = np.zeros(n)

    # Add components
    curve[:geometric_length] = geometric_values[:n]
    curve[:linear_length] += linear_values

    # Add constant components in case curves end before end of sequence
    if linear_end != n:
        curve[linear_length:] += end_value * (1 - geometric_component)
    if geometric_end != n:
        curve[geometric_length:] += end_value * geometric_component

    return curve
