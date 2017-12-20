from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from util.utilities import save_fig


def linear_geometric_curve(n, starting_value, end_value, geometric_component=0.5, constant_at=1.0, geometric_end=1.0):
    """
    Used for creating a linear combination of a linear curve and a geometric curve.
    I think it's nice for learning rates.
    :param int n: Length of sequence.
    :param float starting_value: Initial value of curve.
    :param float end_value: Final values of curve.
    :param float geometric_component: Relative importance of geometric component vs linear component (between 0 and 1).
    :param float constant_at: Relative end of linear component.
    :param float geometric_end: Relative end of geometric component.
    :return: np.ndarray
    """
    assert 0 <= geometric_component <= 1

    # Length of linear component and geometric component
    linear_length = int(n * constant_at)
    geometric_length = int(n * geometric_end)

    # Linear component
    linear_values = np.linspace(start=starting_value * (1 - geometric_component),
                                stop=end_value,
                                num=linear_length)

    # Geometric component
    geometric_values = np.geomspace(start=starting_value * geometric_component,
                                    stop=end_value * 1e-3,
                                    num=geometric_length)
    geometric_values = geometric_values - geometric_values[n-1] + end_value * geometric_component

    # Overall array
    curve = np.zeros(n)

    # Add components
    curve[:geometric_length] = geometric_values[:n]
    curve[:linear_length] += linear_values

    # Add constant components in case curves end before end of sequence
    if constant_at != n:
        curve[linear_length:] += end_value * (1 - geometric_component)
    if geometric_end != n:
        curve[geometric_length:] += end_value * geometric_component

    return curve


def primary_secondary_plot(primary_xs, primary_values, secondary_plots, x_limit,
                           secondary_colors=None, primary_label="", secondary_label="",
                           x_label="", title="", grid=True):
    if secondary_colors is None:
        secondary_colors = ["blue", "green", "red", "orange"]

    # Get axes
    ax1 = plt.gca()  # type: plt.Axes
    ax2 = None
    if secondary_plots:
        ax2 = plt.twinx(ax1)  # type: plt.Axes
        plt.ticklabel_format(style='sci', axis='y')

    ###
    # Primary

    # Main plot
    ax1.plot(primary_xs, primary_values, '-', color="black")
    ax1.set_ylabel(primary_label, color="black")

    # Set x-label and title
    ax1.set_xlabel(x_label)
    ax1.set_title(title, fontsize=20)

    # Optional grid
    if grid:
        ax1.grid('on')

    # Set limits
    ax1.set_ylim(0, np.math.ceil(max(primary_values)))
    ax1.set_xlim(0, x_limit)

    ###
    # Secondary

    # X-values across batches
    x_values = range(x_limit)

    # Secondary axis is given learning rate
    for nr, values in enumerate(secondary_plots):
        if len(values) == x_limit:
            ax2.plot(x_values, values, color=secondary_colors[nr], alpha=1.0)
        else:
            ax2.plot(primary_xs, values, color=secondary_colors[nr], alpha=1.0)

    # Set limit of axes
    ax2.set_xlim(0, x_limit)

    # Ensure y-range is from zero
    ylim = tuple(ax2.get_ylim())
    ax2.set_ylim(0, ylim[1])

    # Mark ticks with first secondary color
    ax2.tick_params(axis='y', colors=secondary_colors[0])
    ax2.set_ylabel(secondary_label, color=secondary_colors[0])


if __name__ == "__main__":

    plt.close("all")

    n_batches = 1000

    # Make learning rates
    learning_rates = linear_geometric_curve(n=n_batches,
                                            starting_value=1e-2,
                                            end_value=1e-8,
                                            geometric_component=3. / 4,
                                            geometric_end=1.4)

    validation_x = range(0, n_batches, 5)
    validation = [val / n_batches for val in validation_x]

    primary_secondary_plot(
        primary_xs=validation_x,
        primary_values=validation,
        secondary_plots=[learning_rates],
        x_limit=n_batches,
        secondary_label="Learning rate",
        primary_label="Accuracy",
        title="Validation",
        x_label="Batch"
    )

    save_fig(Path("delete"))
