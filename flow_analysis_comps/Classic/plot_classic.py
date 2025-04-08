import numpy as np
import matplotlib.pyplot as plt
import colorcet

from flow_analysis_comps.Classic.model_parameters import videoDeltas


def plot_fields(
    fourier_images: np.ndarray, speed_images: np.ndarray, video_deltas: videoDeltas
):
    kymo_extent = [
        0,
        video_deltas.delta_x * len(fourier_images[0][0]),
        video_deltas.delta_t * len(fourier_images[0]),
        0,
    ]

    fig, ax = plt.subplot_mosaic(
        [["fourierLeft", "fourierRight"], ["speedLeft", "speedRight"]],
        layout="constrained",
    )

    ax["fourierLeft"].imshow(fourier_images[0], cmap="cet_CET_L20", extent=kymo_extent)
    ax["fourierRight"].imshow(fourier_images[1], cmap="cet_CET_L20", extent=kymo_extent)

    speed_abs = max_speed(speed_images)

    leftIm = ax["speedLeft"].imshow(
        speed_images[0],
        vmin=-speed_abs,
        vmax=speed_abs,
        cmap="cet_CET_D1A",
        extent=kymo_extent,
    )
    rightIm = ax["speedRight"].imshow(
        speed_images[1],
        vmin=-speed_abs,
        vmax=speed_abs,
        cmap="cet_CET_D1A",
        extent=kymo_extent,
    )

    for ax_title in ax:
        ax[ax_title].set_title(ax_title)
        ax[ax_title].set_aspect("auto")
        ax[ax_title].set_xlabel("Curvilinear distance ($\mu m$)")
        ax[ax_title].set_ylabel("time (s)")

    fig.colorbar(rightIm, ax=[ax["speedLeft"], ax["speedRight"]], aspect=40)


def max_speed(speed_images):
    speed_min = np.nanmin(speed_images)
    speed_max = np.nanmax(speed_images)

    print(speed_max, speed_min)
    speed_abs = np.max([abs(speed_min), speed_max])
    return speed_abs

    # fig.tight_layout()


def plot_summary(fourier_images, speed_images, video_deltas: videoDeltas):
    speedmax = max_speed(speed_images)
    kymo_extent = [
        0,
        video_deltas.delta_x * len(fourier_images[0][0]),
        video_deltas.delta_t * len(fourier_images[0]),
        0,
    ]

    time_axis_points = np.linspace(0, video_deltas.delta_t * len(fourier_images[0]), len(fourier_images[0]))

    speed_bins = np.linspace(-speedmax, speedmax, 1001)
    # speed_bins_trunc = (abs(speed_bins) < 7.0)[:-1]
    speed_histo_left = np.array(
        [np.histogram(row, speed_bins)[0] for row in speed_images[1]]
    )
    speed_histo_right = np.array(
        [np.histogram(row, speed_bins)[0] for row in speed_images[0]]
    )
    speed_histo = (speed_histo_left + speed_histo_right) / (2 * len(speed_images[0][0]))

    fig, ax = plt.subplot_mosaic(
        [["kymograph", "temporal histogram"], ["speed plot", "temporal histogram"]],
        layout="constrained",
    )

    ax["kymograph"].imshow(
        fourier_images[0] + fourier_images[1], cmap="cet_CET_L20", extent=kymo_extent
    )
    ax["kymograph"].set_xlabel("Curvilinear distance ($\mu m$)")
    ax["kymograph"].set_ylabel("time (s)")

    ax["speed plot"].plot(
        time_axis_points, np.nanmean(speed_images[0], axis=1), c="tab:orange", label="speed left"
    )
    # ax["speed plot"].fill_between()

    ax["speed plot"].plot(
        time_axis_points, np.nanmean(speed_images[1], axis=1), c="tab:blue", label="speed right"
    )
    ax["speed plot"].axhline(0, linestyle="--", c="black")
    ax["speed plot"].legend()
    ax["speed plot"].set_xlabel("time (s)")
    ax["speed plot"].set_ylabel("Speed ($\mu m / s$)")

    ax["temporal histogram"].imshow(
        speed_histo.T,
        extent=[0, len(speed_histo) * video_deltas.delta_t, -speedmax, speedmax],
        origin="lower",
    )
    ax["temporal histogram"].set_xlabel("time (s)")
    ax["temporal histogram"].set_ylabel("Speed ($\mu m / s$)")


    for ax_title in ax:
        ax[ax_title].set_title(ax_title)
        ax[ax_title].set_aspect("auto")
    return fig, ax
