from flow_analysis_comps.data_structs.kymographs import kymoOutputs
import matplotlib.pyplot as plt
import colorcet

class kymoVisualizer:
    """
    Class to visualize kymographs.
    """

    def __init__(self, data: kymoOutputs):
        """
        Initialize the kymograph visualizer with data.

        Parameters:
        data (kymoOutputs): The kymograph data to visualize.
        """
        self.data = data

    def plot(self):
        
        # Plot original kymograph, decomposed and non-static kymographs
        extent = (
            0,
            self.data.deltas.delta_x * len(self.data.kymograph[0]),
            self.data.deltas.delta_t * len(self.data.kymograph),
            0,
        )
        
        fig, ax = plt.subplots(2, 2, figsize=(5, 5))
        ax[0, 0].imshow(self.data.kymograph, cmap="cet_CET_L20", extent=extent)
        ax[0, 0].set_title("Original Kymograph")
        ax[0, 1].imshow(self.data.kymo_left, cmap="cet_CET_L20", extent=extent)
        ax[0, 1].set_title("Decomposed Left Kymograph")
        ax[1, 1].imshow(self.data.kymo_right, cmap="cet_CET_L20", extent=extent)
        ax[1, 1].set_title("Decomposed Right Kymograph")
        ax[1, 0].imshow(self.data.kymo_no_static, cmap="cet_CET_L20", extent=extent)
        ax[1, 0].set_title("Non-static Kymograph")
        for a in ax.flat:
            a.set_xlabel("Curvilinear distance ($\mu m$)")
            a.set_ylabel("Time (s)")
            a.set_aspect("auto")
        plt.tight_layout()
        plt.show()