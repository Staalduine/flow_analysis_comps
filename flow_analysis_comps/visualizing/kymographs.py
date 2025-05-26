from flow_analysis_comps.data_structs.kymographs import kymoOutputs


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
        """
        Plot the kymograph data.
        """
        import matplotlib.pyplot as plt
        plt.imshow(self.data.kymograph, aspect='auto', cmap='gray')
        plt.colorbar()
        plt.title('Kymograph')
        plt.ylabel('Time')
        plt.xlabel('Position')
        plt.show()