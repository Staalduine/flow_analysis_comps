import numpy as np
from typing import List, Tuple


class GaussianLineGenerator:
    def __init__(self, image_size: Tuple[int, int] = (100, 100)):
        """
        Initialize the line generator.
        Args:
            image_size: Tuple of (height, width) for the output image
        """
        self.image_size = image_size
        self.image = np.zeros(image_size)

    def add_line(
        self,
        angle: float,
        std_dev: float,
        amplitude: float = 1.0,
        center: Tuple[float, float] | None = None,
    ):
        """
        Add a line with Gaussian width profile to the image.
        Args:
            angle: Angle of the line in degrees
            std_dev: Standard deviation of the Gaussian profile
            amplitude: Peak intensity of the line
            center: (y, x) coordinates of line center. If None, uses image center
        """
        if center is None:
            center = (self.image_size[0] / 2, self.image_size[1] / 2)

        # Create coordinate grids
        y, x = np.ogrid[: self.image_size[0], : self.image_size[1]]

        # Convert angle to radians
        theta = np.radians(angle)

        # Calculate distance from each point to the line
        # Line equation: x*sin(theta) - y*cos(theta) = 0
        distance = x * np.sin(theta) - y * np.cos(theta)
        distance = distance - (center[1] * np.sin(theta) - center[0] * np.cos(theta))

        # Create Gaussian profile
        gaussian = amplitude * np.exp(-(distance**2) / (2 * std_dev**2))

        # Add to image
        self.image += gaussian

    def get_image(self) -> np.ndarray:
        """
        Returns the generated image.
        """
        return self.image

    def clear_image(self):
        """
        Reset the image to zeros.
        """
        self.image = np.zeros(self.image_size)


def single_line_img(angle:float, std_dev:float = 3, img_size : Tuple[int, int] = (100, 100)):
    generator = GaussianLineGenerator(img_size)
    img_center = (img_size[0] // 2, img_size[1] // 2)

    generator.add_line(angle, std_dev=std_dev, center=img_center)
    return generator.get_image()

def crossing_line_img(angles: list[float], std_devs: list[float], img_size : Tuple[int, int] = (100, 100)):
    assert len(angles) == len(std_devs)
    
    generator = GaussianLineGenerator(img_size)
    img_center = (img_size[0] // 2, img_size[1] // 2)
    for angle, std_dev in zip(angles, std_devs):
        generator.add_line(angle, std_dev, center=img_center)
    return generator.get_image()