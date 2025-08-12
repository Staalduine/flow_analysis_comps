from numpydantic import NDArray, Shape

image_float = NDArray[Shape["* x, * y"], float]  # type: ignore # A 2D image with float values
image_int = NDArray[Shape["* x, * y"], int]  # type: ignore # A 2D image with integer values