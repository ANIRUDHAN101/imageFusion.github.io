import torch
import torch.functional as F
import torch.nn as nn
import random
from torchvision.transforms import CenterCrop
from torchvision.transforms import GaussianBlur
from torchvision.transforms.functional import gaussian_blur
from torchvision.transforms import functional

def blur_simulation_v1(image: torch.Tensor, mask: torch.Tensor, filter_size=None, deviation=None, low_threshold=.2, upper_threshold=.7, multiple_blur_choices=10):
    c = image.shape[1]
    n = image.shape[0]
    if image.shape != mask.shape:
        mask = mask.repeat(n, c, 1, 1)

    if filter_size is None:
        filter_size = random.choice(range(3, 7, 2))

    if deviation is None:
        deviation = random.choice(range(5, 30))

    blurred_image = functional.gaussian_blur(image, filter_size, (1, deviation))

    masked_image = torch.where(mask >= upper_threshold, 0, image)
    
    img_blurred_for_background = functional.gaussian_blur(masked_image, filter_size, (1, deviation))
    mask_blurred = functional.gaussian_blur(mask, filter_size, (1, deviation))
    img_blurred_for_foreground = functional.gaussian_blur(image, filter_size, (1, deviation))

    no_of_blures = random.choice(range(0, multiple_blur_choices))
    
    for _ in range(no_of_blures-1):
        img_blurred_for_background = functional.gaussian_blur(img_blurred_for_background, filter_size, (1, deviation))
        mask_blurred = functional.gaussian_blur(mask_blurred, filter_size, (1, deviation))
        img_blurred_for_foreground = functional.gaussian_blur(img_blurred_for_foreground, filter_size, (1, deviation))

    # Create a mask for the object
    mask_for_object = (1 - mask)*(mask_blurred) + (mask)
    #mask_for_object = torch.where(mask >=1, mask, mask_blurred)

    # Apply blurring for the object
    object_blurred = torch.where(mask_for_object <= low_threshold, image, img_blurred_for_foreground)

    background_blurred = ((1-mask)*(img_blurred_for_foreground)) + (image*(mask))

    mask_for_object = torch.where(mask_for_object <= low_threshold,0,
                                    torch.where(mask_for_object <= upper_threshold, .5, 1))

    return object_blurred, background_blurred, mask_for_object
    
def blur_simulation(image: torch.Tensor, mask: torch.Tensor, filter_size=None, deviation=None, low_threshold=.2, upper_threshold=.7, multiple_blur_choices=10):
    
    c = image.shape[1]
    n = image.shape[0]
    if image.shape != mask.shape:
        mask = mask.repeat(n, c, 1, 1)

    if filter_size is None:
        filter_size = random.choice(range(3, 7, 2))

    if deviation is None:
        deviation = random.choice(range(5, 30))

    blurred_image = functional.gaussian_blur(image, filter_size, (1, deviation))

    masked_image = torch.where(mask >= upper_threshold, 0, image)
    
    img_blurred_for_background = functional.gaussian_blur(masked_image, filter_size, (1, deviation))
    mask_blurred = functional.gaussian_blur(mask, filter_size, (1, deviation))
    img_blurred_for_foreground = functional.gaussian_blur(image, filter_size, (1, deviation))

    no_of_blures = random.choice(range(0, multiple_blur_choices))
    
    for _ in range(no_of_blures-1):
        img_blurred_for_background = functional.gaussian_blur(img_blurred_for_background, filter_size, (1, deviation))
        mask_blurred = functional.gaussian_blur(mask_blurred, filter_size, (1, deviation))
        img_blurred_for_foreground = functional.gaussian_blur(img_blurred_for_foreground, filter_size, (1, deviation))

    # Create a mask for the object
    mask_for_object = (1 - mask)*(mask_blurred) + (mask)
    #mask_for_object = torch.where(mask >=1, mask, mask_blurred)

    # Apply blurring for the object
    object_blurred = torch.where(mask_for_object <= low_threshold, image, img_blurred_for_foreground)

    background_blurred = ((1-mask)*(img_blurred_for_foreground)) + (image*(mask))

    mask_for_object = torch.where(mask_for_object <= low_threshold,0,
                                    torch.where(mask_for_object <= upper_threshold, .5, 1))

    return object_blurred, background_blurred, mask_for_object

def repeated_gaussian_blur(image, filter_size, deviation, multiple_blur_choices):
    if filter_size is None:
        filter_size = random.choice(range(3, 5, 2))

    if deviation is None:
        deviation = random.choice(range(5, 30))

    blurred_image = functional.gaussian_blur(image, filter_size, (1, deviation))
    
    no_of_blures = random.choice(range(0, multiple_blur_choices))

    for _ in range(no_of_blures-1):
        blurred_image = functional.gaussian_blur(blurred_image, filter_size, (1, deviation))
    
    return blurred_image

class GaussianBlur(nn.Module):
    """
    Applies Gaussian blur to an input tensor.

    Args:
        kernel_size (int): The size of the Gaussian kernel.
        sigma (int): The standard deviation of the Gaussian distribution.
        device (str): The device to be used for computation.
        dtype (str): The data type of the input images and masks.

    Attributes:
        kernal (torch.Tensor): The Gaussian kernel.
        kernel_size (int): The size of the Gaussian kernel.

    Methods:
        make_kernal: Creates the Gaussian kernel.
        forward: Performs the forward pass of the Gaussian blur.

    """

    def __init__(self, kernel_size: int, sigma: int, device: str) -> None:
        super(GaussianBlur, self).__init__()

        self.kernal = self.make_kernal(kernel_size, sigma)

        self.kernal = self.kernal.to(device)
        self.kernel_size = kernel_size

    def make_kernal(self, kernel_size: int, sigma: int):
        """
        Creates a Gaussian kernel.

        Args:
            kernel_size (int): The size of the Gaussian kernel.
            sigma (int): The standard deviation of the Gaussian distribution.

        Returns:
            torch.Tensor: The Gaussian kernel.

        """
        kernel = torch.zeros((kernel_size, kernel_size))

        for i in range(kernel_size):
            for j in range(kernel_size):
                kernel[i, j] = torch.exp(-torch.tensor(i - kernel_size / 2)**2 / (2 * sigma**2) - (j - kernel_size / 2)**2 / (2 * sigma**2))

        kernel = kernel / torch.sum(kernel)

        kernel = kernel.repeat(3, 3, 1, 1)

        return kernel
    
    def forward(self, x):
        """
        Performs the forward pass of the Gaussian blur.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The blurred tensor.

        """
        # x = nn.functional.pad(x, ((self.kernel_size - 1) // 2), mode="reflect")
        return nn.functional.conv2d(input=x, weight=self.kernal, padding=(self.kernel_size - 1) // 2 )

class RandomGaussianBlur(nn.Module):
    """
    A module that applies random Gaussian blur to an image.

    Args:
        filter_size (int): The size of the filter kernel for blurring.
        deviation (int): The deviation value for blurring.
        device: The device to be used for computation.

    Attributes:
        _LOWER_THRESHOLD (float): The lower threshold for mask values.
        _UPPER_THRESHOLD (float): The upper threshold for mask values.
        filter_size (int): The size of the filter kernel for blurring.
        deviation (int): The deviation value for blurring.
        gaussianBlur (GaussianBlur): The GaussianBlur module.

    Methods:
        forward(image, mask, no_of_blurs=1): Perform forward pass through the RandomGaussianBlur module.
    """

    _LOWER_THRESHOLD = .1
    _UPPER_THRESHOLD = .7
    _MAX_REPEATED_BLUR = 20

    def __init__(self, deviation: int, device :str, filter_size: int = 3) -> None:
        super(RandomGaussianBlur, self).__init__()
        assert isinstance(deviation, int)
        assert isinstance(filter_size, int)
        assert isinstance(device, str)
        # kernal size is set to 3 because larger kernals will create unrealestic effects blurring in the image
        self.filter_size = filter_size

        self.deviation = random.choice(range(5, 30)) if deviation is None else deviation
        # #self.gaussianBlur = GaussianBlur(kernel_size=self.filter_size, sigma = self.deviation, device=device)
        # self.gaussianblur = ga
        # self.gaussianBlurs = nn.ModuleList()
        # for _ in range(self._MAX_REPEATED_BLUR):
        #     self.gaussianBlurs.append(GaussianBlur(self.filter_size, self.deviation, device=device))
    
    def gaussianBlur(self, image):
        return gaussian_blur(image, self.filter_size, self.deviation)
    
    def forward(self, image: torch.Tensor, mask: torch.Tensor, multiple_blur_choices=1, dtype: torch.dtype=torch.float32):
        """
        Perform forward pass through the RandomGaussianBlur module.

        Args:
            image (torch.Tensor): The input image tensor.
            mask (torch.Tensor): The input mask tensor.
            no_of_blurs (int): The number of blurs to be applied.
            dtype (str): The data type of the input tensors.

        Returns:
            object_blurred (torch.Tensor): The blurred object tensor.
            background_blurred (torch.Tensor): The blurred background tensor.
            mask_for_object (torch.Tensor): The mask for the blurred object.
        """
        if image.dtype != dtype:
            image = image.type(dtype)
            mask = mask.type(dtype)

        object_blurred, background_blurred, mask_for_object = blur_simulation(image, mask, self.filter_size, self.deviation, self._LOWER_THRESHOLD, self._UPPER_THRESHOLD, multiple_blur_choices)
        
        centerCrop = CenterCrop(image.shape[-2:])
        # to ensure the shape of the input is maintained due to padding
        object_blurred = centerCrop(object_blurred)
        background_blurred = centerCrop(background_blurred)
        mask_blurred = centerCrop(mask_for_object)

        output = {}
        output['image'] = image
        output['input_img_1'] = object_blurred
        output['input_img_2'] = background_blurred
        output['mask'] = mask_for_object
        return output

        