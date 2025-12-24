import random
import numpy as np
import collections.abc
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from .backend import _OPENCV_AVAILABLE, get_backend
import tensorplay as tp

if _OPENCV_AVAILABLE:
    import cv2

def _is_numpy_image(img):
    return isinstance(img, np.ndarray)

def _get_image_size(img):
    if _is_numpy_image(img):
        return img.shape[1], img.shape[0]  # W, H
    return img.size  # W, H

def _pil_interp_to_cv2(interpolation):
    if not _OPENCV_AVAILABLE: return interpolation
    if interpolation == Image.NEAREST: return cv2.INTER_NEAREST
    if interpolation == Image.BILINEAR: return cv2.INTER_LINEAR
    if interpolation == Image.BICUBIC: return cv2.INTER_CUBIC
    if interpolation == Image.LANCZOS: return cv2.INTER_LANCZOS4
    return cv2.INTER_LINEAR

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class RandomHorizontalFlip:
    """Horizontally flip the given PIL Image or NumPy array randomly with a given probability."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            if _is_numpy_image(img):
                if not _OPENCV_AVAILABLE:
                     return np.ascontiguousarray(img[:, ::-1, ...])
                return cv2.flip(img, 1)
            elif isinstance(img, Image.Image):
                return img.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                raise TypeError("img should be PIL Image or NumPy array")
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomResizedCrop:
    """Crop the given PIL Image or NumPy array to random size and aspect ratio."""
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        if isinstance(size, collections.abc.Sequence):
            self.size = size
        else:
            self.size = (size, size)
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop."""
        width, height = _get_image_size(img)
        area = width * height

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (np.log(ratio[0]), np.log(ratio[1]))
            aspect_ratio = np.exp(random.uniform(*log_ratio))

            w = int(round(np.sqrt(target_area * aspect_ratio)))
            h = int(round(np.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to center crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        
        if _is_numpy_image(img):
            if not _OPENCV_AVAILABLE:
                raise ImportError("OpenCV is required for numpy array transforms")
            # Crop
            cropped = img[i:i+h, j:j+w]
            # Resize
            cv_interp = _pil_interp_to_cv2(self.interpolation)
            return cv2.resize(cropped, self.size[::-1], interpolation=cv_interp)

        elif isinstance(img, Image.Image):
            img = img.crop((j, i, j + w, i + h))
            return img.resize(self.size[::-1], self.interpolation)
        else:
            raise TypeError("img should be PIL Image or NumPy array")

    def __repr__(self):
        interpolate_str = str(self.interpolation)
        return self.__class__.__name__ + '(size={0}, scale={1}, ratio={2}, interpolation={3})'.format(
            self.size, self.scale, self.ratio, interpolate_str)

class ToTensor:
    """Convert a PIL Image or numpy.ndarray to tensor."""
    def __call__(self, pic):
        if isinstance(pic, Image.Image):
            # Handle PIL Image
            if pic.mode == 'I':
                img = np.array(pic, np.int32, copy=False)
            elif pic.mode == 'I;16':
                img = np.array(pic, np.int16, copy=False)
            elif pic.mode == 'F':
                img = np.array(pic, np.float32, copy=False)
            elif pic.mode == '1':
                img = 255 * np.array(pic, np.uint8, copy=False)
            else:
                # Optimized path for standard images (RGB, L, etc.) -> uint8
                # Note: copy=False might cause issues with nanobind if memory layout is weird
                img = np.array(pic, np.uint8)
                if img.ndim == 2:
                    img = img[:, :, None]
                if not img.flags['C_CONTIGUOUS']:
                    img = np.ascontiguousarray(img)
                
                # Use C++ optimized conversion: HWC(uint8) -> CHW(float32) / 255.0
                if hasattr(tp, 'vision_to_tensor'):
                    return tp.vision_to_tensor(img)
                
            # Put it from HWC to CHW format
            if pic.mode == 'YCbCr':
                nchannel = 3
            elif pic.mode == 'I;16':
                nchannel = 1
            else:
                nchannel = len(pic.mode)
                
            img = img.reshape(pic.size[1], pic.size[0], nchannel)
            img = img.transpose((2, 0, 1))
            img = np.ascontiguousarray(img)
            
            t = tp.tensor(img)
            
            if t.dtype == tp.uint8:
                t = t.to(tp.float32).div(255.0)
            elif t.dtype == tp.int16:
                t = t.to(tp.float32).div(32767.0)
            elif t.dtype == tp.int32:
                t = t.to(tp.float32).div(2147483647.0)
                 
            return t
            
        elif isinstance(pic, np.ndarray):
            # handle numpy array
            if pic.ndim == 2:
                pic = pic[:, :, None]
            
            if pic.dtype == np.uint8:
                 # Optimized path
                 if not pic.flags['C_CONTIGUOUS']:
                     pic = np.ascontiguousarray(pic)
                 if hasattr(tp, 'vision_to_tensor'):
                     return tp.vision_to_tensor(pic)

            # HWC to CHW
            img = pic.transpose((2, 0, 1))
            img = np.ascontiguousarray(img)
            t = tp.tensor(img)
             
            if t.dtype == tp.uint8:
                return t.to(tp.float32).div_(255.0)
            
            return t
        else:
            raise TypeError(f"pic should be PIL Image or ndarray. Got {type(pic)}")

    def __repr__(self):
        return self.__class__.__name__ + '()'

def from_image(img):
    """
    Directly create a Tensor from a PIL Image or NumPy array.
    Uses optimized C++ path if possible.
    """
    if isinstance(img, Image.Image):
        if img.mode == 'I':
            arr = np.array(img, np.int32, copy=False)
        elif img.mode == 'I;16':
            arr = np.array(img, np.int16, copy=False)
        elif img.mode == 'F':
            arr = np.array(img, np.float32, copy=False)
        elif img.mode == '1':
            arr = 255 * np.array(img, np.uint8, copy=False)
        else:
            arr = np.array(img, np.uint8)
    elif isinstance(img, np.ndarray):
        arr = img
    else:
        raise TypeError(f"img should be PIL Image or ndarray. Got {type(img)}")

    if arr.ndim == 2:
        arr = arr[:, :, None]
        
    if arr.dtype == np.uint8 and hasattr(tp, 'vision_to_tensor'):
        if not arr.flags['C_CONTIGUOUS']:
            arr = np.ascontiguousarray(arr)
        return tp.vision_to_tensor(arr)
        
    # Fallback to standard ToTensor logic (simplified here)
    arr = arr.transpose((2, 0, 1))
    arr = np.ascontiguousarray(arr)
    t = tp.tensor(arr)
    if t.dtype == tp.uint8:
        t = t.to(tp.float32).div_(255.0)
    return t

def from_file(path, backend=None):
    """
    Read an image from path and return a Tensor.
    """
    from .datasets import default_loader
    # default_loader respects global backend settings
    img = default_loader(path)
    return from_image(img)

class Resize:
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        if _is_numpy_image(img):
            if not _OPENCV_AVAILABLE:
                raise ImportError("OpenCV is required for numpy array resizing")
            
            h, w = img.shape[:2]
            if isinstance(self.size, int):
                if (w <= h and w == self.size) or (h <= w and h == self.size):
                    return img
                if w < h:
                    ow = self.size
                    oh = int(self.size * h / w)
                else:
                    oh = self.size
                    ow = int(self.size * w / h)
                
                cv_interp = _pil_interp_to_cv2(self.interpolation)
                return cv2.resize(img, (ow, oh), interpolation=cv_interp)
            else:
                 return cv2.resize(img, self.size[::-1], interpolation=cv2.INTER_LINEAR)

        if not isinstance(img, Image.Image):
            raise TypeError("img should be PIL Image or NumPy array")
        
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size[::-1], self.interpolation)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, self.interpolation)

class CenterCrop:
    def __init__(self, size):
        if isinstance(size, collections.abc.Sequence):
            self.size = size
        else:
            self.size = (int(size), int(size))
        
    def __call__(self, img):
        if _is_numpy_image(img):
            h, w = img.shape[:2]
            th, tw = self.size
            x1 = int(round((w - tw) / 2.))
            y1 = int(round((h - th) / 2.))
            return img[y1:y1+th, x1:x1+tw]

        if not isinstance(img, Image.Image):
             raise TypeError("img should be PIL Image or NumPy array")
             
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th))

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class Normalize:
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        if not isinstance(tensor, tp.Tensor):
             raise TypeError("tensor should be a tensorplay Tensor")
        
        if tensor.ndim < 3:
             raise ValueError("Expected tensor to be a tensor image of size (C, H, W)")
             
        if not self.inplace:
            tensor = tensor.clone()
            
        dtype = tensor.dtype
        mean = tp.tensor(self.mean, dtype=dtype).to(tensor.device)
        std = tp.tensor(self.std, dtype=dtype).to(tensor.device)
        
        mean = mean.reshape(-1, 1, 1)
        std = std.reshape(-1, 1, 1)
        
        tensor.sub_(mean).div_(std)
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class RandomVerticalFlip:
    """Vertically flip the given PIL Image or NumPy array randomly with a given probability."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            if _is_numpy_image(img):
                if not _OPENCV_AVAILABLE:
                     return np.ascontiguousarray(img[::-1, ...])
                return cv2.flip(img, 0)
            elif isinstance(img, Image.Image):
                return img.transpose(Image.FLIP_TOP_BOTTOM)
            else:
                raise TypeError("img should be PIL Image or NumPy array")
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomRotation:
    """Rotate the image by angle."""
    def __init__(self, degrees, interpolation=Image.NEAREST, expand=False, center=None, fill=0):
        if isinstance(degrees, (int, float)):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.interpolation = interpolation
        self.expand = expand
        self.center = center
        self.fill = fill
        
    @staticmethod
    def get_params(degrees):
        angle = random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, img):
        angle = self.get_params(self.degrees)

        if _is_numpy_image(img):
            if not _OPENCV_AVAILABLE:
                raise ImportError("OpenCV is required for numpy array rotation")
            
            h, w = img.shape[:2]
            center = self.center
            if center is None:
                center = (w / 2, h / 2)
            
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            if self.expand:
                # Calculate new bounding box
                cos = np.abs(matrix[0, 0])
                sin = np.abs(matrix[0, 1])
                new_w = int((h * sin) + (w * cos))
                new_h = int((h * cos) + (w * sin))
                matrix[0, 2] += (new_w / 2) - center[0]
                matrix[1, 2] += (new_h / 2) - center[1]
                w, h = new_w, new_h

            cv_interp = _pil_interp_to_cv2(self.interpolation)
            return cv2.warpAffine(img, matrix, (w, h), flags=cv_interp, borderMode=cv2.BORDER_CONSTANT, borderValue=self.fill)

        elif isinstance(img, Image.Image):
            return img.rotate(angle, self.interpolation, self.expand, self.center, fillcolor=self.fill)
        else:
            raise TypeError("img should be PIL Image or NumPy array")

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', interpolation={0}'.format(self.interpolation)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        if self.fill is not None:
            format_string += ', fill={0}'.format(self.fill)
        format_string += ')'
        return format_string

class GaussianBlur:
    """Blurs image with randomly chosen Gaussian blur."""
    def __init__(self, kernel_size, sigma=(0.1, 2.0)):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.sigma = sigma
        
    def __call__(self, img):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        
        if _is_numpy_image(img):
            if not _OPENCV_AVAILABLE:
                 raise ImportError("OpenCV required")
            k_w, k_h = self.kernel_size
            if k_w % 2 == 0: k_w += 1 # CV2 requires odd
            if k_h % 2 == 0: k_h += 1
            return cv2.GaussianBlur(img, (k_w, k_h), sigma)

        elif isinstance(img, Image.Image):
             return img.filter(ImageFilter.GaussianBlur(radius=sigma))
        else:
             raise TypeError("img should be PIL Image or NumPy array")

    def __repr__(self):
        return self.__class__.__name__ + '(kernel_size={0}, sigma={1})'.format(self.kernel_size, self.sigma)

class ColorJitter:
    """Randomly change the brightness, contrast, saturation and hue of an image."""
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=0.5, clip_first_on_zero=False)
        
    def _check_input(self, value, name, center=1, bound=0, clip_first_on_zero=True):
        if isinstance(value, (int, float)):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)):
            if len(value) != 2:
                raise ValueError("If {} is a sequence, it must be of len 2.".format(name))
            if not bound:
                if value[0] < 0 or value[1] < 0:
                    raise ValueError("{} values should be non negative.".format(name))
        else:
            raise TypeError("{} should be a single number or a list/tuple with length 2.".format(name))
        
        if bound != 0: # Check for Hue
            if value[0] < -bound or value[1] > bound:
                raise ValueError("{} values should be between {} and {}.".format(name, -bound, bound))
        
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        transforms = []
        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(lambda img: ImageEnhance.Brightness(img).enhance(brightness_factor))
        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(lambda img: ImageEnhance.Contrast(img).enhance(contrast_factor))
        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(lambda img: ImageEnhance.Color(img).enhance(saturation_factor))
        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            # Hue in PIL is messy, usually requires conversion to HSV. 
            # For simplicity in fallback, we might skip or do a rough implementation.
            # But let's try to do it right if we can, or skip if complex.
            # ImageEnhance doesn't have Hue. 
            pass 

        random.shuffle(transforms)
        return transforms, hue

    def __call__(self, img):
        # Fallback: Convert to PIL if numpy, because ColorJitter is complex in pure CV2
        is_numpy = _is_numpy_image(img)
        if is_numpy:
             img = Image.fromarray(img)

        # Note: Implementing full ColorJitter in PIL requires some work for Hue.
        # Here we implement Brightness, Contrast, Saturation. Hue is often skipped in simple impls.
        
        b = self.brightness
        c = self.contrast
        s = self.saturation
        h = self.hue
        
        # Simplified execution for PIL fallback
        transforms_list = []
        if b is not None:
             val = random.uniform(b[0], b[1])
             transforms_list.append(lambda x: ImageEnhance.Brightness(x).enhance(val))
        if c is not None:
             val = random.uniform(c[0], c[1])
             transforms_list.append(lambda x: ImageEnhance.Contrast(x).enhance(val))
        if s is not None:
             val = random.uniform(s[0], s[1])
             transforms_list.append(lambda x: ImageEnhance.Color(x).enhance(val))
        if h is not None:
             val = random.uniform(h[0], h[1])
             def adjust_hue(x):
                  x = np.array(x.convert('HSV'))
                  x[:, :, 0] = (x[:, :, 0].astype(int) + int(val * 255)) % 255
                  return Image.fromarray(x, 'HSV').convert('RGB')
             transforms_list.append(adjust_hue)

        random.shuffle(transforms_list)
        for t in transforms_list:
             img = t(img)

        if is_numpy:
             img = np.array(img)
        
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'

class Pad:
    """Pad the given image on all sides."""
    def __init__(self, padding, fill=0, padding_mode='constant'):
        if isinstance(padding, int):
            self.padding = (padding, padding, padding, padding)
        elif len(padding) == 2:
            self.padding = (padding[0], padding[1], padding[0], padding[1])
        else:
            self.padding = padding
            
        self.fill = fill
        self.padding_mode = padding_mode
        
    def __call__(self, img):
        l, t, r, b = self.padding
        if _is_numpy_image(img):
            return cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=self.fill)
        elif isinstance(img, Image.Image):
            return ImageOps.expand(img, (l, t, r, b), fill=self.fill)
        else:
            raise TypeError("img should be PIL Image or NumPy array")

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.format(self.padding, self.fill, self.padding_mode)


