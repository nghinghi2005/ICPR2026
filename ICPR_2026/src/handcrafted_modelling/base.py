from abc import ABC, abstractmethod
from typing import Optional, Union
import cv2
import numpy as np

class UpScaler(ABC):
    @abstractmethod
    def __call__(self, ratio: Optional[Union[float, int]]):
        pass

class NEAREST_NEIGHBOR(UpScaler):
    def __init__(self, img: np.ndarray, scale: Optional[Union[float, int]]):
        self.img = img
        self.scale = scale
    
    def __call__(self):
        return cv2.resize(self.img, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)

class BILINEAR(UpScaler):
    def __init__(self, img: np.ndarray, scale: Optional[Union[float, int]]):
        self.img = img
        self.scale = scale
    
    def __call__(self):
        return cv2.resize(self.img, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)

class BICUBIC(UpScaler):
    def __init__(self, img: np.ndarray, scale: Optional[Union[float, int]]):
        self.img = img
        self.scale = scale
    
    def __call__(self):
        return cv2.resize(self.img, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)

class LANCZOS(UpScaler):
    def __init__(self, img: np.ndarray, scale: Optional[Union[float, int]]):
        self.img = img
        self.scale = scale
    
    def __call__(self):
        return cv2.resize(self.img, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LANCZOS4)

class LANCZOS_2(UpScaler):
    def __init__(self, img: np.ndarray, scale: Optional[Union[float, int]]):
        self.img = img
        self.scale = scale
    
    def __call__(self):
        return cv2.resize(self.img, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LANCZOS4)

class BILATERAL_ADAPTIVE(UpScaler):
    def __init__(self, img: np.ndarray, scale: Optional[Union[float, int]]):
        self.img = img
        self.scale = scale
    
    def __call__(self):
        return cv2.resize(self.img, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LANCZOS4)

class MULTISCALE(UpScaler):
    def __init__(self, img: np.ndarray, scale: Optional[Union[float, int]]):
        self.img = img
        self.scale = scale
    
    def __call__(self):
        return cv2.resize(self.img, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LANCZOS4)

class DENOISE_CLAHE_SHARPEN(UpScaler):
    def __init__(self, img: np.ndarray, scale: Optional[Union[float, int]]):
        self.img = img
        self.scale = scale
    
    def __call__(self):
        # Phóng to
        upscaled = cv2.resize(self.img, None, fx=self.scale, fy=self.scale, 
                            interpolation=cv2.INTER_LANCZOS4)
        
        # Giảm nhiễu
        denoised = cv2.fastNlMeansDenoisingColored(upscaled, None, 10, 10, 7, 21)
        
        # CLAHE trên mỗi kênh màu
        lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        # Sharpening
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        result = cv2.filter2D(enhanced, -1, kernel)
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result

class EDGE_ENHANCE(UpScaler):
    def __init__(self, img: np.ndarray, scale: Optional[Union[float, int]]):
        self.img = img
        self.scale = scale
    
    def __call__(self):
        upscaled = cv2.resize(self.img, None, fx=self.scale, fy=self.scale, 
                            interpolation=cv2.INTER_LANCZOS4)
        
        # Edge-preserving filter
        filtered = cv2.edgePreservingFilter(upscaled, flags=1, sigma_s=60, sigma_r=0.4)
        
        # Tăng contrast
        lab = cv2.cvtColor(filtered, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        result = cv2.merge([l, a, b])
        result = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)
        return result

class FREQ_ENHANCE(UpScaler):
    def __init__(self, img: np.ndarray, scale: Optional[Union[float, int]]):
        self.img = img
        self.scale = scale
    
    def __call__(self):
        upscaled = cv2.resize(self.img, None, fx=self.scale, fy=self.scale, 
                            interpolation=cv2.INTER_LANCZOS4)
        
        # Xử lý từng kênh màu
        result = np.zeros_like(upscaled)
        for i in range(3):
            # FFT
            f = np.fft.fft2(upscaled[:,:,i])
            fshift = np.fft.fftshift(f)
            
            # Tăng cường tần số cao (high-pass filter)
            rows, cols = upscaled.shape[:2]
            crow, ccol = rows//2, cols//2
            
            # Tạo mask tăng cường tần số cao
            mask = np.ones((rows, cols), np.float32)
            r = 30
            center = [crow, ccol]
            x, y = np.ogrid[:rows, :cols]
            mask_area = (x - center[0])**2 + (y - center[1])**2 <= r*r
            mask[mask_area] = 0.5  # Giảm tần số thấp
            
            fshift = fshift * mask
            
            # IFFT
            f_ishift = np.fft.ifftshift(fshift)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.abs(img_back)
            
            result[:,:,i] = img_back
        
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result

class HandCrafter(ABC):
    METHOD_MAPPING = {
        "nearest": NEAREST_NEIGHBOR,
        "bilinear": BILINEAR,
        "bicubic": BICUBIC,
        "lanczos": LANCZOS,
        "lanczos2": LANCZOS_2,
        "denoise_clahe_sharpen": DENOISE_CLAHE_SHARPEN,
        "edge_enhance": EDGE_ENHANCE,
        "freq_enhance": FREQ_ENHANCE,
        "bilateral_adaptive": BILATERAL_ADAPTIVE,
        "multiscale": MULTISCALE
    }

    def use_method(self, method: str, img: np.ndarray, scale: Optional[Union[float, int]]):
        if method not in self.METHOD_MAPPING:
            raise ValueError(f"Method {method} not found in one of these {self.METHOD_MAPPING.keys()}")
        else:
            return self.METHOD_MAPPING[method](img=img, scale=scale)
