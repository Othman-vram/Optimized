"""
Fragment data structure and management
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass, field
import uuid
import cv2

@dataclass
class Fragment:
    """Represents a tissue fragment with its image data and transformation state"""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    image_data: Optional[np.ndarray] = None
    original_image_data: Optional[np.ndarray] = None
    transformed_image_cache: Optional[np.ndarray] = None
    cache_valid: bool = False
    
    # Position and transformation
    x: float = 0.0
    y: float = 0.0
    rotation: float = 0.0  # Any angle in degrees
    flip_horizontal: bool = False
    flip_vertical: bool = False
    
    # Display properties
    visible: bool = True
    selected: bool = False
    opacity: float = 1.0
    
    # Metadata
    file_path: str = ""
    original_size: Tuple[int, int] = (0, 0)
    pixel_size: float = 1.0  # microns per pixel
    
    def __post_init__(self):
        """Post-initialization processing"""
        if self.image_data is not None and self.original_image_data is None:
            self.original_image_data = self.image_data.copy()
            self.original_size = (self.image_data.shape[1], self.image_data.shape[0])
            self.cache_valid = False
    
    def get_transformed_image(self) -> np.ndarray:
        """Get the image with current transformations applied"""
        if self.original_image_data is None:
            return None
            
        # Check if cache is valid
        if self.cache_valid and self.transformed_image_cache is not None:
            return self.transformed_image_cache
            
        img = self.original_image_data.copy()
        
        # Apply horizontal flip
        if self.flip_horizontal:
            img = np.fliplr(img)
            
        # Apply vertical flip
        if self.flip_vertical:
            img = np.flipud(img)
            
        # Apply rotation (any angle)
        if abs(self.rotation) > 0.01:  # Only rotate if angle is significant
            img = self._rotate_image(img, self.rotation)
            
        # Cache the result
        self.transformed_image_cache = img
        self.cache_valid = True
        
        # Also update the main image_data for compatibility
        self.image_data = img
            
        return img
        
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by arbitrary angle"""
        if abs(angle) < 0.01:
            return image
            
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new bounding box
        cos_val = abs(rotation_matrix[0, 0])
        sin_val = abs(rotation_matrix[0, 1])
        new_width = int((height * sin_val) + (width * cos_val))
        new_height = int((height * cos_val) + (width * sin_val))
        
        # Adjust rotation matrix for new center
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        # Apply rotation with proper interpolation and border handling
        if len(image.shape) == 3 and image.shape[2] == 4:
            # Handle RGBA images properly
            rotated = cv2.warpAffine(
                image, rotation_matrix, (new_width, new_height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0)  # Transparent background
            )
        else:
            rotated = cv2.warpAffine(
                image, rotation_matrix, (new_width, new_height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )
            
        return rotated
        
    def invalidate_cache(self):
        """Invalidate the transformed image cache"""
        self.cache_valid = False
        self.transformed_image_cache = None
    
    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """Get the tight bounding box of the actual tissue content (x, y, width, height)"""
        if self.image_data is None:
            return (self.x, self.y, 0, 0)
            
        transformed_img = self.get_transformed_image()
        if transformed_img is None:
            return (self.x, self.y, 0, 0)
            
        # Find the actual content bounds by looking for non-transparent pixels
        content_bounds = self._find_content_bounds(transformed_img)
        if content_bounds is None:
            height, width = transformed_img.shape[:2]
            return (self.x, self.y, width, height)
            
        min_x, min_y, max_x, max_y = content_bounds
        content_width = max_x - min_x
        content_height = max_y - min_y
        
        # Adjust position based on content offset
        actual_x = self.x + min_x
        actual_y = self.y + min_y
        
        return (actual_x, actual_y, content_width, content_height)
    
    def _find_content_bounds(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Find the bounding box of non-transparent content in the image"""
        if image is None or image.size == 0:
            return None
            
        # Handle different image formats
        if len(image.shape) == 3:
            if image.shape[2] == 4:  # RGBA
                # Use alpha channel to find content
                alpha = image[:, :, 3]
                mask = alpha > 0
            else:  # RGB
                # Find non-black pixels (assuming black is background)
                gray = np.mean(image, axis=2)
                mask = gray > 10  # Small threshold to handle noise
        else:  # Grayscale
            mask = image > 10
            
        # Find bounding box of content
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return None
            
        min_row, max_row = np.where(rows)[0][[0, -1]]
        min_col, max_col = np.where(cols)[0][[0, -1]]
        
        return (min_col, min_row, max_col + 1, max_row + 1)
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is within the actual tissue content bounds"""
        bbox_x, bbox_y, bbox_w, bbox_h = self.get_bounding_box()
        return (bbox_x <= x <= bbox_x + bbox_w and 
                bbox_y <= y <= bbox_y + bbox_h)
    
    def get_full_image_bounds(self) -> Tuple[float, float, float, float]:
        """Get the full image bounds including transparent areas (x, y, width, height)"""
        if self.image_data is None:
            return (self.x, self.y, 0, 0)
            
        transformed_img = self.get_transformed_image()
        if transformed_img is None:
            return (self.x, self.y, 0, 0)
            
        height, width = transformed_img.shape[:2]
        return (self.x, self.y, width, height)
    
    def reset_transform(self):
        """Reset all transformations to default"""
        self.rotation = 0.0
        self.flip_horizontal = False
        self.flip_vertical = False
        self.invalidate_cache()
        
    def to_dict(self) -> dict:
        """Convert fragment to dictionary for serialization"""
        return {
            'id': self.id,
            'name': self.name,
            'file_path': self.file_path,
            'x': self.x,
            'y': self.y,
            'rotation': self.rotation,
            'flip_horizontal': self.flip_horizontal,
            'flip_vertical': self.flip_vertical,
            'visible': self.visible,
            'opacity': self.opacity,
            'original_size': self.original_size,
            'pixel_size': self.pixel_size
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Fragment':
        """Create fragment from dictionary"""
        fragment = cls()
        for key, value in data.items():
            if hasattr(fragment, key):
                setattr(fragment, key, value)
        return fragment