"""
Fragment data structure and management
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass, field
import uuid

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
    rotation: int = 0  # 0, 90, 180, 270 degrees
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
        if self.image_data is None:
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
            
        # Apply rotation
        if self.rotation != 0:
            k = self.rotation // 90
            img = np.rot90(img, k)
            
        # Cache the result
        self.transformed_image_cache = img
        self.cache_valid = True
            
        return img
        
    def invalidate_cache(self):
        """Invalidate the transformed image cache"""
        self.cache_valid = False
        self.transformed_image_cache = None
    
    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """Get the bounding box of the transformed fragment (x, y, width, height)"""
        if self.image_data is None:
            return (self.x, self.y, 0, 0)
            
        transformed_img = self.get_transformed_image()
        height, width = transformed_img.shape[:2]
        
        return (self.x, self.y, width, height)
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is within the fragment bounds"""
        bbox_x, bbox_y, bbox_w, bbox_h = self.get_bounding_box()
        return (bbox_x <= x <= bbox_x + bbox_w and 
                bbox_y <= y <= bbox_y + bbox_h)
    
    def reset_transform(self):
        """Reset all transformations to default"""
        self.rotation = 0
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