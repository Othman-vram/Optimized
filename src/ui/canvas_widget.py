"""
High-performance canvas widget for tissue fragment visualization
"""

import numpy as np
from typing import List, Optional, Tuple
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPoint, QRect
from PyQt6.QtGui import (QPainter, QPixmap,QImage, QPen, QBrush, QColor, 
                        QMouseEvent, QWheelEvent, QPaintEvent, QResizeEvent)
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
import cv2

from ..core.fragment import Fragment

class CanvasWidget(QOpenGLWidget):
    """High-performance OpenGL canvas for tissue fragment display"""
    
    fragment_selected = pyqtSignal(str)  # fragment_id
    fragment_moved = pyqtSignal(str, float, float)  # fragment_id, x, y
    viewport_changed = pyqtSignal(float, float, float)  # zoom, pan_x, pan_y
    
    def __init__(self):
        super().__init__()
        self.fragments: List[Fragment] = []
        self.selected_fragment_id: Optional[str] = None
        
        # Viewport state
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.min_zoom = 0.01
        self.max_zoom = 50.0
        
        # Interaction state
        self.is_panning = False
        self.is_dragging_fragment = False
        self.last_mouse_pos = QPoint()
        self.dragged_fragment_id: Optional[str] = None
        self.drag_offset = QPoint()
        
        # Rendering
        self.composite_pixmap: Optional[QPixmap] = None
        self.cached_composite: Optional[QPixmap] = None
        self.low_quality_composite: Optional[QPixmap] = None
        self.needs_redraw = True
        self.needs_full_redraw = True
        self.dirty_regions = set()
        
        # Performance settings
        self.high_quality_mode = True
        self.lod_threshold = 0.5  # Use LOD when zoom < 0.5
        
        # Performance timers
        self.render_timer = QTimer()
        self.render_timer.setSingleShot(True)
        self.render_timer.timeout.connect(self.update_canvas)
        
        self.fast_render_timer = QTimer()
        self.fast_render_timer.setSingleShot(True)
        self.fast_render_timer.timeout.connect(self.fast_update_canvas)
        
        # Background rendering
        from PyQt6.QtCore import QThread, QMutex
        self.render_mutex = QMutex()
        self.background_render_thread = None
        
        # Setup
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
    def update_fragments(self, fragments: List[Fragment]):
        """Update the fragment list and trigger redraw"""
        self.fragments = fragments
        self.needs_redraw = True
        self.needs_full_redraw = True
        self.cached_composite = None
        self.dirty_regions.clear()
        self.schedule_redraw()
        
    def set_selected_fragment(self, fragment_id: Optional[str]):
        """Set the selected fragment"""
        self.selected_fragment_id = fragment_id
        self.needs_redraw = True
        self.schedule_redraw()
        
    def schedule_redraw(self, fast: bool = False):
        """Schedule a canvas redraw with debouncing"""
        if fast and self.is_dragging_fragment:
            # Fast rendering during drag operations
            if not self.fast_render_timer.isActive():
                self.fast_render_timer.start(8)  # ~120 FPS for responsiveness
        else:
            # Normal rendering
            if not self.render_timer.isActive():
                self.render_timer.start(33)  # 30 FPS for better performance
                
    def fast_update_canvas(self):
        """Fast update for interactive operations"""
        if self.is_dragging_fragment:
            self.render_low_quality_composite()
        self.update()
            
    def update_canvas(self):
        """Update the canvas rendering"""
        if self.needs_redraw:
            if self.needs_full_redraw or not self.cached_composite:
                self.render_composite()
                self.needs_full_redraw = False
            else:
                self.update_dirty_regions()
            self.needs_redraw = False
        self.update()
        
    def render_composite(self):
        """Render all fragments into a composite image"""
        if not self.fragments:
            self.composite_pixmap = None
            return
            
        # Calculate canvas bounds
        canvas_bounds = self.calculate_canvas_bounds()
        if not canvas_bounds:
            return
            
        min_x, min_y, max_x, max_y = canvas_bounds
        canvas_width = int(max_x - min_x)
        canvas_height = int(max_y - min_y)
        
        # Create composite image with alpha channel
        composite = np.zeros((canvas_height, canvas_width, 4), dtype=np.uint8)
        
        # Render each visible fragment
        for fragment in self.fragments:
            if not fragment.visible or fragment.image_data is None:
                continue
                
            self.render_fragment_to_composite(fragment, composite, min_x, min_y)
            
        # Convert to QPixmap
        height, width, channel = composite.shape
        bytes_per_line = 4 * width
        q_image = QPixmap.fromImage(
            QImage(composite.data, width, height, bytes_per_line, QImage.Format.Format_RGBA8888)
        )
        self.composite_pixmap = q_image
        
    def render_fragment_to_composite(self, fragment: Fragment, composite: np.ndarray, 
                                   offset_x: float, offset_y: float):
        """Render a single fragment to the composite image"""
        transformed_image = fragment.get_transformed_image()
        if transformed_image is None:
            return
            
        # Calculate position in composite
        frag_x = int(fragment.x - offset_x)
        frag_y = int(fragment.y - offset_y)
        
        # Get fragment dimensions
        frag_h, frag_w = transformed_image.shape[:2]
        comp_h, comp_w = composite.shape[:2]
        
        # Calculate intersection with composite bounds
        src_x1 = max(0, -frag_x)
        src_y1 = max(0, -frag_y)
        src_x2 = min(frag_w, comp_w - frag_x)
        src_y2 = min(frag_h, comp_h - frag_y)
        
        dst_x1 = max(0, frag_x)
        dst_y1 = max(0, frag_y)
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)
        
        # Check if there's any overlap
        if src_x2 <= src_x1 or src_y2 <= src_y1:
            return
            
        # Extract the overlapping region
        fragment_region = transformed_image[src_y1:src_y2, src_x1:src_x2]
        
        # Alpha blending with proper transparency support
        if fragment_region.shape[2] == 4:  # RGBA
            # Extract alpha channel from fragment
            frag_alpha = fragment_region[:, :, 3:4] / 255.0 * fragment.opacity
            frag_rgb = fragment_region[:, :, :3]
            
            # Get existing composite region
            comp_region = composite[dst_y1:dst_y2, dst_x1:dst_x2]
            comp_alpha = comp_region[:, :, 3:4] / 255.0
            comp_rgb = comp_region[:, :, :3]
            
            # Alpha blending formula: C = αA*A + (1-αA)*αB*B / (αA + (1-αA)*αB)
            # Simplified for overlay: C = αA*A + (1-αA)*B
            out_alpha = frag_alpha + (1 - frag_alpha) * comp_alpha
            
            # Avoid division by zero
            mask = out_alpha[:, :, 0] > 0
            out_rgb = np.zeros_like(frag_rgb, dtype=np.float32)
            out_rgb[mask, :] = (frag_alpha[mask, :] * frag_rgb[mask, :] + 
                               (1 - frag_alpha[mask, :]) * comp_rgb[mask, :])
            
            # Update composite
            composite[dst_y1:dst_y2, dst_x1:dst_x2, :3] = np.clip(out_rgb, 0, 255).astype(np.uint8)
            composite[dst_y1:dst_y2, dst_x1:dst_x2, 3:4] = np.clip(out_alpha * 255, 0, 255).astype(np.uint8)
        else:
            # Fallback for RGB images (shouldn't happen with new loader)
            alpha = fragment.opacity
            composite[dst_y1:dst_y2, dst_x1:dst_x2, :3] = (
                alpha * fragment_region + 
                (1 - alpha) * composite[dst_y1:dst_y2, dst_x1:dst_x2, :3]
            ).astype(np.uint8)
            composite[dst_y1:dst_y2, dst_x1:dst_x2, 3] = 255  # Full alpha
            
    def calculate_canvas_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """Calculate the bounding box of all visible fragments"""
        visible_fragments = [f for f in self.fragments if f.visible and f.image_data is not None]
        if not visible_fragments:
            return None
            
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')
        
        for fragment in visible_fragments:
            bbox = fragment.get_bounding_box()
            min_x = min(min_x, bbox[0])
            min_y = min(min_y, bbox[1])
            max_x = max(max_x, bbox[0] + bbox[2])
            max_y = max(max_y, bbox[1] + bbox[3])
            
        return (min_x, min_y, max_x, max_y)
        
    def paintEvent(self, event: QPaintEvent):
        """Paint the canvas"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Fill background
        painter.fillRect(self.rect(), QColor(42, 42, 42))
        
        if self.composite_pixmap is None:
            return
            
        # Apply viewport transformation
        painter.save()
        painter.scale(self.zoom, self.zoom)
        painter.translate(self.pan_x, self.pan_y)
        
        # Draw composite image
        canvas_bounds = self.calculate_canvas_bounds()
        if canvas_bounds:
            min_x, min_y, _, _ = canvas_bounds
            painter.drawPixmap(int(min_x), int(min_y), self.composite_pixmap)
            
        # Draw selection outlines
        self.draw_selection_outlines(painter)
        
        painter.restore()
        
    def draw_selection_outlines(self, painter: QPainter):
        """Draw selection outlines for fragments"""
        for fragment in self.fragments:
            if not fragment.visible or fragment.image_data is None:
                continue
                
            # Draw selection outline
            if fragment.selected or fragment.id == self.selected_fragment_id:
                pen = QPen(QColor(74, 144, 226), 2.0 / self.zoom)
                painter.setPen(pen)
                painter.setBrush(QBrush())
                
                bbox = fragment.get_bounding_box()
                rect = QRect(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                painter.drawRect(rect)
                
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press events"""
        if event.button() == Qt.MouseButton.LeftButton:
            world_pos = self.screen_to_world(event.pos())
            clicked_fragment = self.get_fragment_at_position(world_pos.x(), world_pos.y())
            
            if clicked_fragment:
                # Select and start dragging fragment
                self.fragment_selected.emit(clicked_fragment.id)
                self.is_dragging_fragment = True
                self.dragged_fragment_id = clicked_fragment.id
                self.drag_offset = QPoint(
                    int(world_pos.x() - clicked_fragment.x),
                    int(world_pos.y() - clicked_fragment.y)
                )
            else:
                # Start panning
                self.is_panning = True
                
        elif event.button() == Qt.MouseButton.MiddleButton:
            # Always pan with middle button
            self.is_panning = True
            
        self.last_mouse_pos = event.pos()
        
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move events"""
        if self.is_dragging_fragment and self.dragged_fragment_id:
            # Move fragment
            world_pos = self.screen_to_world(event.pos())
            new_x = world_pos.x() - self.drag_offset.x()
            new_y = world_pos.y() - self.drag_offset.y()
            
            self.fragment_moved.emit(self.dragged_fragment_id, new_x, new_y)
            
        elif self.is_panning:
            # Pan viewport
            delta = event.pos() - self.last_mouse_pos
            self.pan_x += delta.x() / self.zoom
            self.pan_y += delta.y() / self.zoom
            self.viewport_changed.emit(self.zoom, self.pan_x, self.pan_y)
            self.schedule_redraw()
            
        self.last_mouse_pos = event.pos()
        
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release events"""
        self.is_panning = False
        self.is_dragging_fragment = False
        self.dragged_fragment_id = None
        
    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel events for zooming"""
        # Get mouse position in world coordinates before zoom
        mouse_world_before = self.screen_to_world(event.position().toPoint())
        
        # Calculate zoom factor
        zoom_factor = 1.2 if event.angleDelta().y() > 0 else 1.0 / 1.2
        new_zoom = np.clip(self.zoom * zoom_factor, self.min_zoom, self.max_zoom)
        
        if new_zoom != self.zoom:
            # Update zoom
            self.zoom = new_zoom
            
            # Adjust pan to keep mouse position fixed
            mouse_world_after = self.screen_to_world(event.position().toPoint())
            self.pan_x += mouse_world_before.x() - mouse_world_after.x()
            self.pan_y += mouse_world_before.y() - mouse_world_after.y()
            
            # Invalidate cache on zoom change
            self.needs_full_redraw = True
            self.cached_composite = None
            
            self.viewport_changed.emit(self.zoom, self.pan_x, self.pan_y)
            self.schedule_redraw()
            
    def resizeEvent(self, event: QResizeEvent):
        """Handle resize events"""
        super().resizeEvent(event)
        self.needs_full_redraw = True
        self.cached_composite = None
        self.schedule_redraw()
        
    def screen_to_world(self, screen_pos: QPoint) -> QPoint:
        """Convert screen coordinates to world coordinates"""
        world_x = (screen_pos.x() / self.zoom) - self.pan_x
        world_y = (screen_pos.y() / self.zoom) - self.pan_y
        return QPoint(int(world_x), int(world_y))
        
    def world_to_screen(self, world_pos: QPoint) -> QPoint:
        """Convert world coordinates to screen coordinates"""
        screen_x = (world_pos.x() + self.pan_x) * self.zoom
        screen_y = (world_pos.y() + self.pan_y) * self.zoom
        return QPoint(int(screen_x), int(screen_y))
        
    def get_fragment_at_position(self, x: float, y: float) -> Optional[Fragment]:
        """Get the topmost fragment at the given position"""
        # Check fragments in reverse order (top to bottom)
        for fragment in reversed(self.fragments):
            if fragment.visible and fragment.contains_point(x, y):
                return fragment
        return None
        
    def zoom_to_fit(self):
        """Zoom to fit all visible fragments"""
        canvas_bounds = self.calculate_canvas_bounds()
        if not canvas_bounds:
            return
            
        min_x, min_y, max_x, max_y = canvas_bounds
        content_width = max_x - min_x
        content_height = max_y - min_y
        
        if content_width <= 0 or content_height <= 0:
            return
            
        # Calculate zoom to fit with padding
        widget_width = self.width()
        widget_height = self.height()
        
        zoom_x = widget_width / content_width
        zoom_y = widget_height / content_height
        self.zoom = min(zoom_x, zoom_y) * 0.9  # 90% to add padding
        
        # Center the content
        content_center_x = (min_x + max_x) / 2
        content_center_y = (min_y + max_y) / 2
        
        self.pan_x = (widget_width / 2 / self.zoom) - content_center_x
        self.pan_y = (widget_height / 2 / self.zoom) - content_center_y
        
        self.viewport_changed.emit(self.zoom, self.pan_x, self.pan_y)
        self.schedule_redraw()
        
    def zoom_to_100(self):
        """Reset zoom to 100%"""
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.viewport_changed.emit(self.zoom, self.pan_x, self.pan_y)
        self.schedule_redraw()
        
    def export_view(self) -> QPixmap:
        """Export the current view as a pixmap"""
        pixmap = QPixmap(self.size())
        painter = QPainter(pixmap)
        self.render(painter)
        painter.end()
        return pixmap