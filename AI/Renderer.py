import pygame
from FrameInfo import FrameInfo
from Constants import *

WHITE = (255, 255, 255)


class Renderer:
    def __init__(self, arena_size, never_display = False) -> None:
        pygame.init()
        self.arena_size = arena_size
        self.never_display = never_display
        self.rendering_surface = pygame.surface.Surface(arena_size)
        self.output_surface = None if never_display else pygame.display.set_mode((NN_VIEW_WIDTH, NN_VIEW_HEIGHT))
    
    def render_frame(self, frame_info: FrameInfo):
        if (not self.never_display):
            for _ in pygame.event.get():
                pass
        
        self.rendering_surface.fill(WHITE)

        for blob in frame_info.blobs:
            blob.x += ARENA_WIDTH/2
            pygame.draw.circle(self.rendering_surface, BLOB_COLORS[blob.type], (blob.x, blob.y), BLOB_RADII[blob.type])

        pygame.transform.scale(self.rendering_surface, (NN_VIEW_WIDTH, NN_VIEW_HEIGHT), self.output_surface)
        
    
    def display_frame(self):
        if (self.never_display):
            return
        pygame.display.flip()

    def get_pixels(self):
        return pygame.surfarray.array3d(self.output_surface)[:,:,0]

    def close_window(self):
        pygame.display.quit()
        pygame.quit()
    

