import pygame
from FrameInfo import FrameInfo
from Constants import *

WHITE = (255, 255, 255)

OUTPUT_SIZE = (int(ARENA_WIDTH/2), int(ARENA_HEIGHT/2 + NEXT_BLOB_HEIGHT))

class Renderer:
    def __init__(self, never_display = False, window_title = "Blob Game") -> None:
        self.never_display = never_display
        self.rendering_surface = pygame.surface.Surface(ACTUAL_ARENA_SIZE)
        self.scaled_surface = pygame.surface.Surface((NN_VIEW_WIDTH, NN_VIEW_HEIGHT))
        self.output_surface = None if never_display else pygame.display.set_mode(OUTPUT_SIZE)
        pygame.display.set_caption(window_title)
    
    def render_frame(self, frame_info: FrameInfo, current_t: float):
        self.rendering_surface.fill(WHITE)

        for blob in frame_info.blobs:
            pygame.draw.circle(self.rendering_surface, BLOB_COLORS[blob.type], (blob.x + ARENA_WIDTH/2, blob.y + 2*NEXT_BLOB_HEIGHT), BLOB_RADII[blob.type])

        if (frame_info.can_drop):
            pygame.draw.circle(self.rendering_surface, BLOB_COLORS[frame_info.current_blob], (ARENA_WIDTH * current_t, NEXT_BLOB_HEIGHT*1.5), BLOB_RADII[frame_info.current_blob])
        pygame.draw.circle(self.rendering_surface, BLOB_COLORS[frame_info.next_blob], (ARENA_WIDTH * current_t, NEXT_BLOB_HEIGHT*0.5), BLOB_RADII[frame_info.next_blob])
        
        pygame.transform.scale(self.rendering_surface, (NN_VIEW_WIDTH, NN_VIEW_HEIGHT), self.scaled_surface)


    
    def display_frame(self):
        if (self.never_display):
            return
        
        for _ in pygame.event.get():
            pass

        pygame.transform.scale(self.scaled_surface, OUTPUT_SIZE, self.output_surface)
        pygame.display.flip()

    def get_pixels(self):
        return pygame.surfarray.array3d(self.scaled_surface)[:,:,0]

    @staticmethod
    def init():
        pygame.init()

    @staticmethod
    def quit():
        pygame.display.quit()
        pygame.quit()
    

