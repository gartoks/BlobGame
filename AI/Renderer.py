from FrameInfo import FrameInfo
from Constants import *
import numpy as np
import cv2

OUTPUT_SIZE = (int(ARENA_WIDTH / 2), int(ARENA_HEIGHT / 2 + NEXT_BLOB_HEIGHT))


class Renderer:
    def __init__(self, never_display=False, window_title="Blob Game") -> None:
        self.never_display = never_display
        self.rendering_surface = np.zeros(
            (ACTUAL_ARENA_SIZE[1], ACTUAL_ARENA_SIZE[0], 3), dtype=np.uint8
        )
        self.scaled_surface = np.zeros(
            (NN_VIEW_HEIGHT, NN_VIEW_WIDTH, 3), dtype=np.uint8
        )
        self.output_surface = None if never_display else cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        if not never_display:
            cv2.resizeWindow(window_title, OUTPUT_SIZE)
        
        self.window_title = window_title

    def render_frame(self, frame_info: FrameInfo, current_t: float):
        self.rendering_surface.fill(255)

        for blob in frame_info.blobs:
            cv2.circle(
                self.rendering_surface,
                (int(blob.x + ARENA_WIDTH / 2), int(blob.y + 2 * NEXT_BLOB_HEIGHT)),
                int(BLOB_RADII[blob.type]),
                BLOB_COLORS[blob.type],
                thickness=cv2.FILLED,
            )

        if frame_info.can_drop:
            cv2.circle(
                self.rendering_surface,
                (int(ARENA_WIDTH * current_t), int(NEXT_BLOB_HEIGHT * 1.5)),
                int(BLOB_RADII[frame_info.current_blob]),
                BLOB_COLORS[frame_info.current_blob],
                thickness=cv2.FILLED,
            )
        cv2.circle(
            self.rendering_surface,
            (int(ARENA_WIDTH * current_t), int(NEXT_BLOB_HEIGHT * 0.5)),
            int(BLOB_RADII[frame_info.next_blob]),
            BLOB_COLORS[frame_info.next_blob],
            thickness=cv2.FILLED,
        )

        self.scaled_surface = cv2.resize(
            self.rendering_surface, (NN_VIEW_WIDTH, NN_VIEW_HEIGHT)
        )

    def display_frame(self):
        if self.never_display:
            return

        output_surface = cv2.resize(self.scaled_surface, OUTPUT_SIZE, interpolation=cv2.INTER_NEAREST_EXACT)

        cv2.line(
            output_surface,
            (0, NEXT_BLOB_HEIGHT),
            (OUTPUT_SIZE[0], NEXT_BLOB_HEIGHT),
            (0, 0, 255),
            thickness=1,
        )
        cv2.imshow(self.window_title, output_surface)
        cv2.waitKey(1)

    def get_pixels(self):
        return np.transpose(cv2.cvtColor(self.scaled_surface, cv2.COLOR_BGR2GRAY))


    @staticmethod
    def quit():
        cv2.destroyAllWindows()
