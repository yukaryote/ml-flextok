from PIL import Image, ImageDraw
from math import sqrt
from itertools import product
from multiprocessing import Pool

class SyntheticFaceGenerator:
    """
    Generates simple images of faces with varying attributes.
    Each image is a 256x256 pixel image with a colored shape representing a face, with different shapes/colors for eyes/nose/mouth
    Attributes include color (red, green, blue) and shape (circle, square, triangle).
    """

    def __init__(self, size=(256, 256)):
        self.size = size
        self.colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
        self.face_shapes = ['circle', 'square', 'tall_rectangle', 'wide_rectangle']
        self.eye_shapes = ['circle', 'square']
        self.nose_shapes = ['square', 'circle', 'tall_rectangle']
        self.mouth_shapes = ['smile', 'frown', 'surprised', 'straight']
        self.features = ['face', 'eyes', 'nose', 'mouth']

        self.face_max_width = self.size[1] * 0.8
        self.face_max_height = self.size[0] * 0.8
        self.face_square_topleft = ((self.size[0] - self.face_max_height) // 2, (self.size[1] - self.face_max_width) // 2)
        self.face_square_bottomright = (self.face_square_topleft[0] + self.face_max_height, self.face_square_topleft[1] + self.face_max_width)

        self.left_eye_pos = (2 * self.size[0] // 5, 2 * self.size[1] // 5)
        self.right_eye_pos = (3 * self.size[0] // 5, 2 * self.size[1] // 5)
        self.eye_radius = 16
        self.nose_pos = (self.size[0] // 2, self.size[1] // 2)
        self.nose_radius = 10
        self.mouth_pos = (self.size[0] // 2, 2 * self.size[1] // 3)
        self.mouth_radius = 16

    def __len__(self):
        """
        Number of possible faces, minus invalid combinations (e.g., some/all features same color as the face).
        """
        return len(self.eye_shapes) * \
            len(self.face_shapes) * \
            len(self.nose_shapes) * \
            len(self.mouth_shapes) * \
            len(self.colors)**4 - \
            (len(self.colors) + len(self.colors)*3 - len(self.colors)*3)  # all same color as face + 2 features same color + 1 feature same color
    
    def draw_face(self, face_dict: dict, save_path=None) -> Image.Image:
        """
        Draws face using PIL based on the attributes in face_dict.
        Args:
            face_dict: Dictionary with keys 'face', 'eyes', 'nose', 'mouth', each mapping to a dict with 'shape' and 'color'.
            save_path: Optional path to save the generated image.
        Returns:
            PIL Image of the drawn face.
        """
        BLACK, WHITE = (0, 0, 0), (255, 255, 255)

        im = Image.new("RGB", size=self.size, color=WHITE)
        draw = ImageDraw.Draw(im)
        for face_feature, attrs in face_dict.items():
            shape, color = attrs['shape'], attrs['color']
            if face_feature == 'face':
                if shape == 'circle':
                    draw.circle(xy=(self.size[0]//2, self.size[1]//2), radius=self.face_max_height//2, fill=color, outline=BLACK)
                elif shape == 'square':
                    draw.rectangle(xy=(self.face_square_topleft, self.face_square_bottomright), fill=color, outline=BLACK)
                elif shape == 'tall_rectangle':
                    draw.rectangle([(64, 16), (192, 240)], fill=color, outline=BLACK)
                elif shape == 'wide_rectangle':
                    draw.rectangle([(16, 64), (240, 192)], fill=color, outline=BLACK)
            elif face_feature == 'eyes':
                if shape == 'circle':
                    draw.circle(xy=self.left_eye_pos, radius=self.eye_radius, fill=color, outline=BLACK)  # left eye
                    draw.circle(xy=self.right_eye_pos, radius=self.eye_radius, fill=color, outline=BLACK)  # right eye
                elif shape == 'square':
                    eye_top_height = self.left_eye_pos[1] - self.eye_radius
                    eye_bottom_height = self.left_eye_pos[1] + self.eye_radius
                    left_eye_x = self.left_eye_pos[0] - self.eye_radius
                    right_eye_x = self.right_eye_pos[0] - self.eye_radius
                    draw.rectangle([(left_eye_x, eye_top_height), (left_eye_x + 2*self.eye_radius, eye_bottom_height)], fill=color, outline=BLACK)  # left eye
                    draw.rectangle([(right_eye_x, eye_top_height), (right_eye_x + 2*self.eye_radius, eye_bottom_height)], fill=color, outline=BLACK)  # right eye
            elif face_feature == 'nose':
                if shape == 'square':
                    top_xy = (self.nose_pos[0] - self.nose_radius, self.nose_pos[1] - self.nose_radius)
                    bottom_xy = (self.nose_pos[0] + self.nose_radius, self.nose_pos[1] + self.nose_radius)
                    draw.rectangle([top_xy, bottom_xy], fill=color, outline=BLACK)
                elif shape == 'wide_rectangle':
                    draw.rectangle([(112, 128), (144, 160)], fill=color, outline=BLACK)
                elif shape == 'circle':
                    draw.circle(xy=self.nose_pos, radius=self.nose_radius, fill=color, outline=BLACK)
                elif shape == 'tall_rectangle':
                    top_xy = (self.nose_pos[0] - self.nose_radius, self.nose_pos[1] - self.nose_radius*2)
                    bottom_xy = (self.nose_pos[0] + self.nose_radius, self.nose_pos[1] + self.nose_radius*2)
                    draw.rectangle([top_xy, bottom_xy], fill=color, outline=BLACK)
                elif shape == 'triangle':
                    draw.rectangle([(120, 128), (136, 160)], fill=color, outline=BLACK)
            elif face_feature == 'mouth':
                if shape == 'smile':
                    draw.arc([(96, 140), (160, 180)], start=0, end=180, fill=color, width=4)
                elif shape == 'frown':
                    draw.arc([(96, 160), (160, 200)], start=180, end=360, fill=color, width=4)
                elif shape == 'surprised':
                    draw.circle(xy=self.mouth_pos, radius=self.mouth_radius, fill=color, outline=BLACK)
                elif shape == 'straight':
                    draw.line([(96, 170), (160, 170)], fill=color, width=4)
        if save_path is not None:
            im.save(save_path)
        return im 

    def generate_face_pool_worker(self, args):
        face_attrs, save_path = args
        for face_color in self.colors:
            for eye_color in self.colors:
                for nose_color in self.colors:
                    for mouth_color in self.colors:
                        # Skip invalid combinations where some/all features have same color as face
                        if (eye_color == face_color) or (nose_color == face_color) or (mouth_color == face_color):
                            continue
                        face_dict = {
                            'face': {'shape': face_attrs[0], 'color': face_color},
                            'eyes': {'shape': face_attrs[1], 'color': eye_color},
                            'nose': {'shape': face_attrs[2], 'color': nose_color},
                            'mouth': {'shape': face_attrs[3], 'color': mouth_color},
                        }
                        if save_path is not None:
                            img = self.draw_face(face_dict)
                            filename = f"face_{face_attrs[0]}_{face_color}_eyes_{face_attrs[1]}_{eye_color}_nose_{face_attrs[2]}_{nose_color}_mouth_{face_attrs[3]}_{mouth_color}.png"
                            img.save(f"{save_path}/{filename}")

    def generate_all_faces(self, save_dir=None):
        all_face_attrs = product(self.face_shapes, self.eye_shapes, self.nose_shapes, self.mouth_shapes)
        with Pool() as pool:
            pool.map(self.generate_face_pool_worker, [(face_attrs, save_dir) for face_attrs in all_face_attrs])


if __name__ == "__main__":
    generator = SyntheticFaceGenerator()
    face_attrs = {
        'face': {'shape': 'wide_rectangle', 'color': 'yellow'},
        'eyes': {'shape': 'circle', 'color': 'blue'},
        'nose': {'shape': 'square', 'color': 'red'},
        'mouth': {'shape': 'straight', 'color': 'green'},
    }
    img = generator.draw_face(face_attrs, save_path="test_face.png")
    save_path = "/home/iyu/ml-flextok/data/synth_faces/images"
    import os
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    all_faces = generator.generate_all_faces(save_dir=save_path)