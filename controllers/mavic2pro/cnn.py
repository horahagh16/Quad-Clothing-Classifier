import keras.models
import numpy as np
import math
import cv2
from keras.preprocessing.image import img_to_array, array_to_img
from PIL import Image


def rotate_image(image, angle):
    def largest_rotated_rect(w, h, angle):
        quadrant = int(math.floor(angle / (math.pi / 2))) & 3
        sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
        alpha = (sign_alpha % math.pi + math.pi) % math.pi

        bb_w = w * math.cos(alpha) + h * math.sin(alpha)
        bb_h = w * math.sin(alpha) + h * math.cos(alpha)

        gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

        delta = math.pi - alpha - gamma

        length = h if (w < h) else w

        d = length * math.cos(alpha)
        a = d * math.sin(alpha) / math.sin(delta)

        y = a * math.cos(gamma)
        x = y * math.tan(gamma)

        return (
            bb_w - 2 * x,
            bb_h - 2 * y
        )

    def crop_around_center(image, width, height):
        image_size = (image.shape[1], image.shape[0])
        image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

        if width > image_size[0]:
            width = image_size[0]

        if height > image_size[1]:
            height = image_size[1]

        x1 = int(image_center[0] - width * 0.5)
        x2 = int(image_center[0] + width * 0.5)
        y1 = int(image_center[1] - height * 0.5)
        y2 = int(image_center[1] + height * 0.5)

        return image[y1:y2, x1:x2]

    image_height, image_width = image.shape[0:2]
    img = Image.fromarray(image.astype(np.uint8))
    rotated_img = np.array(img.rotate(angle, resample=Image.BICUBIC, expand=True))
    return np.array(crop_around_center(
        rotated_img,
        *largest_rotated_rect(
            image_width,
            image_height,
            math.radians(angle)
        )
    ))


def crop(image):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    x_min, x_max = grey.shape[0]-1, 0
    y_min, y_max = grey.shape[1]-1, 0
    for i in range(grey.shape[0]):
        for j in range(grey.shape[1]):
            if grey[i, j] < 48:
                x_min, x_max = min(x_min, i), max(x_max, i)
                y_min, y_max = min(y_min, j), max(y_max, j)

    output_image = image[x_min:x_max, y_min:y_max]
    return output_image


def invert(img):
    w, h, _ = img.shape
    inverted = (255 - img)
    return np.array(cv2.equalizeHist(cv2.cvtColor(inverted, cv2.COLOR_BGR2GRAY))).reshape((w, h, 1))


def prepare_image(img_array, target_size=(28, 28)):
    # Convert to PIL Image
    img = array_to_img(img_array)
    # Resize the image
    img = img.convert('L').resize(target_size)
    # Convert to array
    img_array = img_to_array(img)
    # Reshape into a single sample with 1 channel
    img_array = np.expand_dims(img_array, axis=0)
    # Prepare pixel data
    img_array = img_array.astype('float32') / 255.0
    return img_array


def load_model():
    return keras.models.load_model('model_100.h5')


def predict_class(model, img, angle_degrees):
    rotated_img = rotate_image(img, angle_degrees)
    show(rotated_img, 'rotated')
    cropped_img = crop(rotated_img)
    show(cropped_img, 'cropped')
    model_input = prepare_image(cropped_img)

    predictions = model.predict(model_input)
    predicted_class_index = np.argmax(predictions[0])
    return predicted_class_index


def show(img, title):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    img_angle = [
        ('img0.png', 0, -55),
        ('img1.png', 1, 0),
        ('img2.png', 3, 76),
        ('img3.png', 4, 50),
        ('img4.png', 2, -35),
    ]
    model = load_model()
    for path, t, angle in img_angle:
        img0 = np.array(cv2.imread(path))
        y = predict_class(model, img0, angle)
        print(f'{path}: predicted {y}, is {t}')
