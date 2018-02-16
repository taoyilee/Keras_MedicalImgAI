import os

import cv2


def resize_img(image_path, output_dir, output_size):
    _, fn = os.path.split(image_path)
    fn_root, _ = os.path.splitext(fn)
    image_out_base = f"{fn_root}.png"
    image_out_path = os.path.join(output_dir, image_out_base)
    if not os.path.isfile(image_out_path):
        print(f"Processing {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            raise Exception(f"{image_path} is empty image?")
        w = int(img.shape[0])
        h = int(img.shape[1])
        landscape = w > h
        portrait = h > w
        starting_index = int(abs(w - h) / 2)
        if landscape:
            img = img[starting_index:starting_index + h, :]
        if portrait:
            img = img[:, starting_index:starting_index + w]
        img = cv2.resize(img, (output_size, output_size))
        print(f"Writing resized to {image_out_path} {img.shape[0]}X{img.shape[1]}")
        cv2.imwrite(image_out_path, img)
    else:
        print(f"Skip {image_path}")
