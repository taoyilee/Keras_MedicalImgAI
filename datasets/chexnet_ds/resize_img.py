import os

import cv2


def resize_img(csvwriter, image_path, output_dir, output_size):
    print(f"Processing {image_path} {os.path.isfile(image_path)}")
    img = cv2.imread(image_path)
    _, fn = os.path.split(image_path)
    fn_root, _ = os.path.splitext(fn)
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
    image_out_base = f"{prefix_str[i]}_{fn_root}.png"
    image_out_path = os.path.join(output_dir, image_out_base)
    print(f"Writing resized to {image_out_path} {img.shape[0]}X{img.shape[1]}")
    cv.imwrite(image_out_path, img)
    csvwriter.writerow([image_out_base, diag_str[i], patient_ID, img.shape[0], img.shape[1]])
