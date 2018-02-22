import csv
import os

import cv2
import numpy as np

import app.grad_cam as gc
from app.datasets.dataset_loader import DataSetTest
from app.main.Actions import Actions
from app.models.model_factory import get_model
from app.utilities import metrics


class Test(Actions):
    y = None
    y_hat = None

    def __init__(self, config_file):
        super().__init__(config_file)

    def grad_cam(self):
        print("** perform grad cam **")
        os.makedirs("imgdir", exist_ok=True)
        pred_log_path = os.path.join(self.conf.output_dir, "predicted_class.csv")
        with open(pred_log_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_header = ['ID', 'Most probable diagnosis']
            for i, v in enumerate(self.DSConfig.class_names):
                csv_header.append(f"{v}_Prob")
            csvwriter.writerow(csv_header)
            for i, v in enumerate(self.y_hat):
                print(f"** y_hat[{i}] = {np.round(v, 3)}")
                print(f"** y[{i}] = {self.y[i]}")
                predicted_class = np.argmax(v)
                labeled_classes = ",".join(
                    [self.DSConfig.class_names[yi] for yi, yiv in enumerate(self.y[i]) if yiv == 1])
                if labeled_classes == "":
                    labeled_classes = "Normal"

                print(f"** Label/Prediction: {labeled_classes}/{self.DSConfig.class_names[predicted_class]}")
                csv_row = [str(i + 1), f"{self.DSConfig.class_names[predicted_class]}"] + [str(vi.round(3)) for vi in v]
                csvwriter.writerow(csv_row)
                x_orig = self.test_generator.orig_input(i).squeeze()
                x_orig = cv2.cvtColor(x_orig, cv2.COLOR_GRAY2RGB)
                x = self.test_generator.model_input(i)
                cam = gc.grad_cam(self.model, x, x_orig, predicted_class, "conv5_blk_scale", self.DSConfig.class_names)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(x_orig, f"Labeled as:{labeled_classes}", (5, 20), font, 1,
                            (255, 255, 255),
                            2, cv2.LINE_AA)

                cv2.putText(cam, f"Predicted as:{self.DSConfig.class_names[predicted_class]}", (5, 20), font, 1,
                            (255, 255, 255),
                            2, cv2.LINE_AA)

                print(f"Writing cam file to imgdir/gradcam_{i}.jpg")

                cv2.imwrite(f"imgdir/gradcam_{i}.jpg", np.concatenate((x_orig, cam), axis=1))

    def prepare_dataset(self):
        dataset0 = DataSetTest(self.DSConfig)

        print("** load test generator **")
        self.test_generator = dataset0.test_generator(verbosity=self.conf.verbosity)

    def prepare_model(self):
        print("** load model **")
        self.model = get_model(self.DSConfig.class_names, weights_path=self.MDConfig.trained_model_weights,
                               image_dimension=self.IMConfig.img_dim, color_mode=self.IMConfig.color_mode,
                               class_mode=self.DSConfig.class_mode)

    def test(self):
        self.prepare_dataset()
        self.prepare_model()
        print("** make predictions **")
        aurocs, mean_auroc, self.y, self.y_hat = metrics.compute_auroc(self.model, self.test_generator,
                                                                       self.conf.class_mode,
                                                                       self.DSConfig.class_names)

        test_log_path = os.path.join(self.conf.output_dir, "test.log")

        with open(test_log_path, "w") as f:
            print(f"** write log to {test_log_path} **")
            for i, v in enumerate(self.DSConfig.class_names):
                f.write(f"{self.DSConfig.class_names[i]}: {aurocs[i]}\n")

            f.write("-------------------------\n")
            f.write(f"mean AUC: {mean_auroc}\n")

        if self.conf.enable_grad_cam:
            self.grad_cam()
