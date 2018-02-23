import csv
import os

import cv2
import numpy as np
from keras.models import Model

import app.grad_cam as gc
from app.datasets.dataset_loader import DataSetTest
from app.main.Actions import Actions
from app.models.model_factory import get_model
from app.utilities import metrics


class Test(Actions):
    y = None
    y_hat = None
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    model: Model = []
    test_generator = None

    def __init__(self, config_file, no_grad_cam):
        super().__init__(config_file)
        self.no_grad_cam = no_grad_cam

    def prediction_summary(self):
        print("** Write prediction summary **")
        pred_log_path = os.path.join(self.conf.output_dir, "predicted_class.csv")
        with open(pred_log_path, 'w', newline='') as csvfile:
            csv_file_handle = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_header = ['ID', 'Labels', 'Most probable diagnosis']
            csv_file_handle.writerow(csv_header)

            for i, yi in enumerate(self.y):
                y = np.array(yi).squeeze()
                y_hat = np.array(self.y_hat[i]).squeeze()
                if self.DSConfig.class_mode == "multibinary":
                    y = y.swapaxes(0, 1)
                    y_hat = y_hat.swapaxes(0, 1)
                predicted_priority = np.argsort(y_hat)
                image_id = str(i + 1)
                labeled_classes = "/".join([self.DSConfig.class_names[yi] for yi, yiv in enumerate(y) if yiv == 1])
                predicted_classes = ["{}({:.3f})".format(self.DSConfig.class_names[p], y_hat[p]) for p in
                                     predicted_priority]
                csv_row = [image_id, labeled_classes] + predicted_classes
                csv_file_handle.writerow(csv_row)

    def grad_cam(self):
        print("** Perform grad cam **")
        os.makedirs(self.conf.grad_cam_outputdir, exist_ok=True)
        for i, yi in enumerate(self.y):
            y = np.array(yi).squeeze()
            y_hat = np.array(self.y_hat[i]).squeeze()
            if self.DSConfig.class_mode == "multibinary":
                y = y.swapaxes(0, 1)
                y_hat = y_hat.swapaxes(0, 1)
            if self.conf.verbosity > 0:
                print(f"** y    [{i}] = ", end="")
                print(",".join(["{:.3f}".format(yi.round(3)) for yi in y[b]]))
                print(f"** y_hat[{i}] = ", end="")
                print(",".join(["{:.3f}".format(y_hati.round(3)) for y_hati in y_hat[b]]))
            predicted_class = np.argmax(y_hat)
            labeled_classes = ",".join([self.DSConfig.class_names[yi] for yi, yiv in enumerate(y) if yiv == 1])
            if labeled_classes == "":
                labeled_classes = "Normal"
            if self.conf.verbosity > 0:
                print("** Label/Prediction: {}/{}({:.3f})".format(labeled_classes,
                                                                  self.DSConfig.class_names[predicted_class],
                                                                  np.round(y_hat[predicted_class], 3)))

            x_orig = self.test_generator.inputs(i, mode="raw").squeeze()
            x_model = self.test_generator.inputs(i, mode="test")
            cam = gc.grad_cam(self.model, x_model, x_orig, predicted_class, "bn", self.DSConfig.class_names)

            cv2.putText(x_orig, f"Labeled as:{labeled_classes}", (5, 20), self.FONT, 1,
                        (255, 255, 255),
                        2, cv2.LINE_AA)
            y_hat_top3 = np.argsort(y_hat)

            for j in range(3):
                if abs(-j - 1) <= len(y_hat_top3):
                    cv2.putText(cam, "Predicted as: ({}) {}({:.3f})".format(j + 1, self.DSConfig.class_names[y_hat_top3[-j - 1]],
                                                              np.round(y_hat[y_hat_top3[-j - 1]], 3)),
                                (5, 20 + 30 * j), self.FONT, 1, (255, 255, 255), 2, cv2.LINE_AA)

            output_file = os.path.join(self.conf.grad_cam_outputdir, f"gradcam_{i}.jpg")
            print(f"Writing cam file to {output_file}")
            cv2.imwrite(output_file, np.concatenate((x_orig, cam), axis=1))

    def prepare_dataset(self):
        dataset0 = DataSetTest(self.DSConfig)

        print("** load test generator **")
        self.test_generator = dataset0.test_generator(verbosity=self.conf.verbosity)

    def prepare_model(self):
        print("** load model **")
        self.MDConfig.use_trained_model_weights = True
        print(f"** Trained Model = {self.MDConfig.trained_model_weights} **")
        self.model = get_model(self.DSConfig.class_names, weights_path=self.MDConfig.trained_model_weights,
                               image_dimension=self.IMConfig.img_dim, color_mode=self.IMConfig.color_mode,
                               class_mode=self.DSConfig.class_mode)

    def test(self):
        self.prepare_dataset()
        self.prepare_model()

        print("** make predictions **")
        aurocs, mean_auroc, self.y, self.y_hat = metrics.compute_auroc(self.model, self.test_generator,
                                                                       self.conf.class_mode,
                                                                       self.DSConfig.class_names,
                                                                       step_test=self.conf.test_steps)

        test_log_path = os.path.join(self.conf.output_dir, "test.log")

        with open(test_log_path, "w") as f:
            print(f"** write log to {test_log_path} **")
            for i, v in enumerate(self.DSConfig.class_names):
                f.write(f"{self.DSConfig.class_names[i]}: {aurocs[i]}\n")

            f.write("-------------------------\n")
            f.write(f"mean AUC: {mean_auroc}\n")
        self.prediction_summary()

        if self.conf.enable_grad_cam and not self.no_grad_cam:
            self.grad_cam()
