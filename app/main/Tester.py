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

    def __init__(self, config_file, grad_cam_only):
        super().__init__(config_file)
        self.grad_cam_only = grad_cam_only

    def grad_cam(self):
        print("** perform grad cam **")
        if self.MDConfig.show_model_summary:
            self.model.summary()
        os.makedirs("imgdir", exist_ok=True)
        pred_log_path = os.path.join(self.conf.output_dir, "predicted_class.csv")
        with open(pred_log_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_header = ['ID', 'Most probable diagnosis']
            for i, v in enumerate(self.DSConfig.class_names):
                csv_header.append(f"{v}_Prob")
            csvwriter.writerow(csv_header)

            for i in range(self.test_generator.__len__()):
                inputs, y, inputs_orig = self.test_generator.getitem_cam(i)
                y = np.array(y).swapaxes(0, 1).squeeze()
                y_hat = np.array(self.model.predict_on_batch(inputs)).squeeze().swapaxes(0, 1)
                for b in range(self.conf.batch_size):
                    imageid = i * self.conf.batch_size + b
                    print(f"** y    [{imageid}] = ", end="")
                    print(",".join(["{:.3f}".format(yi.round(3)) for yi in y[b]]))
                    print(f"** y_hat[{imageid}] = ", end="")
                    print(",".join(["{:.3f}".format(y_hati.round(3)) for y_hati in y_hat[b]]))
                    predicted_class = np.argmax(y_hat[b])
                    labeled_classes = ",".join(
                        [self.DSConfig.class_names[yi] for yi, yiv in enumerate(y[b]) if yiv == 1])
                    if labeled_classes == "":
                        labeled_classes = "Normal"
                    print("** Label/Prediction: {}/{}({:.3f})".format(labeled_classes,
                                                                      self.DSConfig.class_names[predicted_class],
                                                                      np.round(y_hat[b][predicted_class], 3)))
                    csv_row = [str(imageid + 1), f"{self.DSConfig.class_names[predicted_class]}"] + [
                        str(y_hati.round(3)) for y_hati in y_hat[b]]
                    csvwriter.writerow(csv_row)
                    x_orig = inputs_orig[b].squeeze()
                    x = inputs[b][np.newaxis, :, :, :]
                    print(np.shape(x))
                    cam = gc.grad_cam(self.model, x, x_orig, predicted_class, "bn", self.DSConfig.class_names)

                    cv2.putText(x_orig, f"Labeled as:{labeled_classes}", (5, 20), self.FONT, 1,
                                (255, 255, 255),
                                2, cv2.LINE_AA)

                    cv2.putText(cam, "Predicted as:{}({:.3f})".format(self.DSConfig.class_names[predicted_class],
                                                                      np.round(y_hat[b][predicted_class], 3)),
                                (5, 20), self.FONT, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    print(f"Writing cam file to imgdir/gradcam_{imageid}.jpg")

                    cv2.imwrite(f"imgdir/gradcam_{imageid}.jpg", np.concatenate((x_orig, cam), axis=1))

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

        if not self.grad_cam_only:
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

        if self.conf.enable_grad_cam:
            self.grad_cam()
