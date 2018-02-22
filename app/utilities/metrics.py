import numpy as np
from keras.models import Model
from sklearn.metrics import roc_auc_score


def compute_auroc(model: Model, test_generator, class_mode, class_names, step_test=None):
    if step_test is None:
        step_test = test_generator.__len__()

    y = np.array(test_generator.targets(steps=step_test)).squeeze()
    y_hat = np.array(
        model.predict_generator(generator=test_generator, steps=step_test, max_queue_size=10, workers=1,
                                use_multiprocessing=False, verbose=1)).squeeze()

    if class_mode == "multibinary":
        y_hat = y_hat.swapaxes(0, 1)
    print(f"*** dev auroc ***")
    print(f"y = {np.shape(y)}")
    print(f"y_hat = {np.shape(y_hat)}")

    current_auroc = roc_auc_score(y, y_hat, average=None)
    if len(class_names) != len(current_auroc):
        raise Exception(f"Wrong shape in either y or y_hat {len(self.class_names)} != {len(current_auroc)}")
    for i, v in enumerate(class_names):
        print(f" {i+1}. {v} AUC = {np.around(current_auroc[i], 3)}")

    # customize your multiple class metrics here
    mean_auroc = np.mean(current_auroc)
    print(f"Mean AUC: {np.around(mean_auroc, 3)}\n")
    return current_auroc, mean_auroc, y, y_hat
