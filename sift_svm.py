import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
import os
import pandas as pd
from tqdm import tqdm
import time


# Wrapper of the feature extraction function to keep the flag consistent for train, val, and test dataset.
def feature_extractor(max_num_kp=100, enable_enhance=True, enable_warning=True, show_features=False):
    def _extract_features(data_dir):
        file_names = os.listdir(data_dir)
        num_samples = len(file_names)
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=max_num_kp)
        features = np.zeros((num_samples, max_num_kp, 128), dtype=np.float32)
        labels = np.zeros(num_samples, dtype=np.int8)

        warning_log = []

        time.sleep(0.5)  # pause to show progress bar
        for idx, name in enumerate(tqdm(file_names, desc='extracting features from ' + data_dir)):
            labels[idx] = int(name[0])
            img = cv2.imread(os.path.join(data_dir, name), cv2.IMREAD_GRAYSCALE)
            kp, feature = sift.detectAndCompute(img, None)
            # Enhance image if not enough key points found
            if enable_enhance and len(kp) < max_num_kp:
                img_enhanced = cv2.equalizeHist(img)
                kp, feature = sift.detectAndCompute(img_enhanced, None)
            num_kp = len(kp)
            if num_kp == 0:
                warning_log.append(' * Unable to extract features for %s' % name)
                labels[idx] = -1  # mark as UNRECOGNIZABLE
                continue
            elif num_kp < max_num_kp:
                warning_log.append(' + Not enough key points found in %s : %d/%d!' % (name, len(kp), max_num_kp))
            num_kp = min(num_kp, max_num_kp)
            features[idx][:num_kp] = feature[:num_kp]
            if show_features:
                img_with_kp = cv2.drawKeypoints(img, kp, np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                plt.imshow(img_with_kp)
                plt.show()
        time.sleep(0.5)  # pause to show progress bar

        if enable_warning:
            for line in warning_log:
                print(line)

        return features.reshape(num_samples, -1), labels
    return _extract_features


def compute_accuracy(y_gt, y_p):
    return np.sum(y_gt == y_p) / y_gt.shape[0]


def validate_models(c_candidates, gamma_candidates, kernel='linear'):
    best_acc = -1
    best_models = None
    best_params = None
    train_acc_tbl = []
    val_acc_tbl = []

    time.sleep(0.5)  # pause to show progress bar
    with tqdm(total=c_candidates.shape[0] * gamma_candidates.shape[0], desc='tuning hyper-parameters') as pbar:
        for C in c_candidates:
            train_acc_row = []
            val_acc_row = []
            for gamma in gamma_candidates:
                classifier = svm.SVC(C=C, kernel=kernel, gamma=gamma)
                classifier.fit(train_fts, train_labels)
                preds = classifier.predict(train_fts)
                train_acc = compute_accuracy(train_labels, preds)
                preds = classifier.predict(val_fts)
                val_acc = compute_accuracy(val_labels, preds)
                # print('C = %.3e, gamma = %.3e, train_acc: %.3f%%, val_acc: %.3f%%' %
                #     (C, gamma, 100 * train_acc, 100 * val_acc))
                train_acc_row.append(100 * train_acc)
                val_acc_row.append(100 * val_acc)
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_models = [classifier]
                    best_params = [(C, gamma)]
                elif val_acc == best_acc:
                    best_models.append(classifier)
                    best_params.append((C, gamma))
                # Update progress bar
                pbar.update()
            train_acc_tbl.append(train_acc_row)
            val_acc_tbl.append(val_acc_row)
    time.sleep(0.5)  # pause to show progress bar

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    print('\nTrain Acc:')
    print(pd.DataFrame(data=train_acc_tbl, index=gamma_candidates, columns=c_candidates))
    print('Val Acc:')
    print(pd.DataFrame(data=val_acc_tbl, index=gamma_candidates, columns=c_candidates))
    return best_acc, best_params, best_models


base_dir = 'data0229_classic'

# Extract features
print('Extracting features ...')
extract_features = feature_extractor(enable_enhance=False)
train_fts, train_labels = extract_features(os.path.join(base_dir, 'train'))
val_fts, val_labels = extract_features(os.path.join(base_dir, 'val'))
test_fts, test_labels = extract_features(os.path.join(base_dir, 'test'))

# Validate & Test models
print('\nValidating model ...')
val_acc, params, models = validate_models(np.logspace(-10, 10, 7), 1 / np.logspace(-10, 10, 7), kernel='linear')
print()
for best_param, best_model in zip(params, models):
    preds = best_model.predict(test_fts)
    test_acc = compute_accuracy(test_labels, preds)
    stats = (*best_param, 100 * val_acc, 100 * test_acc)
    print('Found best model with C = %.3e, gamma = %.3e, val_acc %.3f%%, Test accuracy: %.3f%%' % stats)

