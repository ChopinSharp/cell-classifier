import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
import os
import pandas as pd
from tqdm import tqdm
# import time
from scipy.stats import gaussian_kde


base_dir = '../datasets/data0229_svm'


# Wrapper of the feature extraction function to keep the flag consistent for train, val, and test dataset.
def feature_extractor(max_num_kp=100, enable_enhance=True, enable_warning=True, show_features=False):
    def _extract_features(data_dir):
        file_names = os.listdir(data_dir)
        num_samples = len(file_names)
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=max_num_kp)
        features = np.zeros((num_samples, max_num_kp, 128), dtype=np.float32)
        labels = np.zeros(num_samples, dtype=np.int8)
        kp_num_log = []

        warning_log = []
        # enhanced_flags = np.zeros(num_samples, dtype=np.float32)

        # time.sleep(0.5)  # pause to show progress bar
        # for idx, name in enumerate(tqdm(file_names, desc='extracting features from ' + data_dir)):
        for idx, name in enumerate(file_names):
            labels[idx] = int(name[0])
            img = cv2.imread(os.path.join(data_dir, name), cv2.IMREAD_GRAYSCALE)
            kp, feature = sift.detectAndCompute(img, None)
            # Enhance image if not enough key points found
            kp_num_log.append(os.path.join(data_dir, name))
            if enable_enhance and len(kp) < max_num_kp:
                img_enhanced = cv2.equalizeHist(img)
                kp, feature = sift.detectAndCompute(img_enhanced, None)
            num_kp = len(kp)
            if num_kp == 0:
                warning_log.append(' * Unable to extract features for %s' % name)
                continue
            elif num_kp < max_num_kp:
                warning_log.append(' + Not enough key points found in %s : %d/%d!' % (name, len(kp), max_num_kp))
            num_kp = min(num_kp, max_num_kp)
            features[idx][:num_kp] = feature[:num_kp]
            if show_features:
                img_with_kp = cv2.drawKeypoints(img, kp, np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                plt.imshow(img_with_kp)
                plt.show()
        # time.sleep(0.5)  # pause to show progress bar

        features = features.reshape(num_samples, -1)

        if enable_warning:
            for line in warning_log:
                print(line)

        return features, labels, kp_num_log
    return _extract_features


def compute_accuracy(y_gt, y_p):
    return np.sum(y_gt == y_p) / y_gt.shape[0]


def validate_models(train_fts, train_labels, val_fts, val_labels, c_candidates, gamma_candidates, kernel='linear', verbose=True):
    best_acc = -1
    best_models = None
    best_params = None
    train_acc_tbl = []
    val_acc_tbl = []

    # time.sleep(0.5)  # pause to show progress bar
    # with tqdm(total=c_candidates.shape[0] * gamma_candidates.shape[0], desc='tuning hyper-parameters') as pbar:
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
            # pbar.update()
        train_acc_tbl.append(train_acc_row)
        val_acc_tbl.append(val_acc_row)
    # time.sleep(0.5)  # pause to show progress bar

    if verbose:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        print('\nTrain Acc:')
        print(pd.DataFrame(data=train_acc_tbl, index=c_candidates, columns=gamma_candidates))
        print('Val Acc:')
        print(pd.DataFrame(data=val_acc_tbl, index=c_candidates, columns=gamma_candidates))
    return best_acc, best_params, best_models


def plot_kp_distribution():
    sift = cv2.xfeatures2d.SIFT_create()
    kp_number = []
    for cls_folder in os.listdir(base_dir):
        path_to_cls_folder = os.path.join(base_dir, cls_folder)
        for name in os.listdir(path_to_cls_folder):
            img = cv2.imread(os.path.join(path_to_cls_folder, name), cv2.IMREAD_GRAYSCALE)
            kp, feature = sift.detectAndCompute(img, None)
            kp_number.append(len(kp))
    plt.title('Density of SIFT Key Points')
    plt.xlabel("kp")
    plt.ylabel("sample")
    x = np.arange(max(kp_number) + 1)
    kde_x = gaussian_kde(kp_number)
    plt.plot(x, kde_x.evaluate(x))
    plt.show()


def main(enable_enhance=False, kp_num=100, verbose=True):
    # Extract features
    print('Extracting features ...')
    extract_features = feature_extractor(max_num_kp=kp_num, enable_enhance=enable_enhance)
    train_fts, train_labels, _ = extract_features(os.path.join(base_dir, 'train'))
    val_fts, val_labels, _ = extract_features(os.path.join(base_dir, 'val'))
    test_fts, test_labels, _ = extract_features(os.path.join(base_dir, 'test'))

    # Validate & Test models
    print('\nValidating model ...')
    val_acc, params, models = validate_models(
        train_fts, train_labels,
        val_fts, val_labels,
        np.logspace(-10, 3, 7),
        np.array([1.]),  # 1 / np.logspace(-10, 10, 7),
        kernel='linear',
        verbose=True
    )
    print()

    if verbose:
        for best_param, best_model in zip(params, models):
            preds = best_model.predict(test_fts)
            test_acc = compute_accuracy(test_labels, preds)
            print('Found best model with C = %.3e, gamma = %.3e, val_acc %.3f%%, Test accuracy: %.3f%%' %
                  (best_param[0], best_param[1], 100 * val_acc, 100 * test_acc))

    preds = models[0].predict(test_fts)
    test_acc = compute_accuracy(test_labels, preds)

    return val_acc, test_acc


def test_kp_num():
    val_acc_list = []
    test_acc_list = []
    for kp_num in tqdm(range(10, 101, 10)):
        print('Using kp number', kp_num)
        val_acc, test_acc = main(False, kp_num, verbose=False)
        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        print('\n* kp number: %d, val_acc: %f, test_acc: %f\n' % (kp_num, val_acc, test_acc))
    plt.xlabel('kp num')
    plt.ylabel('acc')
    plt.plot(range(10, 101, 10), val_acc_list, label='val')
    plt.plot(range(10, 101, 10), test_acc_list, label='test')
    plt.legend(loc='best')
    plt.savefig('result.png')


# legacy code ... theoretically wrong ...
def show_err_dist():
    print('Extracting features ...')
    extract_features = feature_extractor(max_num_kp=100, enable_enhance=False)
    fts = {}
    labels = {}
    logs = {}
    for t in ['train', 'val', 'test']:
        fts[t], labels[t], logs[t] = extract_features(os.path.join(base_dir, t))

    print('Validating models ...')
    val_acc, params, models = validate_models(
        fts['train'], labels['train'],
        fts['val'], labels['val'],
        np.logspace(-10, 3, 7),
        np.array([1.]),  # 1 / np.logspace(-10, 10, 7),
        kernel='linear',
        verbose=False
    )

    model = models[0]
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=100)
    for t in ['train', 'val', 'test']:
        preds = model.predict(fts[t])
        acc = compute_accuracy(labels[t], preds)
        print('%s acc: %f' % (t, acc))
        for file in [logs[t][i] for i in range(len(preds)) if preds[i] != labels[t][i]]:
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            kp, feature = sift.detectAndCompute(img, None)
            print(file, 'with', len(kp), 'key point(s)')


def show_enhance():
    samples = [os.path.join(base_dir, 'train', name) for name in ['1 (45).jpg']]#, '1 (56).jpg', '1 (49).jpg', '1 (50).jpg']]
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=100)
    for idx, path in enumerate(samples):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        kp, _ = sift.detectAndCompute(img, None)
        img_with_kp = cv2.drawKeypoints(img, kp, np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        img_enhanced = cv2.equalizeHist(img)
        kp, _ = sift.detectAndCompute(img_enhanced, None)
        img_enhanced_with_kp = cv2.drawKeypoints(img_enhanced, kp, np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.subplot(len(samples), 2, 1 + 2 * idx)
        plt.imshow(img_with_kp)
        plt.subplot(len(samples), 2, 2 + 2 * idx)
        plt.imshow(img_enhanced_with_kp)
    plt.show()


if __name__ == '__main__':
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=10000)
    dir_ = '../datasets/data0229_svm'
    path = os.path.join(dir_, 'train', '1 (45).jpg')
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    kp, _ = sift.detectAndCompute(img, None)
    print(len(kp))
    dir_ = '../datasets/data0229_svm_enhanced'
    path = os.path.join(dir_, 'train', '1 (45).jpg')
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    kp, _ = sift.detectAndCompute(img, None)
    print(len(kp))
