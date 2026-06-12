import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from src.config import SEED, BATCH_SIZE

def collect_all_images(data_root):
    all_paths, all_labels = [], []
    for split in ['train', 'val', 'test']:
        for cls_idx, cls in enumerate(['NORMAL', 'PNEUMONIA']):
            folder = os.path.join(data_root, split, cls)
            if not os.path.exists(folder):
                continue
            for fname in os.listdir(folder):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_paths.append(os.path.join(folder, fname))
                    all_labels.append(cls_idx)
    return np.array(all_paths), np.array(all_labels)

def stratified_split(paths, labels, train_ratio=0.70, seed=SEED):
    X_train, X_temp, y_train, y_temp = train_test_split(
        paths, labels, test_size=(1 - train_ratio), stratify=labels, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test

def apply_clahe(img_array):
    img_uint8 = (img_array * 255).astype(np.uint8) if img_array.max() <= 1.0 else img_array.astype(np.uint8)
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return enhanced.astype(np.float32) / 255.0

def load_and_preprocess(path, img_size=224, augmenter=None):
    img = cv2.imread(path)
    if img is None:
        return np.zeros((img_size, img_size, 3), dtype=np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = apply_clahe(img)
    if augmenter is not None:
        img = augmenter(image=img)
    return img

class XRayDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, paths, labels, batch_size=BATCH_SIZE, img_size=224, augment=False, shuffle=True, seed=SEED):
        self.paths = np.array(paths)
        self.labels = np.array(labels, dtype=np.float32)
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.indexes = np.arange(len(self.paths))
        if self.shuffle:
            self.rng.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        X = np.zeros((len(batch_idx), self.img_size, self.img_size, 3), dtype=np.float32)
        y = self.labels[batch_idx]
        for i, bi in enumerate(batch_idx):
            img = load_and_preprocess(self.paths[bi], self.img_size)
            if self.augment:
                img = self._augment(img)
            X[i] = img
        return X, y

    def _augment(self, img):
        if self.rng.random() < 0.5:
            img = np.fliplr(img)
        angle = self.rng.uniform(-15, 15)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        zoom = self.rng.uniform(0.9, 1.1)
        nh, nw = int(h * zoom), int(w * zoom)
        img_z = cv2.resize(img, (nw, nh))
        if zoom > 1:
            sy, sx = (nh - h) // 2, (nw - w) // 2
            img = img_z[sy:sy+h, sx:sx+w]
        else:
            pad_y, pad_x = (h - nh) // 2, (w - nw) // 2
            img = np.pad(img_z, ((pad_y, h-nh-pad_y), (pad_x, w-nw-pad_x), (0,0)), mode='reflect')
        factor = self.rng.uniform(0.85, 1.15)
        img = np.clip(img * factor, 0, 1)
        return img.astype(np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indexes)

def get_class_weights(y_train):
    class_weights_array = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    return {0: class_weights_array[0], 1: class_weights_array[1]}