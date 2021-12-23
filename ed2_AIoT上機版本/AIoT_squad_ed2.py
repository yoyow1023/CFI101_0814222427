import io
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import sys
import tqdm
import csv
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose
import math

def show_image(img, figsize=(10, 10)):
    """Shows output PIL image."""
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.show()


## 1. 人體姿勢特徵嵌入(Pose Embedding)

class FullBodyPoseEmbedder(object):
    """Converts 3D pose landmarks into 3D embedding."""

    def __init__(self, torso_size_multiplier=2.5):
        # Multiplier to apply to the torso to get minimal body size.
        self._torso_size_multiplier = torso_size_multiplier

        # Names of the landmarks as they appear in the prediction.
        self._landmark_names = [
            'nose',
            'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear',
            'mouth_left', 'mouth_right',
            'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist',
            'left_pinky_1', 'right_pinky_1',
            'left_index_1', 'right_index_1',
            'left_thumb_2', 'right_thumb_2',
            'left_hip', 'right_hip',
            'left_knee', 'right_knee',
            'left_ankle', 'right_ankle',
            'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index',
        ]

    def __call__(self, landmarks):
        """Normalizes pose landmarks and converts to embedding

        Args:
          landmarks - NumPy array with 3D landmarks of shape (N, 3).

        Result:
          Numpy array with pose embedding of shape (M, 3) where `M` is the number of
          pairwise distances defined in `_get_pose_distance_embedding`.
        """
        assert landmarks.shape[0] == len(self._landmark_names), 'Unexpected number of landmarks: {}'.format(landmarks.shape[0])

        # Get pose landmarks.
        landmarks = np.copy(landmarks)

        # Normalize landmarks.
        landmarks = self._normalize_pose_landmarks(landmarks)

        # Get embedding.
        embedding = self._get_pose_distance_embedding(landmarks)

        return embedding

    def _normalize_pose_landmarks(self, landmarks):
        """Normalizes landmarks translation and scale."""
        landmarks = np.copy(landmarks)

        # Normalize translation.
        pose_center = self._get_pose_center(landmarks)
        landmarks -= pose_center

        # Normalize scale.
        pose_size = self._get_pose_size(landmarks, self._torso_size_multiplier)
        landmarks /= pose_size
        # Multiplication by 100 is not required, but makes it eaasier to debug.
        landmarks *= 100

        return landmarks

    def _get_pose_center(self, landmarks):
        """Calculates pose center as point between hips."""
        left_hip = landmarks[self._landmark_names.index('left_hip')]
        right_hip = landmarks[self._landmark_names.index('right_hip')]
        center = (left_hip + right_hip) * 0.5
        return center

    def _get_pose_size(self, landmarks, torso_size_multiplier):
        """Calculates pose size.

        It is the maximum of two values:
          * Torso size multiplied by `torso_size_multiplier`
          * Maximum distance from pose center to any pose landmark
        """
        # This approach uses only 2D landmarks to compute pose size.
        landmarks = landmarks[:, :2]

        # Hips center.
        left_hip = landmarks[self._landmark_names.index('left_hip')]
        right_hip = landmarks[self._landmark_names.index('right_hip')]
        hips = (left_hip + right_hip) * 0.5

        # Shoulders center.
        left_shoulder = landmarks[self._landmark_names.index('left_shoulder')]
        right_shoulder = landmarks[self._landmark_names.index('right_shoulder')]
        shoulders = (left_shoulder + right_shoulder) * 0.5

        # Torso size as the minimum body size.
        torso_size = np.linalg.norm(shoulders - hips)

        # Max dist to pose center.
        pose_center = self._get_pose_center(landmarks)
        max_dist = np.max(np.linalg.norm(landmarks - pose_center, axis=1))

        return max(torso_size * torso_size_multiplier, max_dist)

    def _get_pose_distance_embedding(self, landmarks):
        """Converts pose landmarks into 3D embedding.

        We use several pairwise 3D distances to form pose embedding. All distances
        include X and Y components with sign. We differnt types of pairs to cover
        different pose classes. Feel free to remove some or add new.

        Args:
          landmarks - NumPy array with 3D landmarks of shape (N, 3).

        Result:
          Numpy array with pose embedding of shape (M, 3) where `M` is the number of
          pairwise distances.
        """
        embedding = np.array([
            # One joint.

            self._get_distance(
                self._get_average_by_names(landmarks, 'left_hip', 'right_hip'),
                self._get_average_by_names(landmarks, 'left_shoulder', 'right_shoulder')),

            #self._get_distance_by_names(landmarks, 'left_shoulder', 'left_elbow'),
            #self._get_distance_by_names(landmarks, 'right_shoulder', 'right_elbow'),

            #self._get_distance_by_names(landmarks, 'left_elbow', 'left_wrist'),
            #self._get_distance_by_names(landmarks, 'right_elbow', 'right_wrist'),

            self._get_distance_by_names(landmarks, 'left_hip', 'left_knee'),
            self._get_distance_by_names(landmarks, 'right_hip', 'right_knee'),

            self._get_distance_by_names(landmarks, 'left_knee', 'left_ankle'),
            self._get_distance_by_names(landmarks, 'right_knee', 'right_ankle'),

            # Two joints.

            #self._get_distance_by_names(landmarks, 'left_shoulder', 'left_wrist'),
            #self._get_distance_by_names(landmarks, 'right_shoulder', 'right_wrist'),

            self._get_distance_by_names(landmarks, 'left_hip', 'left_ankle'),
            self._get_distance_by_names(landmarks, 'right_hip', 'right_ankle'),

            # Four joints.

            #self._get_distance_by_names(landmarks, 'left_hip', 'left_elbow'),
            #self._get_distance_by_names(landmarks, 'right_hip', 'right_elbow'),
            self._get_distance_by_names(landmarks, 'left_shoulder','left_knee'),
            self._get_distance_by_names(landmarks, 'right_shoulder','right_knee'),

            # Five joints.

            self._get_distance_by_names(landmarks, 'left_shoulder', 'left_ankle'),
            self._get_distance_by_names(landmarks, 'right_shoulder', 'right_ankle'),

            #self._get_distance_by_names(landmarks, 'left_hip', 'left_wrist'),
            #self._get_distance_by_names(landmarks, 'right_hip', 'right_wrist'),

            # Cross body.

            #self._get_distance_by_names(landmarks, 'left_elbow', 'right_elbow'),
            self._get_distance_by_names(landmarks, 'left_knee', 'right_knee'),

            #self._get_distance_by_names(landmarks, 'left_wrist', 'right_wrist'),
            self._get_distance_by_names(landmarks, 'left_ankle', 'right_ankle'),

            # Body bent direction.

            # self._get_distance(
            #     self._get_average_by_names(landmarks, 'left_wrist', 'left_ankle'),
            #     landmarks[self._landmark_names.index('left_hip')]),
            # self._get_distance(
            #     self._get_average_by_names(landmarks, 'right_wrist', 'right_ankle'),
            #     landmarks[self._landmark_names.index('right_hip')]),
        ])

        return embedding

    def _get_average_by_names(self, landmarks, name_from, name_to):
        lmk_from = landmarks[self._landmark_names.index(name_from)]
        lmk_to = landmarks[self._landmark_names.index(name_to)]
        return (lmk_from + lmk_to) * 0.5

    def _get_distance_by_names(self, landmarks, name_from, name_to):
        lmk_from = landmarks[self._landmark_names.index(name_from)]
        lmk_to = landmarks[self._landmark_names.index(name_to)]
        return self._get_distance(lmk_from, lmk_to)

    def _get_distance(self, lmk_from, lmk_to):
        return lmk_to - lmk_from


## 2. & 3.定義人體姿勢分類使用到的輔助類別
 #兩者功能都是用來儲存屬性方便後續調用

class PoseSample(object):

    def __init__(self, name, landmarks, class_name, embedding):
        self.name = name
        self.landmarks = landmarks
        self.class_name = class_name

        self.embedding = embedding


class PoseSampleOutlier(object):

    def __init__(self, sample, detected_class, all_classes):
        self.sample = sample
        self.detected_class = detected_class
        self.all_classes = all_classes


## 4. 人體姿勢分類

class PoseClassifier(object):
    """Classifies pose landmarks."""

    def __init__(self,
                 pose_samples_folder,
                 pose_embedder,
                 file_extension='csv',
                 file_separator=',',
                 n_landmarks=33,
                 n_dimensions=3,
                 #                top_n_by_max_distance=30,
                 top_n_by_mean_distance=10,
                 axes_weights=(1., 1., 0.2)):
        self._pose_embedder = pose_embedder
        self._n_landmarks = n_landmarks
        self._n_dimensions = n_dimensions
        self._axes_weights = axes_weights
        self._pose_samples = self._load_pose_samples(pose_samples_folder,
                                                     file_extension,
                                                     file_separator,
                                                     n_landmarks,
                                                     n_dimensions,
                                                     pose_embedder)

        self._top_n_by_mean_distance = top_n_by_mean_distance
        # ------------------------------- #selfchange_ed1_更改篩選訓練圖片的張數，從30張改為所有訓練圖80%張數:
        # 自行更改: 篩選outlier使用的np.max，從只選前30張誤差最小的圖改為選取所有圖片的0.8倍數量 (2021/12/14)
        # 其中還註解掉3個地方的 top_n_by_max_distance=30 參數
        self._num = 0
        file_names = [name for name in os.listdir(pose_samples_folder) if name.endswith(file_extension)]
        for file_name in file_names:
            # Use file name as pose class name.
            class_name = file_name[:-(len(file_extension) + 1)]
            with open(os.path.join(pose_samples_folder, file_name)) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=file_separator)
                for idx, _ in enumerate(csv_reader):
                    self._num += 1

        self._top_n_by_max_distance = math.floor(0.8 * self._num)

    def printnum(self):
        return print(self._num)

    # -------------------------------

    # ------------------------------- #selfchange_ed1_自定義函數:計算與正確姿勢的所有距離特徵誤差(2021/12/18)
    # 自行更改: 增加一個函數,用來定義分數:
    def compare_correct_pose_distance(self, pose_landmarks, class_name):
        # 將處於蹲著的sample找出:
        target_pose_sample = []
        for sample in self._pose_samples:
            if sample.class_name == class_name:
                target_pose_sample.append(sample)

        # 確認輸入特徵資料是否與目標姿勢擁有相同的shape:
        assert pose_landmarks.shape == (self._n_landmarks, self._n_dimensions), 'Unexpected shape: {}'.format(
            pose_landmarks.shape)

        # 做人體姿勢的特徵嵌入(包含原始圖與進鏡像圖).
        pose_embedding = self._pose_embedder(pose_landmarks)
        flipped_pose_embedding = self._pose_embedder(pose_landmarks * np.array([-1, 1, 1]))

        # 計算距離特徵平均差距，從小到大排列並挑選出差距前15%小的圖片(有做鏡像處理) (2021/12/20)
        mean_dist_heap = []
        for sample_idx, sample in enumerate(target_pose_sample):
            mean_dist = min(
                np.mean(np.abs((sample.embedding - pose_embedding) * self._axes_weights)),
                np.mean(np.abs((sample.embedding - flipped_pose_embedding) * self._axes_weights)),
            )
            mean_dist_heap.append([mean_dist, sample_idx])

        mean_dist_heap_idx = sorted(mean_dist_heap, key=lambda x: x[0])
        mean_dist_heap_idx = mean_dist_heap_idx[:math.floor(0.15 * self._num)]

        # 將上面取出前15%角度最接近的正確姿勢圖拿來與frame做距離特徵相減，並且取平均:
        # (用單張frame正常及鏡像的距離特徵，與正確姿勢圖比較差距,選擇較小的所有向量當作真正的差距，以此解決鏡像問題)
        distance_error_list = []
        for _, idx_number in mean_dist_heap_idx:
            sample = target_pose_sample[idx_number]
            distance_error_list.append(
                np.min([(sample.embedding - pose_embedding) * self._axes_weights,
                        (sample.embedding - flipped_pose_embedding) * self._axes_weights], 0))
        # (接著透過平均，將最接近角度的數個樣本的特徵向量最小距離做平均: 最後依然會是(M,3)的矩陣
        pose_embedding_compare_error = np.mean(distance_error_list, 0)

        return pose_embedding_compare_error

        # -------------------------------

    # ------------------------------- #selfchange_ed2_自定義函數:計算膝蓋平行評分 (2021/12/22)
    def find_knees_score(self, pose_distance_error, T=30, alpha1=0.02, alpha2=0.05, alpha3=0.1, distance1=5,
                         distance2=8, distance3=12):
        '''我們會把用來預測的影片,其每張frame與正確姿勢計算得到的相對誤差,拿來做計算。

        Required input: (13,3) distance error feature [from previous function]
        Output: one value which means knees_parallel_score
        方法: 會使用到 左膝蓋到右膝蓋的向量以及左腳踝到右腳踝的向量，將這兩個向量與正確姿勢平均向量相減，
            會得到兩個相對向量，再將膝蓋跟腳踝兩個相對向量相減，可以得到一個比較向量。
            我們會透過比較向量的y軸
        '''
        distance = abs(pose_distance_error[11][1] - pose_distance_error[12][1])

        coef = math.pi / (2 * T)
        y2 = np.cos(coef * distance)
        if distance < distance1:
            y3 = y2
        else:
            y3 = y2 * math.exp(-alpha1 * (distance - distance1))

        if distance < distance2:
            y4 = y3
        else:
            y4 = y3 * math.exp(-alpha2 * (distance - distance2))

        if distance < distance3:
            y5 = y4
        else:
            y5 = y4 * math.exp(-alpha3 * (distance - distance3))

        return distance, y5 * 100

    # -------------------------------

    def _load_pose_samples(self,
                           pose_samples_folder,
                           file_extension,
                           file_separator,
                           n_landmarks,
                           n_dimensions,
                           pose_embedder):
        """Loads pose samples from a given folder.

        Required folder structure:
          neutral_standing.csv
          pushups_down.csv
          pushups_up.csv
          squats_down.csv
          ...

        Required CSV structure:
          sample_00001,x1,y1,z1,x2,y2,z2,....
          sample_00002,x1,y1,z1,x2,y2,z2,....
          ...
        """
        # Each file in the folder represents one pose class.
        file_names = [name for name in os.listdir(pose_samples_folder) if name.endswith(file_extension)]

        pose_samples = []
        for file_name in file_names:
            # Use file name as pose class name.
            class_name = file_name[:-(len(file_extension) + 1)]

            # Parse CSV.
            with open(os.path.join(pose_samples_folder, file_name)) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=file_separator)
                for row in csv_reader:
                    assert len(row) == n_landmarks * n_dimensions + 1, 'Wrong number of values: {}'.format(len(row))
                    landmarks = np.array(row[1:], np.float32).reshape([n_landmarks, n_dimensions])
                    pose_samples.append(PoseSample(
                        name=row[0],
                        landmarks=landmarks,
                        class_name=class_name,
                        embedding=pose_embedder(landmarks),
                    ))

        return pose_samples

    def find_pose_sample_outliers(self):
        """Classifies each sample against the entire database."""
        # Find outliers in target poses
        outliers = []
        for sample in self._pose_samples:
            # Find nearest poses for the target one.
            pose_landmarks = sample.landmarks.copy()
            pose_classification = self.__call__(pose_landmarks)
            class_names = [class_name for class_name, count in pose_classification.items() if
                           count == max(pose_classification.values())]

            # Sample is an outlier if nearest poses have different class or more than
            # one pose class is detected as nearest.
            if sample.class_name not in class_names or len(class_names) != 1:
                outliers.append(PoseSampleOutlier(sample, class_names, pose_classification))

        return outliers

    def __call__(self, pose_landmarks):
        """Classifies given pose.

        Classification is done in two stages:
          * First we pick top-N samples by MAX distance. It allows to remove samples
            that are almost the same as given pose, but has few joints bent in the
            other direction.
          * Then we pick top-N samples by MEAN distance. After outliers are removed
            on a previous step, we can pick samples that are closes on average.

        Args:
          pose_landmarks: NumPy array with 3D landmarks of shape (N, 3).

        Returns:
          Dictionary with count of nearest pose samples from the database. Sample:
            {
              'pushups_down': 8,
              'pushups_up': 2,
            }
        """
        # Check that provided and target poses have the same shape.
        assert pose_landmarks.shape == (self._n_landmarks, self._n_dimensions), 'Unexpected shape: {}'.format(
            pose_landmarks.shape)

        # Get given pose embedding.
        pose_embedding = self._pose_embedder(pose_landmarks)
        flipped_pose_embedding = self._pose_embedder(pose_landmarks * np.array([-1, 1, 1]))

        # Filter by max distance.
        #
        # That helps to remove outliers - poses that are almost the same as the
        # given one, but has one joint bent into another direction and actually
        # represnt a different pose class.
        max_dist_heap = []
        for sample_idx, sample in enumerate(self._pose_samples):
            max_dist = min(
                np.max(np.abs(sample.embedding - pose_embedding) * self._axes_weights),
                np.max(np.abs(sample.embedding - flipped_pose_embedding) * self._axes_weights),
            )
            max_dist_heap.append([max_dist, sample_idx])

        max_dist_heap = sorted(max_dist_heap, key=lambda x: x[0])
        max_dist_heap = max_dist_heap[:self._top_n_by_max_distance]

        # Filter by mean distance.
        #
        # After removing outliers we can find the nearest pose by mean distance.
        mean_dist_heap = []
        for _, sample_idx in max_dist_heap:
            sample = self._pose_samples[sample_idx]
            mean_dist = min(
                np.mean(np.abs(sample.embedding - pose_embedding) * self._axes_weights),
                np.mean(np.abs(sample.embedding - flipped_pose_embedding) * self._axes_weights),
            )
            mean_dist_heap.append([mean_dist, sample_idx])

        mean_dist_heap = sorted(mean_dist_heap, key=lambda x: x[0])
        mean_dist_heap = mean_dist_heap[:self._top_n_by_mean_distance]

        # Collect results into map: (class_name -> n_samples)
        class_names = [self._pose_samples[sample_idx].class_name for _, sample_idx in mean_dist_heap]
        result = {class_name: class_names.count(class_name) for class_name in set(class_names)}
        return result


## 5. 姿勢分類的結果平滑
#使用股票分析常會用到的EMA(Exponential Moving Average指數型移動平均)

class EMADictSmoothing(object):
    """Smoothes pose classification."""

    def __init__(self, window_size=10, alpha=0.2):
        self._window_size = window_size
        self._alpha = alpha

        self._data_in_window = []

    def __call__(self, data):
        """Smoothes given pose classification.

        Smoothing is done by computing Exponential Moving Average for every pose
        class observed in the given time window. Missed pose classes arre replaced
        with 0.

        Args:
          data: Dictionary with pose classification. Sample:
              {
                'pushups_down': 8,
                'pushups_up': 2,
              }

        Result:
          Dictionary in the same format but with smoothed and float instead of
          integer values. Sample:
            {
              'pushups_down': 8.3,
              'pushups_up': 1.7,
            }
        """
        # Add new data to the beginning of the window for simpler code.
        self._data_in_window.insert(0, data)
        self._data_in_window = self._data_in_window[:self._window_size]

        # Get all keys.
        keys = set([key for data in self._data_in_window for key, _ in data.items()])

        # Get smoothed values.
        smoothed_data = dict()
        for key in keys:
            factor = 1.0
            top_sum = 0.0
            bottom_sum = 0.0
            for data in self._data_in_window:
                value = data[key] if key in data else 0.0

                top_sum += factor * value
                bottom_sum += factor

                # Update factor.
                factor *= (1.0 - self._alpha)

            smoothed_data[key] = top_sum / bottom_sum

        return smoothed_data

## 5.2 膝蓋平行分數平滑：

class Knees_score_smoothing(object):
    """Smoothes pose classification."""

    def __init__(self, window_size=10, alpha=0.2):
        self._window_size = window_size
        self._alpha = alpha

        self._data_in_window = []

    def __call__(self, data):
        # Add new data to the beginning of the window for simpler code.
        self._data_in_window.insert(0, data)
        self._data_in_window = self._data_in_window[:self._window_size]

        # Get smoothed values.
        factor = 1.0
        top_sum = 0.0
        bottom_sum = 0.0
        for data in self._data_in_window:
            top_sum += factor * data
            bottom_sum += factor

            # Update factor.
            factor *= (1.0 - self._alpha)

        smoothed_data = top_sum / bottom_sum

        return smoothed_data

    def _distance_compare(self, test_list, t_num):
        distance = abs(test_list[t_num][11][1] - test_list[t_num][12][1])
        return distance


## 6.動作計數器

class RepetitionCounter(object):
    """Counts number of repetitions of given target pose class."""

    def __init__(self, class_name, enter_threshold=6, exit_threshold=4):
        self._class_name = class_name

        # If pose counter passes given threshold, then we enter the pose.
        self._enter_threshold = enter_threshold
        self._exit_threshold = exit_threshold

        # Either we are in given pose or not.
        self._pose_entered = False

        # Number of times we exited the pose.
        self._n_repeats = 0

    # ------------------------- #selfchange_ed2_自定義函數:(計算膝蓋平行評分)取出運動計次開關狀態 (2021/12/21)
    def get_switch(self):
        return self._pose_entered

    # -------------------------
    @property
    def n_repeats(self):
        return self._n_repeats

    def __call__(self, pose_classification):
        """Counts number of repetitions happend until given frame.

        We use two thresholds. First you need to go above the higher one to enter
        the pose, and then you need to go below the lower one to exit it. Difference
        between the thresholds makes it stable to prediction jittering (which will
        cause wrong counts in case of having only one threshold).

        Args:
          pose_classification: Pose classification dictionary on current frame.
            Sample:
              {
                'pushups_down': 8.3,
                'pushups_up': 1.7,
              }

        Returns:
          Integer counter of repetitions.
        """
        # Get pose confidence.
        pose_confidence = 0.0
        if self._class_name in pose_classification:
            pose_confidence = pose_classification[self._class_name]

        # On the very first frame or if we were out of the pose, just check if we
        # entered it on this frame and update the state.
        if not self._pose_entered:
            self._pose_entered = pose_confidence > self._enter_threshold
            return self._n_repeats

        # If we were in the pose and are exiting it, then increase the counter and
        # update the state.
        if pose_confidence < self._exit_threshold:
            self._n_repeats += 1
            self._pose_entered = False

        return self._n_repeats


# 7. 資料可視化

class PoseClassificationVisualizer(object):
    """Keeps track of classifcations for every frame and renders them."""

    def __init__(self,
                 class_name,
                 plot_location_x=0.05,
                 plot_location_y=0.05,
                 plot_max_width=0.6,
                 plot_max_height=0.6,
                 plot_figsize=(9, 4),
                 plot_x_max=None,
                 plot_y_max=None,
                 counter_location_x=0.75,
                 counter_location_y=0.10,
                 #                counter_font_path='https://github.com/googlefonts/roboto/blob/main/src/hinted/Roboto-Regular.ttf?raw=true',
                 counter_font_color='red',
                 counter_font_size=0.08):
        self._class_name = class_name
        self._plot_location_x = plot_location_x
        self._plot_location_y = plot_location_y
        self._plot_max_width = plot_max_width
        self._plot_max_height = plot_max_height
        self._plot_figsize = plot_figsize
        self._plot_x_max = plot_x_max
        self._plot_y_max = plot_y_max
        self._counter_location_x = counter_location_x
        self._counter_location_y = counter_location_y
        #         self._counter_font_path = counter_font_path
        self._counter_font_color = counter_font_color
        self._counter_font_size = counter_font_size

        self._counter_font = None

        self._pose_classification_history = []
        self._pose_classification_filtered_history = []

    def __call__(self,
                 frame,
                 pose_classification,
                 pose_classification_filtered,
                 repetitions_count,
                 knees_score):
        """Renders pose classifcation and counter until given frame."""
        # Extend classification history.
        self._pose_classification_history.append(pose_classification)
        self._pose_classification_filtered_history.append(pose_classification_filtered)

        # Output frame with classification plot and counter.
        output_img = Image.fromarray(frame)

        output_width = output_img.size[0]
        output_height = output_img.size[1]

        # Draw the plot.
        img = self._plot_classification_history(output_width, output_height)
        img.thumbnail((int(output_width * self._plot_max_width),
                       int(output_height * self._plot_max_height)),
                      Image.ANTIALIAS)
        output_img.paste(img,
                         (int(output_width * self._plot_location_x),
                          int(output_height * self._plot_location_y)))

        # Draw the count.
        output_img_draw = ImageDraw.Draw(output_img)
        #         if self._counter_font is None:
        #             self._font_size = int(output_height * self._counter_font_size)
        #             font_request = requests.get(self._counter_font_path, allow_redirects=True)     #*****
        #             self._counter_font = ImageFont.truetype(io.BytesIO(font_request.content), size=font_size)     #*****
        self._counter_font = ImageFont.truetype("msjhbd.ttc", int(output_height * self._counter_font_size),
                                                encoding='utf-8')
        output_img_draw.text((output_width * self._counter_location_x,
                              output_height * self._counter_location_y),
                             str(repetitions_count),
                             font=self._counter_font,  # *****
                             fill=self._counter_font_color)

        # ------------------------------- #selfchange_ed2_增加文字出現位置: (2021/12/22)
        text_location_x = 0.65
        text_location_y1 = 0.05
        text_location_y2 = 0.2
        text_location_x2 = 0.75
        text_location_y3 = 0.22
        counter_font = ImageFont.truetype("msjhbd.ttc", int(0.6 * output_height * self._counter_font_size),
                                          encoding='utf-8')
        output_img_draw.text((output_width * text_location_x,
                              output_height * text_location_y1),
                             "次數:",
                             font=counter_font,  # *****
                             fill=self._counter_font_color)

        counter_font = ImageFont.truetype("msjhbd.ttc", int(0.35 * output_height * self._counter_font_size),
                                          encoding='utf-8')
        output_img_draw.text((output_width * text_location_x,
                              output_height * text_location_y2),
                             "膝蓋平行評分:",
                             font=counter_font,  # *****
                             fill=self._counter_font_color)

        counter_font = ImageFont.truetype("msjhbd.ttc", int(0.6 * output_height * self._counter_font_size),
                                          encoding='utf-8')
        output_img_draw.text((output_width * text_location_x2,
                              output_height * text_location_y3),
                             str(knees_score),
                             font=counter_font,  # *****
                             fill=self._counter_font_color)
        # -------------------------------

        return output_img

    def _plot_classification_history(self, output_width, output_height):
        fig = plt.figure(figsize=self._plot_figsize)

        for classification_history in [self._pose_classification_history,
                                       self._pose_classification_filtered_history]:
            y = []
            for classification in classification_history:
                if classification is None:
                    y.append(None)
                elif self._class_name in classification:
                    y.append(classification[self._class_name])
                else:
                    y.append(0)
            # -----------------------------#selfchange_ed1_更改信心分數歷史折線圖顏色 (2021/12/19)
            # 自行更改:更改折線圖顏色:
            if classification_history == self._pose_classification_history:
                plt.plot(y, linewidth=7, color='green')
            elif classification_history == self._pose_classification_filtered_history:
                plt.plot(y, linewidth=7, color='red')
            # -----------------------------

        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Frame')
        plt.ylabel('Confidence')
        plt.title('Classification history for `{}`'.format(self._class_name))
        plt.legend(loc='upper right')

        if self._plot_y_max is not None:
            plt.ylim(top=self._plot_y_max)
        if self._plot_x_max is not None:
            plt.xlim(right=self._plot_x_max)

        # Convert plot to image.
        buf = io.BytesIO()
        dpi = min(
            output_width * self._plot_max_width / float(self._plot_figsize[0]),
            output_height * self._plot_max_height / float(self._plot_figsize[1]))
        fig.savefig(buf, dpi=dpi)
        buf.seek(0)
        img = Image.open(buf)
        plt.close()

        return img


## 8. mediapipe 提取訓練集關鍵點座標

class BootstrapHelper(object):
    """Helps to bootstrap images and filter pose samples for classification."""

    def __init__(self,
                 images_in_folder,
                 images_out_folder,
                 csvs_out_folder):
        self._images_in_folder = images_in_folder
        self._images_out_folder = images_out_folder
        self._csvs_out_folder = csvs_out_folder

        # Get list of pose classes and print image statistics.
        self._pose_class_names = sorted([n for n in os.listdir(self._images_in_folder) if not n.startswith('.')])

    def bootstrap(self, per_pose_class_limit=None):
        """Bootstraps images in a given folder.

        Required image in folder (same use for image out folder):
          pushups_up/
            image_001.jpg
            image_002.jpg
            ...
          pushups_down/
            image_001.jpg
            image_002.jpg
            ...
          ...

        Produced CSVs out folder:
          pushups_up.csv
          pushups_down.csv

        Produced CSV structure with pose 3D landmarks:
          sample_00001,x1,y1,z1,x2,y2,z2,....
          sample_00002,x1,y1,z1,x2,y2,z2,....
        """
        # Create output folder for CVSs.
        if not os.path.exists(self._csvs_out_folder):
            os.makedirs(self._csvs_out_folder)

        for pose_class_name in self._pose_class_names:
            print('Bootstrapping ', pose_class_name, file=sys.stderr)

            # Paths for the pose class.
            images_in_folder = os.path.join(self._images_in_folder, pose_class_name)
            images_out_folder = os.path.join(self._images_out_folder, pose_class_name)
            csv_out_path = os.path.join(self._csvs_out_folder, pose_class_name + '.csv')
            if not os.path.exists(images_out_folder):
                os.makedirs(images_out_folder)

            with open(csv_out_path, 'w', newline='') as csv_out_file:
                csv_out_writer = csv.writer(csv_out_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                # Get list of images.
                image_names = sorted([n for n in os.listdir(images_in_folder) if not n.startswith('.')])
                if per_pose_class_limit is not None:
                    image_names = image_names[:per_pose_class_limit]

                # Bootstrap every image.
                for image_name in tqdm.tqdm(image_names):
                    # Load image.
                    input_frame = cv2.imread(os.path.join(images_in_folder, image_name))
                    input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)

                    # Initialize fresh pose tracker and run it.
                    with mp_pose.Pose(upper_body_only=False) as pose_tracker:
                        result = pose_tracker.process(image=input_frame)
                        pose_landmarks = result.pose_landmarks

                    # Save image with pose prediction (if pose was detected).
                    output_frame = input_frame.copy()
                    if pose_landmarks is not None:
                        mp_drawing.draw_landmarks(
                            image=output_frame,
                            landmark_list=pose_landmarks,
                            connections=mp_pose.POSE_CONNECTIONS)
                    output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(images_out_folder, image_name), output_frame)

                    # Save landmarks if pose was detected.
                    if pose_landmarks is not None:
                        # Get landmarks.
                        frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
                        pose_landmarks = np.array(
                            [[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                             for lmk in pose_landmarks.landmark],
                            dtype=np.float32)
                        assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(
                            pose_landmarks.shape)
                        csv_out_writer.writerow([image_name] + pose_landmarks.flatten().astype(np.str).tolist())

                    # Draw XZ projection and concatenate with the image.
                    projection_xz = self._draw_xz_projection(
                        output_frame=output_frame, pose_landmarks=pose_landmarks)
                    output_frame = np.concatenate((output_frame, projection_xz), axis=1)

    def _draw_xz_projection(self, output_frame, pose_landmarks, r=0.5, color='red'):
        frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
        img = Image.new('RGB', (frame_width, frame_height), color='white')

        if pose_landmarks is None:
            return np.asarray(img)

        # Scale radius according to the image width.
        r *= frame_width * 0.01

        draw = ImageDraw.Draw(img)
        for idx_1, idx_2 in mp_pose.POSE_CONNECTIONS:
            # Flip Z and move hips center to the center of the image.
            x1, y1, z1 = pose_landmarks[idx_1] * [1, 1, -1] + [0, 0, frame_height * 0.5]
            x2, y2, z2 = pose_landmarks[idx_2] * [1, 1, -1] + [0, 0, frame_height * 0.5]

            draw.ellipse([x1 - r, z1 - r, x1 + r, z1 + r], fill=color)
            draw.ellipse([x2 - r, z2 - r, x2 + r, z2 + r], fill=color)
            draw.line([x1, z1, x2, z2], width=int(r), fill=color)

        return np.asarray(img)

    def align_images_and_csvs(self, print_removed_items=False):
        """Makes sure that image folders and CSVs have the same sample.

        Leaves only intersetion of samples in both image folders and CSVs.
        """
        for pose_class_name in self._pose_class_names:
            # Paths for the pose class.
            images_out_folder = os.path.join(self._images_out_folder, pose_class_name)
            csv_out_path = os.path.join(self._csvs_out_folder, pose_class_name + '.csv')

            # Read CSV into memory.
            rows = []
            with open(csv_out_path) as csv_out_file:
                csv_out_reader = csv.reader(csv_out_file, delimiter=',')
                for row in csv_out_reader:
                    rows.append(row)

            # Image names left in CSV.
            image_names_in_csv = []

            # Re-write the CSV removing lines without corresponding images.
            with open(csv_out_path, 'w', newline='') as csv_out_file:
                csv_out_writer = csv.writer(csv_out_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                for row in rows:
                    if row != []:
                        image_name = row[0]
                        image_path = os.path.join(images_out_folder, image_name)
                        if os.path.exists(image_path):
                            image_names_in_csv.append(image_name)
                            csv_out_writer.writerow(row)
                        elif print_removed_items:
                            print('Removed image from CSV: ', image_path)

            # Remove images without corresponding line in CSV.
            for image_name in os.listdir(images_out_folder):
                if image_name not in image_names_in_csv:
                    image_path = os.path.join(images_out_folder, image_name)
                    os.remove(image_path)
                    if print_removed_items:
                        print('Removed image from folder: ', image_path)

    def analyze_outliers(self, outliers):
        """Classifies each sample agains all other to find outliers.

        If sample is classified differrrently than the original class - it sould
        either be deleted or more similar samples should be aadded.
        """
        for outlier in outliers:
            image_path = os.path.join(self._images_out_folder, outlier.sample.class_name, outlier.sample.name)

            print('Outlier')
            print('  sample path =    ', image_path)
            print('  sample class =   ', outlier.sample.class_name)
            print('  detected class = ', outlier.detected_class)
            print('  all classes =    ', outlier.all_classes)

            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            show_image(img, figsize=(20, 20))

    def remove_outliers(self, outliers):
        """Removes outliers from the image folders."""
        for outlier in outliers:
            image_path = os.path.join(self._images_out_folder, outlier.sample.class_name, outlier.sample.name)
            os.remove(image_path)

    def print_images_in_statistics(self):
        """Prints statistics from the input image folder."""
        self._print_images_statistics(self._images_in_folder, self._pose_class_names)

    def print_images_out_statistics(self):
        """Prints statistics from the output image folder."""
        self._print_images_statistics(self._images_out_folder, self._pose_class_names)

    def _print_images_statistics(self, images_folder, pose_class_names):
        print('Number of images per pose class:')
        for pose_class_name in pose_class_names:
            n_images = len([
                n for n in os.listdir(os.path.join(images_folder, pose_class_name))
                if not n.startswith('.')])
            print('  {}: {}'.format(pose_class_name, n_images))


## 9. 建立分類器

def squat_data_to_csv(train_folder_path):
    # 指定訓練集的路徑:
    bootstrap_images_in_folder = train_folder_path #輸入本來給的附帶資料夾名稱 'squat_data'

    # bootrap圖片得到的csv檔案輸出的資料夾:
    bootstrap_images_out_folder = 'squat_images_out'
    bootstrap_csvs_out_folder = 'squat_csvs_out'

    # 初始化 bootstrap的 class:
    bootstrap_helper = BootstrapHelper(
        images_in_folder=bootstrap_images_in_folder,
        images_out_folder=bootstrap_images_out_folder,
        csvs_out_folder=bootstrap_csvs_out_folder,
    )

    # 提取特征
    bootstrap_helper.bootstrap(per_pose_class_limit=None)

    # Align CSVs with filtered images.
    bootstrap_helper.align_images_and_csvs(print_removed_items=False)
    bootstrap_helper.print_images_out_statistics()


## 10. 生成CSV檔案

def dump_for_the_app():
    pose_samples_folder = 'squat_csvs_out'
    pose_samples_csv_path = 'squat_csvs_out_basic.csv'
    file_extension = 'csv'
    file_separator = ','

    # Each file in the folder represents one pose class.
    file_names = [name for name in os.listdir(pose_samples_folder) if name.endswith(file_extension)]

    with open(pose_samples_csv_path, 'w',newline='') as csv_out:
        csv_out_writer = csv.writer(csv_out, delimiter=file_separator, quoting=csv.QUOTE_MINIMAL)
        for file_name in file_names:
            # Use file name as pose class name.
            class_name = file_name[:-(len(file_extension) + 1)]

            # One file line: `sample_00001,x1,y1,x2,y2,....`.
            with open(os.path.join(pose_samples_folder, file_name)) as csv_in:
                csv_in_reader = csv.reader(csv_in, delimiter=file_separator)
                for row in csv_in_reader:
                    row.insert(1, class_name)
                    csv_out_writer.writerow(row)


## 11. 對拍攝的影片進行預測

def video_prediction_output_video(video_path='IMG_0914.MOV', out_video_path='IMG_0914-output_t3.MOV'):
    # 指定影片路徑以及輸出路徑的名稱
    video_path = video_path
    class_name = 'down'
    out_video_path = out_video_path

    # 讀入影片
    video_cap = cv2.VideoCapture(video_path)
    # 得到影片參數，方便後續作預測時產生輸出影片用:
    video_n_frames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_fps = video_cap.get(cv2.CAP_PROP_FPS)
    video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 各個函數初始化:
    # Folder with pose class CSVs. That should be the same folder you using while
    # building classifier to output CSVs.
    pose_samples_folder = 'squat_csvs_out'

    # Initialize tracker.
    pose_tracker = mp_pose.Pose(upper_body_only=False)

    # Initialize embedder.
    pose_embedder = FullBodyPoseEmbedder()

    # Initialize classifier.
    # Ceck that you are using the same parameters as during bootstrapping.
    pose_classifier = PoseClassifier(
        pose_samples_folder=pose_samples_folder,
        pose_embedder=pose_embedder,
        #     top_n_by_max_distance=30,
        top_n_by_mean_distance=10)

    # # Uncomment to validate target poses used by classifier and find outliers.
    # outliers = pose_classifier.find_pose_sample_outliers()
    # print('Number of pose sample outliers (consider removing them): ', len(outliers))

    # Initialize EMA smoothing.
    pose_classification_filter = EMADictSmoothing(
        window_size=10,
        alpha=0.2)

    # --------------------------- selfchange_ed2_膝蓋平行分數平滑
    # Initialize knees score smoothing.
    knees_score_filter = Knees_score_smoothing(window_size=10,
                                               alpha=0.2)
    # ---------------------------

    # 指定動作的兩個閾值
    repetition_counter = RepetitionCounter(
        class_name=class_name,
        enter_threshold=6,
        exit_threshold=4)

    # Initialize renderer.
    pose_classification_visualizer = PoseClassificationVisualizer(
        class_name=class_name,
        plot_x_max=video_n_frames,
        # Graphic looks nicer if it's the same as `top_n_by_mean_distance`.
        plot_y_max=10)

    # 開始每幀圖片的分析，最終儲存成影片
    # Open output video.
    out_video = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (video_width, video_height))

    frame_idx = 0
    output_frame = None

    with tqdm.tqdm(total=video_n_frames, position=0, leave=True) as pbar:
        # ========================= #selfchange_ed2_各種拿來應用的初始值設定 (2021/12/22)
        test2_list = []  # [not_necessary]作圖偵錯用,實際版本可註解
        switch_list = []  # [not_necessary]作圖偵錯用,實際版本可註解
        switch_open_list = []
        switch = False
        knees_avg_score = 0
        knees_min_score = 0
        # =========================
        while True:
            # Get next frame of the video.
            success, input_frame = video_cap.read()
            if not success:
                break

            # Run pose tracker.
            input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
            result = pose_tracker.process(image=input_frame)
            pose_landmarks = result.pose_landmarks

            # Draw pose prediction.
            output_frame = input_frame.copy()
            if pose_landmarks is not None:
                mp_drawing.draw_landmarks(
                    image=output_frame,
                    landmark_list=pose_landmarks,
                    connections=mp_pose.POSE_CONNECTIONS)

            if pose_landmarks is not None:
                # Get landmarks.
                frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
                pose_landmarks = np.array([[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                                           for lmk in pose_landmarks.landmark], dtype=np.float32)
                assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)

                # Classify the pose on the current frame.
                pose_classification = pose_classifier(pose_landmarks)
                # ------------------------------- #selfchange_ed1&ed2_自定義函數實際套用的位置 (功能: 計算與正確姿勢的距離誤差)
                # 一樣使用 pose_classifier 進行自訂函數的套用測試:
                # 自訂函數功能: 替(影片單張frame的pose_landmarks)找出類似角度的正確姿勢,平均後拿來與frame求出所有距離特徵誤差:
                pose_distance_error = pose_classifier.compare_correct_pose_distance(
                    pose_landmarks=pose_landmarks,
                    class_name=class_name)
                knees_analyze = pose_classifier.find_knees_score(pose_distance_error)
                knees_score_filtered = knees_score_filter(knees_analyze[1])

                test2_list.append(pose_distance_error)  # [not_necessary]作圖偵錯用,實際版本可註解

                # -------------------------------

                # Smooth classification using EMA.
                pose_classification_filtered = pose_classification_filter(pose_classification)

                # Count repetitions.
                repetitions_count = repetition_counter(pose_classification_filtered)

                # ------------------------------- #selfchange_ed2_各種開關資訊獲取 (2021/12/22)
                switch = repetition_counter.get_switch()
                switch_list.append(switch)
                if switch == True:
                    switch_open_list.append(knees_score_filtered)
                if len(switch_list) >= 2:
                    if switch_list[-2] == True:
                        if switch == False:
                            print(switch_open_list)
                            knees_min_score = np.min(switch_open_list)
                            knees_avg_score = np.average(switch_open_list)

                            switch_open_list = []
                # -------------------------------

            else:
                # No pose => no classification on current frame.
                pose_classification = None

                # Still add empty classification to the filter to maintaing correct
                # smoothing for future frames.
                pose_classification_filtered = pose_classification_filter(dict())
                pose_classification_filtered = None

                # Don't update the counter presuming that person is 'frozen'. Just
                # take the latest repetitions count.
                repetitions_count = repetition_counter.n_repeats

            # Draw classification plot and repetition counter.
            output_frame = pose_classification_visualizer(
                frame=output_frame,
                pose_classification=pose_classification,
                pose_classification_filtered=pose_classification_filtered,
                repetitions_count=repetitions_count,
                knees_score=math.floor(knees_min_score))

            # Save the output frame.
            out_video.write(cv2.cvtColor(np.array(output_frame), cv2.COLOR_RGB2BGR))

            # Show intermediate frames of the video to track progress.
            if frame_idx % 50 == 0:
                # show_image(output_frame)
                # ----------------------- #selfchange_ed2_測試練習套用位子 (2021/12/21)
                print(switch)
                if frame_idx % 100 == 0:
                    print(pose_distance_error)
                    print(f'knees_analyze: (1)error_distance: {knees_analyze[0]}, (2)knees_score: {knees_analyze[1]}')
                    print(f'knees_score_filtered: {knees_score_filtered}')
                    print(f'knees_min_score:{knees_min_score},knees_avg_score:{knees_avg_score}')
                # -----------------------
            frame_idx += 1
            pbar.update()

    # Close output video.
    out_video.release()

    # Release MediaPipe resources.
    pose_tracker.close()

    # Show the last frame of the video.
    if output_frame is not None:
        show_image(output_frame)
    return "All the work is done!"


if __name__ == "__main__":
    # squat_data_to_csv('squat_data') #輸入的參數: 儲存訓練資料的資料夾名稱 (放入附帶檔案)
    # dump_for_the_app()
    video_prediction_output_video(video_path='IMG_0914.MOV', out_video_path='IMG_0914-output_t3.MOV') #第一個參數為準備分析的影片檔路徑
                                                                                            #第二個參數為完成分析的影片輸出檔路徑