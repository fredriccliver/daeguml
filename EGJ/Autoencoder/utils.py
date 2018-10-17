# utils.py

import os
import errno
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# PyCharm IDE 기준 (IntelliJ) 전용 Root Path
PROJECT_ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
OUTPUT_DIR = "outputs"

# matplotlib.pyplot 한글 폰트 적용
# fontList.json (fontList-v300.json) 파일에 폰트 설치
# 폰트 설정 경로: matplotlib.get_configdir()
# 폰트 캐시 경로: matplotlib.get_cachedir()
plt.rcParams["font.family"] = "AppleGothic"

# matplotlib.pyplot 그래프에서 마이너스 폰트 깨지는 문제에 대한 대처
plt.rcParams["axes.unicode_minus"] = False


# 파일 이름 확인 (in FILE_NAME), PRIVATE METHOD
def _file_base_name(file_name):
    if "." in file_name:
        separator_index = file_name.index(".")
        base_name = file_name[:separator_index]
        return base_name
    else:
        return file_name


# 파일 이름 확인 (in PATH)
def path_base_name(path):
    return _file_base_name(os.path.basename(path))


# 일관된 출력을 위해 유사난수 초기화
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


# 28x28 흑백 이미지를 그리기 위한 함수
def plot_image(image, shape=[28, 28]):
    plt.imshow(image.reshape(shape), cmap="Greys", interpolation="nearest")
    plt.axis("off")


# 이미지를 그리기 위한 함수
def plot_multiple_images(images, n_rows, n_cols, pad=2):
    images = images - images.min()  # 최소값을 0으로 만들어 패딩이 하얗게 보이도록 합니다.
    w,h = images.shape[1:]
    image = np.zeros(((w+pad)*n_rows+pad, (h+pad)*n_cols+pad))
    for y in range(n_rows):
        for x in range(n_cols):
            image[(y*(h+pad)+pad):(y*(h+pad)+pad+h),(x*(w+pad)+pad):(x*(w+pad)+pad+w)] = images[y*n_cols+x]
    plt.imshow(image, cmap="Greys", interpolation="nearest")
    plt.axis("off")


def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch


# 경로 체크 및 생성
def os_path_exists(path):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


# 그림 저장; matplotlib.pyplot
def save_fig(file_name, root_path=PROJECT_ROOT_DIR, tight_layout=True):
    # 그림을 저장할 폴더
    path = os.path.join(root_path, OUTPUT_DIR, 'images', file_name + '.png')
    os_path_exists(path)

    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


# 파일 저장; tensorflow
def save_tf(sess, file_name, root_path=PROJECT_ROOT_DIR):
    file_path = os.path.join(root_path, OUTPUT_DIR, 'tf_sess', file_name + '.ckpt')
    os_path_exists(file_path)

    saver = tf.train.Saver()
    saver.save(sess, file_path)


# 파일 로드; tensorflow
def load_tf(sess, file_name, root_path=PROJECT_ROOT_DIR):
    file_path = os.path.join(root_path, OUTPUT_DIR, 'tf_sess', file_name + '.ckpt')
    saver = tf.train.Saver()
    try:
        saver.restore(sess, file_path)
    except Exception as e:
        print(e)
        raise


# 파일 로드; tensorflow, 사전학습용
def load_pre_tf(var_list, sess, file_name, root_path=PROJECT_ROOT_DIR):
    file_path = os.path.join(root_path, OUTPUT_DIR, 'tf_sess', file_name + '.ckpt')
    saver = tf.train.Saver(var_list)
    try:
        saver.restore(sess, file_path)
    except Exception as e:
        print(e)
        raise


# 모델을 로드하고 테스트 세트에서 이를 평가; 재구성 오차를 측정; 재구성 이미지 드로잉
def show_reconstructed_digits(outputs, X, X_test, model_path=None, root_path=PROJECT_ROOT_DIR, n_test_digits=2):
    with tf.Session() as sess:
        if model_path:
            load_tf(sess, model_path, root_path=root_path)

        # outputs_val
        outputs.eval(feed_dict={X: X_test[:n_test_digits]})

    # fig
    plt.figure(figsize=(8, 3 * n_test_digits))
    for digit_index in range(n_test_digits):
        plt.subplot(n_test_digits, 2, digit_index * 2 + 1)
        plot_image(X_test[digit_index])
        plt.subplot(n_test_digits, 2, digit_index * 2 + 2)
        plot_image(X_test[digit_index])
