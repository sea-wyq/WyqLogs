import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import npu_device

# 初始化NPU并设置为默认设备
npu = npu_device.open().as_default()
print(f"使用NPU设备: {npu.name()}")

# 设置环境变量
os.environ["ASCEND_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def generate_synthetic_data(num_samples, image_shape=(28, 28, 1), num_classes=10):
    """生成模拟数据"""
    images = np.random.rand(num_samples, *image_shape).astype('float32')
    labels = np.random.randint(0, num_classes, size=num_samples)
    return images, labels


# 生成并预处理数据
train_images, train_labels = generate_synthetic_data(60000)
test_images, test_labels = generate_synthetic_data(10000)

train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)


# 低版本TF兼容的数据管道
def prepare_dataset(images, labels, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle:
        dataset = dataset.shuffle(10000)
    dataset = dataset.batch(64)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

train_dataset = prepare_dataset(train_images, train_labels)
test_dataset = prepare_dataset(test_images, test_labels, shuffle=False)


# 低版本TF兼容：使用device函数而非tf.context.device
with tf.device(npu.name()):  # 替换tf.context.device为tf.device
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )


# 验证模型设备
print(f"模型权重设备: {model.layers[0].weights[0].device}")


# 训练与评估函数
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = model.compiled_loss(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    model.compiled_metrics.update_state(labels, predictions)
    return loss


@tf.function
def test_step(images, labels):
    predictions = model(images, training=False)
    loss = model.compiled_loss(labels, predictions)
    model.compiled_metrics.update_state(labels, predictions)
    return loss


# 执行训练（使用tf.device指定设备）
print("\n开始NPU训练...")
for epoch in range(5):
    # 训练阶段
    train_loss = 0.0
    train_steps = 0
    with tf.device(npu.name()):  # 替换为tf.device
        for images, labels in train_dataset:
            loss = train_step(images, labels)
            train_loss += loss.numpy()
            train_steps += 1
    
    # 验证阶段
    test_loss = 0.0
    test_steps = 0
    with tf.device(npu.name()):  # 替换为tf.device
        for images, labels in test_dataset:
            loss = test_step(images, labels)
            test_loss += loss.numpy()
            test_steps += 1
    
    # 计算指标
    train_acc = model.metrics[0].result().numpy()
    model.metrics[0].reset_states()
    test_acc = model.metrics[0].result().numpy()
    model.metrics[0].reset_states()
    
    print(f"Epoch {epoch+1}/5")
    print(f"训练损失: {train_loss/train_steps:.4f}, 训练准确率: {train_acc:.4f}")
    print(f"测试损失: {test_loss/test_steps:.4f}, 测试准确率: {test_acc:.4f}\n")


# model.save('npu_trained_model.h5')
# print(f"模型已保存，最终测试准确率: {test_acc:.4f}")