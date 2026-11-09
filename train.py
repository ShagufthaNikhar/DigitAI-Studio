"""Train CNN and MLP baseline models on MNIST and save evaluation artifacts."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)
tf.keras.utils.set_random_seed(42)


def build_cnn():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(10, activation="softmax"),
    ], name="CNN")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_mlp():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ], name="MLP_Baseline")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    print("Loading MNIST...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train_cnn, x_test_cnn = x_train[..., None], x_test[..., None]

    callback = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)]

    print("\nTraining CNN...")
    cnn = build_cnn()
    hist_cnn = cnn.fit(x_train_cnn, y_train, validation_split=0.1, epochs=8,
                       batch_size=128, callbacks=callback, verbose=1)
    cnn_loss, cnn_acc = cnn.evaluate(x_test_cnn, y_test, verbose=0)
    cnn_probs = cnn.predict(x_test_cnn, batch_size=256, verbose=0)
    cnn_pred = np.argmax(cnn_probs, axis=1)
    cnn.save(MODEL_DIR / "mnist_cnn.keras")

    print("\nTraining MLP baseline...")
    mlp = build_mlp()
    hist_mlp = mlp.fit(x_train, y_train, validation_split=0.1, epochs=8,
                       batch_size=128, callbacks=callback, verbose=1)
    mlp_loss, mlp_acc = mlp.evaluate(x_test, y_test, verbose=0)
    mlp.save(MODEL_DIR / "mnist_mlp.keras")

    metrics = {
        "cnn_test_accuracy": float(cnn_acc),
        "cnn_test_loss": float(cnn_loss),
        "mlp_test_accuracy": float(mlp_acc),
        "mlp_test_loss": float(mlp_loss),
        "cnn_epochs_trained": len(hist_cnn.history["loss"]),
        "mlp_epochs_trained": len(hist_mlp.history["loss"]),
    }
    (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report = classification_report(y_test, cnn_pred, digits=4)
    (REPORT_DIR / "classification_report.txt").write_text(
        f"CNN test accuracy: {cnn_acc:.4%}\nCNN test loss: {cnn_loss:.4f}\n\n{report}",
        encoding="utf-8"
    )

    # Training curves for CNN
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(hist_cnn.history["accuracy"], label="Training")
    axes[0].plot(hist_cnn.history["val_accuracy"], label="Validation")
    axes[0].set_title("CNN Accuracy"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[1].plot(hist_cnn.history["loss"], label="Training")
    axes[1].plot(hist_cnn.history["val_loss"], label="Validation")
    axes[1].set_title("CNN Loss"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend()
    fig.tight_layout(); fig.savefig(REPORT_DIR / "training_curves.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(confusion_matrix(y_test, cnn_pred), display_labels=np.arange(10)).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title("CNN Confusion Matrix — MNIST Test Set")
    fig.tight_layout(); fig.savefig(REPORT_DIR / "confusion_matrix.png", dpi=160); plt.close(fig)

    print("\nTraining complete.")
    print(f"CNN test accuracy: {cnn_acc:.2%}")
    print(f"MLP test accuracy: {mlp_acc:.2%}")
    print(f"Saved models to {MODEL_DIR}")
    print(f"Saved reports to {REPORT_DIR}")


if __name__ == "__main__":
    main()
