"""
MATH 320 Final Project notebook coverted into pyhton script with multi processing to allow us to run faster and compare interation effects on the models.
"""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------

def load_and_prepare_data(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load URL data from csv_path, clean and standardize features.

    Returns:
        A: (n, d) standardized feature matrix
        b: (n,) binary labels as float (0 or 1)
    """
    url = pd.read_csv(csv_path, encoding="latin1")

    # Map has_suspicious_word to 0/1
    url["has_suspicious_word"] = url["has_suspicious_word"].map(
        {True: 1, False: 0, "True": 1, "False": 0}
    )

    feature_cols = [
        "url_length", "num_digits", "digit_ratio", "special_char_ratio",
        "num_hyphens", "num_underscores", "num_slashes", "num_dots",
        "num_question_marks", "num_equals", "num_at_symbols", "num_percent",
        "num_hashes", "num_ampersands", "num_subdomains",
        "is_https", "has_suspicious_word",
    ]

    url_features = url[feature_cols].astype(float)
    url_response = url["status"].fillna(0).astype(int)

    # Standardize features
    x_mean = url_features.mean(axis=0)
    x_std_vec = url_features.std(axis=0) + 1e-8
    x_std = (url_features - x_mean) / x_std_vec

    # Drop rows with any NaNs
    mask = x_std.notna().all(axis=1) & url_response.notna()
    A = x_std[mask].values
    b = url_response[mask].values.astype(float)

    return A, b


# ---------------------------------------------------------------------
# Core logistic regression helpers
# ---------------------------------------------------------------------

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def logistic_loss(w: np.ndarray,
                  b0: float,
                  A: np.ndarray,
                  b: np.ndarray,
                  lam: float = 0.0) -> float:
    """
    Negative log-likelihood + 0.5 * lam * ||w||^2.
    """
    z = A @ w + b0
    p = sigmoid(z)
    eps = 1e-12
    n = len(b)
    nll = -np.mean(b * np.log(p + eps) + (1 - b) * np.log(1 - p + eps))
    reg = 0.5 * lam * np.sum(w ** 2)
    return nll + reg


def logistic_grad(w: np.ndarray,
                  b0: float,
                  A: np.ndarray,
                  b: np.ndarray,
                  lam: float = 0.0) -> tuple[np.ndarray, float]:
    """
    Gradient of the regularized logistic loss with respect to w and b0.
    """
    n = A.shape[0]
    z = A @ w + b0
    p = sigmoid(z)
    grad_w = A.T @ (p - b) / n + lam * w
    grad_b = float(np.mean(p - b))
    return grad_w, grad_b


def logistic_hessian_w(w: np.ndarray,
                       b0: float,
                       A: np.ndarray,
                       b: np.ndarray,
                       lam: float = 0.0) -> np.ndarray:
    """
    Hessian of the logistic loss w.r.t. w (d x d matrix):
    H = (1/n) A^T diag(p(1-p)) A + lam I
    """
    n, d = A.shape
    z = A @ w + b0
    p = sigmoid(z)
    s = p * (1 - p)         # (n,)
    AS = A * s[:, None]     # each row scaled by s_i
    H = (AS.T @ A) / n
    H += lam * np.eye(d)
    return H


def predict_proba(A: np.ndarray,
                  w: np.ndarray,
                  b0: float) -> np.ndarray:
    return sigmoid(A @ w)


def predict_label(A: np.ndarray,
                  w: np.ndarray,
                  b0: float,
                  threshold: float = 0.5) -> np.ndarray:
    return (predict_proba(A, w, b0) >= threshold).astype(int)


def accuracy(A: np.ndarray,
             b: np.ndarray,
             w: np.ndarray,
             b0: float) -> float:
    return float((predict_label(A, w, b0) == b).mean())


# ---------------------------------------------------------------------
# Optimization methods
# ---------------------------------------------------------------------

# ----- Batch Gradient Descent -----


def gd_step(w: np.ndarray,
            b0: float,
            A: np.ndarray,
            b: np.ndarray,
            lr: float,
            lam: float = 0.0) -> tuple[np.ndarray, float]:
    grad_w, grad_b = logistic_grad(w, b0, A, b, lam)
    return w - lr * grad_w, b0 - lr * grad_b


def fit_batch_gd(A: np.ndarray,
                 b: np.ndarray,
                 lr: float = 0.1,
                 n_iters: int = 40,
                 lam: float = 0.0) -> tuple[np.ndarray, float, list[float], list[float]]:
    n, d = A.shape
    w = np.zeros(d)
    b0 = 0.0
    loss_hist: list[float] = []
    acc_hist: list[float] = []

    for _ in range(n_iters):
        w, b0 = gd_step(w, b0, A, b, lr, lam)
        loss_hist.append(logistic_loss(w, b0, A, b, lam))
        acc_hist.append(accuracy(A, b, w, b0))

    return w, b0, loss_hist, acc_hist


# ----- Mini-batch SGD -----


def make_minibatches(n: int,
                     batch_size: int,
                     rng: np.random.Generator) -> list[np.ndarray]:
    perm = rng.permutation(n)
    return [perm[i:i + batch_size] for i in range(0, n, batch_size)]


def sgd_minibatch_step(w: np.ndarray,
                       b0: float,
                       A_batch: np.ndarray,
                       b_batch: np.ndarray,
                       lr: float,
                       lam: float = 0.0) -> tuple[np.ndarray, float]:
    grad_w, grad_b = logistic_grad(w, b0, A_batch, b_batch, lam)
    return w - lr * grad_w, b0 - lr * grad_b


def fit_minibatch_sgd(A: np.ndarray,
                      b: np.ndarray,
                      lr: float = 0.05,
                      n_epochs: int = 10,
                      batch_size: int = 512,
                      lam: float = 0.0,
                      seed: int = 0) -> tuple[np.ndarray, float, list[float], list[float]]:
    n, d = A.shape
    w = np.zeros(d)
    b0 = 0.0
    rng = np.random.default_rng(seed)

    loss_hist: list[float] = []
    acc_hist: list[float] = []

    for _ in range(n_epochs):
        batches = make_minibatches(n, batch_size, rng)
        for idx in batches:
            A_batch = A[idx]
            b_batch = b[idx]
            w, b0 = sgd_minibatch_step(w, b0, A_batch, b_batch, lr, lam)

        loss_hist.append(logistic_loss(w, b0, A, b, lam))
        acc_hist.append(accuracy(A, b, w, b0))

    return w, b0, loss_hist, acc_hist


# ----- Newton's Method -----


def newton_step(w: np.ndarray,
                b0: float,
                A: np.ndarray,
                b: np.ndarray,
                lam: float = 0.0) -> tuple[np.ndarray, float]:
    grad_w, grad_b = logistic_grad(w, b0, A, b, lam)
    H = logistic_hessian_w(w, b0, A, b, lam)
    try:
        delta_w = np.linalg.solve(H, grad_w)
    except np.linalg.LinAlgError:
        # Fallback to gradient step if Hessian is singular
        delta_w = grad_w
    w_new = w - delta_w

    # Scalar second derivative w.r.t. b0
    z = A @ w + b0
    p = sigmoid(z)
    s = p * (1 - p)
    H_b = float(np.mean(s))
    if H_b > 0:
        delta_b = grad_b / H_b
        b0_new = b0 - delta_b
    else:
        b0_new = b0

    return w_new, b0_new


def fit_newton(A: np.ndarray,
               b: np.ndarray,
               n_iters: int = 8,
               lam: float = 0.0) -> tuple[np.ndarray, float, list[float], list[float]]:
    n, d = A.shape
    w = np.zeros(d)
    b0 = 0.0

    loss_hist: list[float] = []
    acc_hist: list[float] = []

    for _ in range(n_iters):
        w, b0 = newton_step(w, b0, A, b, lam)
        loss_hist.append(logistic_loss(w, b0, A, b, lam))
        acc_hist.append(accuracy(A, b, w, b0))

    return w, b0, loss_hist, acc_hist


# ---------------------------------------------------------------------
# Wrappers for multiprocessing
# ---------------------------------------------------------------------

def train_gd(A: np.ndarray,
             b: np.ndarray,
             lam: float = 0.0):
    w, b0, loss_hist, acc_hist = fit_batch_gd(A, b, lr=0.1, n_iters=40, lam=lam)
    return ("gd", w, b0, loss_hist, acc_hist)


def train_sgd(A: np.ndarray,
              b: np.ndarray,
              lam: float = 0.0):
    w, b0, loss_hist, acc_hist = fit_minibatch_sgd(
        A, b, lr=0.05, n_epochs=10, batch_size=512, lam=lam, seed=0
    )
    return ("sgd", w, b0, loss_hist, acc_hist)


def train_newton(A: np.ndarray,
                 b: np.ndarray,
                 lam: float = 0.0):
    w, b0, loss_hist, acc_hist = fit_newton(A, b, n_iters=8, lam=lam)
    return ("newton", w, b0, loss_hist, acc_hist)


def run_all_methods(A: np.ndarray,
                    b: np.ndarray,
                    lam: float = 0.0):
    """
    Run GD, SGD, and Newton in parallel using separate processes.
    """
    with ProcessPoolExecutor(max_workers=3) as ex:
        futures = [
            ex.submit(train_gd, A, b, lam),
            ex.submit(train_sgd, A, b, lam),
            ex.submit(train_newton, A, b, lam),
        ]
        results = [f.result() for f in futures]
    return results


# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------

def make_plots(loss_gd, loss_sgd, loss_new,
               acc_gd, acc_sgd, acc_new,
               img_dir: str):
    os.makedirs(img_dir, exist_ok=True)

    # Loss plot
    plt.figure()
    plt.plot(loss_gd, label="Batch GD")
    plt.plot(loss_sgd, label="Mini-batch SGD (per epoch)")
    plt.plot(loss_new, label="Newton")
    plt.title("Training Loss: GD vs SGD vs Newton")
    plt.xlabel("Iteration / Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(img_dir, "loss_comparison.png"),
                dpi=300, bbox_inches="tight")
    plt.show()

    # Accuracy plot
    plt.figure()
    plt.plot(acc_gd, label="Batch GD")
    plt.plot(acc_sgd, label="Mini-batch SGD (per epoch)")
    plt.plot(acc_new, label="Newton")
    plt.title("Training Accuracy: GD vs SGD vs Newton")
    plt.xlabel("Iteration / Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(img_dir, "accuracy_comparison.png"),
                dpi=300, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Paths relative to this script
    CSV_PATH = os.path.join("..", "data", "urls.csv")
    IMG_DIR = os.path.join("..", "img")

    print("Loading data from:", CSV_PATH)
    A, b = load_and_prepare_data(CSV_PATH)
    print("Data loaded. Shape A:", A.shape, "Shape b:", b.shape)

    lam = 0.0

    print("Running optimization methods in parallel...")
    results = run_all_methods(A, b, lam=lam)

    # Unpack results
    loss_gd = acc_gd = loss_sgd = acc_sgd = loss_new = acc_new = None

    for name, w, b0, loss_hist, acc_hist in results:
        if name == "gd":
            loss_gd, acc_gd = loss_hist, acc_hist
        elif name == "sgd":
            loss_sgd, acc_sgd = loss_hist, acc_hist
        elif name == "newton":
            loss_new, acc_new = loss_hist, acc_hist

    print("Batch GD final accuracy:   ", acc_gd[-1])
    print("Mini-batch SGD final acc:  ", acc_sgd[-1])
    print("Newton final acc:          ", acc_new[-1])

    print("Creating plots in:", IMG_DIR)
    make_plots(loss_gd, loss_sgd, loss_new,
               acc_gd, acc_sgd, acc_new,
               IMG_DIR)

    print("Done.")
