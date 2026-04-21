import numpy as np
import matplotlib.pyplot as plt
from itertools import product as iproduct

def load_mnist():
    try:
        import gzip, urllib.request, os

        base  = 'http://yann.lecun.com/exdb/mnist/'
        files = {
            'train_images': 'train-images-idx3-ubyte.gz',
            'train_labels': 'train-labels-idx1-ubyte.gz',
            'test_images' : 't10k-images-idx3-ubyte.gz',
            'test_labels' : 't10k-labels-idx1-ubyte.gz',
        }

        def download(name):
            path = files[name]
            if not os.path.exists(path):
                print(f"  Downloading {path}...")
                urllib.request.urlretrieve(base + path, path)
            with gzip.open(path, 'rb') as f:
                data = f.read()
            if 'images' in name:
                return np.frombuffer(data, np.uint8,
                                     offset=16).reshape(-1, 784) / 255.0
            return np.frombuffer(data, np.uint8, offset=8)

        X_tr = download('train_images'); y_tr = download('train_labels')
        X_te = download('test_images');  y_te = download('test_labels')
        print(f"  MNIST loaded: train={X_tr.shape}, test={X_te.shape}")
        return X_tr, y_tr, X_te, y_te

    except Exception as e:
        print(f"  MNIST download failed ({e}). Using random placeholder data.")
        np.random.seed(0)
        X_tr = np.random.rand(2000, 784); y_tr = np.random.randint(0, 10, 2000)
        X_te = np.random.rand(400,  784); y_te = np.random.randint(0, 10, 400)
        return X_tr, y_tr, X_te, y_te

def one_hot(y, n_classes=10):
    oh = np.zeros((len(y), n_classes))
    oh[np.arange(len(y)), y] = 1
    return oh

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
def sigmoid_d(z):
    s = sigmoid(z); return s * (1 - s)

def tanh_f(z):  return np.tanh(z)
def tanh_d(z):  return 1 - np.tanh(z) ** 2

def relu(z):    return np.maximum(0, z)
def relu_d(z):  return (z > 0).astype(float)

def leaky_relu(z,   a=0.01): return np.where(z > 0, z, a * z)
def leaky_relu_d(z, a=0.01): return np.where(z > 0, 1.0, a)

def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

ACTIVATIONS = {
    'sigmoid':    (sigmoid,    sigmoid_d),
    'tanh':       (tanh_f,     tanh_d),
    'relu':       (relu,       relu_d),
    'leaky_relu': (leaky_relu, leaky_relu_d),
}

def init_weights(fan_in, fan_out, method, seed=42):
    np.random.seed(seed)
    if method == 'zero':
        return np.zeros((fan_in, fan_out))
    elif method == 'random_normal':
        return np.random.randn(fan_in, fan_out) * 0.01
    elif method == 'xavier':
        lim = np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-lim, lim, (fan_in, fan_out))
    raise ValueError(f"Unknown init: {method}")

class NeuralNetwork:
    """
    Architecture: Input(784) → Dense(hidden_size, activation) → Dense(10, softmax)
    Uses:  Forward Propagation → Cross-Entropy Loss → Backpropagation → SGD
    """
    def __init__(self, input_size=784, hidden_size=128, output_size=10,
                 activation='relu', init_method='xavier', lr=0.05):
        self.lr       = lr
        self.act_fn, self.act_d = ACTIVATIONS[activation]

        self.W1 = init_weights(input_size,  hidden_size,  init_method)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = init_weights(hidden_size, output_size,  init_method)
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.Z1 = X  @ self.W1 + self.b1
        self.A1 = self.act_fn(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = softmax(self.Z2)
        return self.A2

    def loss(self, y_pred, y_true):
        n    = len(y_true)
        logp = -np.log(np.clip(y_pred[np.arange(n), y_true], 1e-12, 1))
        return logp.mean()

    def backward(self, X, y_true):
        n    = X.shape[0]
        y_oh = one_hot(y_true, self.A2.shape[1])

        dZ2  = (self.A2 - y_oh) / n
        dW2  = self.A1.T @ dZ2
        db2  = dZ2.sum(axis=0, keepdims=True)

        dA1  = dZ2 @ self.W2.T
        dZ1  = dA1 * self.act_d(self.Z1)
        dW1  = X.T  @ dZ1
        db1  = dZ1.sum(axis=0, keepdims=True)

        self.W2 -= self.lr * dW2;  self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1;  self.b1 -= self.lr * db1

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    def accuracy(self, X, y):
        return (self.predict(X) == y).mean()

def train(model, X_tr, y_tr, X_te, y_te, epochs=20, batch_size=256):
    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
    n = X_tr.shape[0]

    for ep in range(1, epochs + 1):
        perm = np.random.permutation(n)
        X_sh, y_sh = X_tr[perm], y_tr[perm]
        ep_loss = 0.0
        batches = 0

        for start in range(0, n, batch_size):
            Xb = X_sh[start:start + batch_size]
            yb = y_sh[start:start + batch_size]
            pred = model.forward(Xb)
            ep_loss += model.loss(pred, yb)
            model.backward(Xb, yb)
            batches += 1

        avg_loss  = ep_loss / batches
        train_acc = model.accuracy(X_tr, y_tr)
        test_acc  = model.accuracy(X_te, y_te)

        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

        if ep % 5 == 0 or ep == 1:
            print(f"  Epoch {ep:3d}/{epochs}  "
                  f"Loss={avg_loss:.4f}  "
                  f"TrainAcc={train_acc:.4f}  "
                  f"TestAcc={test_acc:.4f}")
    return history

def run_task2():

    X_tr, y_tr, X_te, y_te = load_mnist()

    ACTIVATIONS_LIST = ['sigmoid', 'tanh', 'relu', 'leaky_relu']
    INIT_LIST        = ['zero', 'random_normal', 'xavier']
    EPOCHS, BATCH    = 20, 256

    all_hist   = {}
    final_accs = {}

    for act, init in iproduct(ACTIVATIONS_LIST, INIT_LIST):
        key = f"{act} + {init}"
        print(f"\n── {key} ──")
        model = NeuralNetwork(activation=act, init_method=init, lr=0.05)
        hist  = train(model, X_tr, y_tr, X_te, y_te,
                      epochs=EPOCHS, batch_size=BATCH)
        all_hist[key]   = hist
        final_accs[key] = hist['test_acc'][-1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for i, act in enumerate(ACTIVATIONS_LIST):
        ax = axes[i // 2][i % 2]
        for init in INIT_LIST:
            ax.plot(all_hist[f"{act} + {init}"]['train_loss'], label=init)
        ax.set_title(f'Loss vs Epochs — {act}', fontsize=12)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.suptitle('Training Loss: Activation × Initializer', fontsize=14)
    plt.tight_layout(); plt.show()

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for i, act in enumerate(ACTIVATIONS_LIST):
        ax = axes[i // 2][i % 2]
        for init in INIT_LIST:
            ax.plot(all_hist[f"{act} + {init}"]['test_acc'], label=init)
        ax.set_title(f'Test Accuracy vs Epochs — {act}', fontsize=12)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.suptitle('Test Accuracy: Activation × Initializer', fontsize=14)
    plt.tight_layout(); plt.show()

    grid = np.array([[final_accs[f"{act} + {init}"]
                      for init in INIT_LIST]
                     for act in ACTIVATIONS_LIST])

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(grid, cmap='YlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(INIT_LIST)));        ax.set_xticklabels(INIT_LIST, fontsize=11)
    ax.set_yticks(range(len(ACTIVATIONS_LIST))); ax.set_yticklabels(ACTIVATIONS_LIST, fontsize=11)
    ax.set_xlabel('Initializer', fontsize=12); ax.set_ylabel('Activation', fontsize=12)
    ax.set_title('Final Test Accuracy — Activation × Initializer', fontsize=13)
    for r in range(len(ACTIVATIONS_LIST)):
        for c in range(len(INIT_LIST)):
            ax.text(c, r, f"{grid[r,c]:.3f}",
                    ha='center', va='center',
                    color='black', fontsize=11, fontweight='bold')
    plt.colorbar(im); plt.tight_layout(); plt.show()

    print("\nFinal Test Accuracy — Ranked:")
    print(f"  {'Configuration':<30} | Test Acc")
    print(f"  {'-'*30}-+---------")
    for key in sorted(final_accs, key=final_accs.get, reverse=True):
        print(f"  {key:<30} | {final_accs[key]:.4f}")

    best = max(final_accs, key=final_accs.get)
    print(f"\nBest Combination : {best}")
    print(f"   Test Accuracy     : {final_accs[best]:.4f}")

run_task2()
