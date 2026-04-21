import numpy as np
import math
import matplotlib.pyplot as plt

def load_data(filepath):
    import csv
    rows = []
    with open(filepath, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            rows.append(row)

    X = np.array([[int(v) for v in row[1:-1]] for row in rows], dtype=np.float64)
    y = ['spam' if int(row[-1]) != 0 else 'ham' for row in rows]

    feature_names = header[1:-1]
    print(f"  Loaded: {X.shape[0]} emails, {X.shape[1]} features")
    print(f"  Spam: {y.count('spam')}  Ham: {y.count('ham')}")
    return X, y, feature_names

def to_tf(X):
    totals = X.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1
    return X / totals

def train_test_split(X, y, test_size=0.2, seed=42):
    np.random.seed(seed)
    idx = np.random.permutation(len(X))
    split = int(len(X) * (1 - test_size))
    tr, te = idx[:split], idx[split:]
    return X[tr], X[te], [y[i] for i in tr], [y[i] for i in te]

class NaiveBayes:
    """
    Multinomial Naive Bayes built from scratch using NumPy only.
    Steps:
      1. Compute Prior P(c) = count(c) / N
      2. Compute Likelihood P(w|c) using word counts + Laplace smoothing
      3. Posterior = log P(c) + sum(x_i * log P(w_i|c))
      4. Predict = argmax of posterior
    """
    def __init__(self, laplace=True):
        self.laplace = laplace
        self.classes = []
        self.log_prior = {}
        self.log_likelihood = {}

    def fit(self, X, y):
        self.classes = list(set(y))
        n = X.shape[0]
        alpha = 1.0 if self.laplace else 1e-10

        for c in self.classes:
            mask = np.array([label == c for label in y])
            X_c  = X[mask]

            self.log_prior[c] = math.log(mask.sum() / n)

            word_counts = X_c.sum(axis=0) + alpha
            self.log_likelihood[c] = np.log(word_counts / word_counts.sum())

    def predict(self, X):
        preds = []
        for x in X:
            scores = {
                c: self.log_prior[c] + np.dot(x, self.log_likelihood[c])
                for c in self.classes
            }
            preds.append(max(scores, key=scores.get))
        return preds

def confusion_matrix_calc(y_true, y_pred, classes):
    idx = {c: i for i, c in enumerate(classes)}
    cm  = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t]][idx[p]] += 1
    return cm

def compute_metrics(cm, classes):
    accuracy = cm.diagonal().sum() / cm.sum()
    results  = {}
    for i, c in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        results[c] = {'precision': prec, 'recall': rec, 'f1': f1}
    return accuracy, results

def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, fontsize=12)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, fontsize=12)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual',    fontsize=12)
    ax.set_title(title, fontsize=13)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    fontsize=14, fontweight='bold',
                    color='white' if cm[i,j] > cm.max()/2 else 'black')
    plt.colorbar(im)
    plt.tight_layout()
    plt.show()

def run_task1():

    X_bow, y, feature_names = load_data('emails.csv')
    X_tf = to_tf(X_bow)

    classes = ['spam', 'ham']
    summary = {}

    configs = [
        ('BoW', X_bow, True),
        ('BoW', X_bow, False),
        ('TF',  X_tf,  True),
        ('TF',  X_tf,  False),
    ]

    for vec_name, X, laplace in configs:
        tag = f"{vec_name} | Laplace={'Yes' if laplace else 'No'}"
        print(f"\n── Config: {tag} ──")

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)

        model = NaiveBayes(laplace=laplace)
        model.fit(X_tr, y_tr)

        y_pred_tr = model.predict(X_tr)
        y_pred_te = model.predict(X_te)

        cm_tr = confusion_matrix_calc(y_tr, y_pred_tr, classes)
        cm_te = confusion_matrix_calc(y_te, y_pred_te, classes)

        acc_tr, _   = compute_metrics(cm_tr, classes)
        acc_te, met = compute_metrics(cm_te, classes)

        print(f"  Train Accuracy : {acc_tr:.4f}")
        print(f"  Test  Accuracy : {acc_te:.4f}")
        for c in classes:
            m = met[c]
            print(f"  [{c:4s}]  "
                  f"Precision={m['precision']:.4f}  "
                  f"Recall={m['recall']:.4f}  "
                  f"F1={m['f1']:.4f}")

        plot_confusion_matrix(cm_te, classes,
                              title=f'Test Confusion Matrix\n{tag}')
        summary[tag] = (acc_tr, acc_te)

    fig, ax = plt.subplots(figsize=(10, 5))
    configs_list = list(summary.keys())
    train_accs   = [summary[k][0] for k in configs_list]
    test_accs    = [summary[k][1] for k in configs_list]
    x = np.arange(len(configs_list))
    bars1 = ax.bar(x - 0.2, train_accs, 0.35, label='Train Acc', color='steelblue')
    bars2 = ax.bar(x + 0.2, test_accs,  0.35, label='Test Acc',  color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(configs_list, rotation=10, ha='right', fontsize=11)
    ax.set_ylim(0.85, 1.02)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Training vs Testing Accuracy — All Configurations', fontsize=13)
    ax.legend(fontsize=11)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.003,
                f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.003,
                f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.show()

    best = max(summary, key=lambda k: summary[k][1])
    print(f"\nBest Configuration: {best}")
    print(f"   Train Acc = {summary[best][0]:.4f} | Test Acc = {summary[best][1]:.4f}")

run_task1()
