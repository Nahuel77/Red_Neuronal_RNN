import numpy as np

# Implementación GRU para RNN

# Dataset
text = "Hola mundo, aprendiendo redes recurrentes!"
chars = sorted(list(set(text)))
char2idx = {ch: i for i, ch in enumerate(chars)}
idx2char = {i: ch for i, ch in enumerate(chars)}
data = [char2idx[c] for c in text]

vocab_size = len(chars)
hidden_size = 64
learning_rate = 1e-2
seq_length = 10

# Pesos GRU
Wz = np.random.randn(hidden_size, vocab_size) * 0.01
Wr = np.random.randn(hidden_size, vocab_size) * 0.01
Wh = np.random.randn(hidden_size, vocab_size) * 0.01

Uz = np.random.randn(hidden_size, hidden_size) * 0.01
Ur = np.random.randn(hidden_size, hidden_size) * 0.01
Uh = np.random.randn(hidden_size, hidden_size) * 0.01

bz = np.zeros((hidden_size, 1))
br = np.zeros((hidden_size, 1))
bh = np.zeros((hidden_size, 1))

Why = np.random.randn(vocab_size, hidden_size) * 0.01
by = np.zeros((vocab_size, 1))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward(inputs, h_prev):
    xs, hs, ys, zs, rs = {}, {}, {}, {}, {}
    hs[-1] = np.copy(h_prev)

    for t in range(len(inputs)):
        x = np.zeros((vocab_size, 1))
        x[inputs[t]] = 1
        xs[t] = x

        z = sigmoid(np.dot(Wz, x) + np.dot(Uz, hs[t-1]) + bz)
        r = sigmoid(np.dot(Wr, x) + np.dot(Ur, hs[t-1]) + br)
        h_tilde = np.tanh(np.dot(Wh, x) + np.dot(Uh, r * hs[t-1]) + bh)
        h = (1 - z) * hs[t-1] + z * h_tilde

        zs[t], rs[t], hs[t] = z, r, h
        ys[t] = np.dot(Why, h) + by

    return xs, hs, ys, zs, rs

def softmax(v):
    expv = np.exp(v - np.max(v))
    return expv / np.sum(expv)

def compute_loss(ys, targets):
    loss = 0
    for t in range(len(targets)):
        ps = softmax(ys[t])
        loss += -np.log(ps[targets[t], 0])
    return loss / len(targets)

# Backprop (simplificada)
def backward(xs, hs, ys, zs, rs, targets):
    dWz, dWr, dWh = np.zeros_like(Wz), np.zeros_like(Wr), np.zeros_like(Wh)
    dUz, dUr, dUh = np.zeros_like(Uz), np.zeros_like(Ur), np.zeros_like(Uh)
    dbz, dbr, dbh = np.zeros_like(bz), np.zeros_like(br), np.zeros_like(bh)
    dWhy, dby = np.zeros_like(Why), np.zeros_like(by)

    dh_next = np.zeros_like(hs[0])

    for t in reversed(range(len(targets))):
        dy = softmax(ys[t])
        dy[targets[t]] -= 1
        dWhy += np.dot(dy, hs[t].T)
        dby += dy

        dh = np.dot(Why.T, dy) + dh_next
        z = zs[t]; r = rs[t]
        h_tilde = np.tanh(np.dot(Wh, xs[t]) + np.dot(Uh, r * hs[t-1]) + bh)

        dh_tilde = dh * z * (1 - h_tilde**2)
        dz = dh * (h_tilde - hs[t-1]) * z * (1 - z)
        dr = np.dot(Uh.T, dh_tilde) * hs[t-1] * r * (1 - r)

        dWh += np.dot(dh_tilde, xs[t].T)
        dUh += np.dot(dh_tilde, (r * hs[t-1]).T)
        dWx = None  # omitido (igual que forward)
        dWz += np.dot(dz, xs[t].T)
        dWr += np.dot(dr, xs[t].T)
        dUz += np.dot(dz, hs[t-1].T)
        dUr += np.dot(dr, hs[t-1].T)
        dbh += dh_tilde
        dbz += dz
        dbr += dr

        dh_next = (1 - z) * dh + np.dot(Uz.T, dz) + np.dot(Ur.T, dr) + np.dot(Uh.T, dh_tilde) * r

    for dparam in [dWz, dWr, dWh, dUz, dUr, dUh, dbz, dbr, dbh, dWhy, dby]:
        np.clip(dparam, -5, 5, out=dparam)

    return dWz, dWr, dWh, dUz, dUr, dUh, dbz, dbr, dbh, dWhy, dby

def train(data, n_epochs=500, seq_length=10):
    global Wz, Wr, Wh, Uz, Ur, Uh, bz, br, bh, Why, by
    h_prev = np.zeros((hidden_size, 1))
    pointer = 0

    for epoch in range(n_epochs):
        if pointer + seq_length + 1 >= len(data):
            pointer = 0
            h_prev = np.zeros((hidden_size, 1))

        inputs = data[pointer:pointer+seq_length]
        targets = data[pointer+1:pointer+seq_length+1]

        xs, hs, ys, zs, rs = forward(inputs, h_prev)
        loss = compute_loss(ys, targets)
        grads = backward(xs, hs, ys, zs, rs, targets)

        dWz, dWr, dWh, dUz, dUr, dUh, dbz, dbr, dbh, dWhy, dby = grads

        # Update
        Wz -= learning_rate * dWz
        Wr -= learning_rate * dWr
        Wh -= learning_rate * dWh
        Uz -= learning_rate * dUz
        Ur -= learning_rate * dUr
        Uh -= learning_rate * dUh
        bz -= learning_rate * dbz
        br -= learning_rate * dbr
        bh -= learning_rate * dbh
        Why -= learning_rate * dWhy
        by -= learning_rate * dby

        h_prev = hs[len(inputs)-1]
        pointer += seq_length

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, loss: {loss:.4f}")

train(data)
