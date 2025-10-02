import numpy as np

# Dataset, una cadena simple

text = "Hola mundo, aprendiendo redes recurrentes!"
chars = sorted(list(set(text)))
char2idx = {ch: i for i, ch in enumerate(chars)}
idx2char = {i: ch for i, ch in enumerate(chars)}
data = [char2idx[c] for c in text]

print(len(data))

vocab_size = len(chars)#42
seq_lenght = 10
hidden_size = 64
learning_rate = 1e-2

Wxh = np.random.randn(hidden_size, vocab_size) * 0.01 #64*42
Whh = np.random.randn(hidden_size, hidden_size) * 0.01 #64*64
Why = np.random.randn(vocab_size, hidden_size) * 0.01 #42*64

bh = np.zeros((hidden_size, 1))
by = np.zeros((vocab_size, 1))

def forward(inputs, h_prev):
    # donde inputs es una muestra reducida de "inputs = data[0:seq_length]"
    # y h_prev una matriz de tamaño hidden_size*1 llena de ceros
    xs, hs, ys = {}, {}, {}
    hs[-1] = np.copy(h_prev)
    
    for t in range(len(inputs)):
        x = np.zeros((vocab_size, 1)) #42*1
        x[inputs[t]] = 1
        xs[t] = x
        hs[t] = np.tanh(np.dot(Wxh, x) + np.dot(Whh, hs[t-1]) + bh)
        # matriz(64x1) + matriz(64x1) + matriz(64x1) = matriz(64x1)
        ys[t] = np.dot(Why, hs[t]) + by
        # matriz(42x1 +  42x1)
 
    return xs, hs, ys

def softmax(v):
    expv = np.exp(v - np.max(v))
    return expv / np.sum(expv)

def compute_loss(ys, targets):
    loss = 0
    for t in range(len(targets)):
        ps = softmax(ys[t])
        loss += -np.log(ps[targets[t], 0])
    return loss / len(targets)

#backward propagation

def backward(xs, hs, ys, targets):
    #inicializar gradientes
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dh_next = np.zeros_like(hs[0])

    for t in reversed(range(len(targets))):
        dy = softmax(ys[t])
        dy[targets[t]] -= 1
        dWhy += np.dot(dy, hs[t].T)
        dby += dy
        dh = np.dot(Why.T, dy) + dh_next
        dh_raw = (1- hs[t] ** 2)*dh
        dbh += dh_raw
        dWxh += np.dot(dh_raw, xs[t].T)
        dWhh += np.dot(dh_raw, hs[t].T)
        dh_next = np.dot(Whh.T, dh_raw)
        
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam)
        
    return dWxh, dWhh, dWhy, dbh, dby

def train(data, n_epochs=1000, seq_length=10):
    h_prev = np.zeros((hidden_size, 1))
    pointer = 0

    for epoch in range(n_epochs):
        #obtener inputs y targets
        if pointer + seq_length + 1 >= len(data):
            pointer = 0
            h_prev = np.zeros((hidden_size, 1))

        inputs = data[pointer:pointer+seq_length]
        targets = data[pointer+1:pointer+seq_length+1]

        xs, hs, ys = forward(inputs, h_prev)#inputs es el estado pasado
        loss = compute_loss(ys, targets)#como barajando naipes

        dWxh, dWhh, dWhy, dbh, dby = backward(xs, hs, ys, targets)

        #actualizar pesos
        global Wxh, Whh, Why, bh, by
        Wxh -= learning_rate * dWxh
        Whh -= learning_rate * dWhh
        Why -= learning_rate * dWhy
        bh -= learning_rate * dbh
        by -= learning_rate * dby

        h_prev = hs[len(inputs)-1]

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, loss: {loss:.4f}")

        pointer += seq_length

train(data)