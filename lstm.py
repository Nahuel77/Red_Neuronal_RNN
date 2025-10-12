import numpy as np

# Implementación LSTM para RNN

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

def rand_params():
    return np.random.randn(hidden_size, vocab_size) * 0.01, np.random.randn(hidden_size, hidden_size) * 0.01, np.zeros((hidden_size, 1))

#En una LSTM, hay más matrices de pesos porque hay cuatro “puertas” internas:
#Puerta de olvido (forget gate)
Wf, Uf, bf = rand_params()
#Puerta de entrada (input gate)
Wi, Ui, bi = rand_params()
#Puerta candidata (cell candidate)
Wo, Uo, bo = rand_params()
#Puerta de salida (output gate)
Wc, Uc, bc = rand_params()

#salidas
Wy = np.random.randn(vocab_size, hidden_size) * 0.01
by = np.zeros((vocab_size, 1))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward(inputs, h_prev, c_prev):
    xs, hs, cs, ys, gates = {}, {}, {}, {}, {}
    hs[-1] = np.copy(h_prev)
    cs[-1] = np.copy(c_prev)
    
    for t in range(len(inputs)):
        x = np.zeros((vocab_size, 1)) #42*1
        x[inputs[t]] = 1
        xs[t] = x
        
        #gates
        f_t = sigmoid(np.dot(Wf, x) + np.dot(Uf, hs[t-1]) + bf)
        i_t = sigmoid(np.dot(Wi, x) + np.dot(Ui, hs[t-1]) + bi)
        o_t = sigmoid(np.dot(Wo, x) + np.dot(Uo, hs[t-1]) + bo)
        c_hat_t = np.tanh(np.dot(Wc, x) + np.dot(Uc, hs[t-1]) + bc)
        
        #estados de celda y salidas
        cs[t] = f_t * cs[t-1] + i_t * c_hat_t
        hs[t] = o_t * np.tanh(cs[t])
        
        #Salida final
        ys[t] = np.dot(Wy, hs[t]) + by
        gates[t] = (f_t, i_t, o_t, c_hat_t)
    
    return xs, hs, cs, ys, gates

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

def backward(xs, hs, cs, ys, gates, targets):
    #inicializar gradientes
    dWf = np.zeros_like(Wf); dUf = np.zeros_like(Uf); dbf = np.zeros_like(bf)
    dWi = np.zeros_like(Wi); dUi = np.zeros_like(Ui); dbi = np.zeros_like(bi)
    dWo = np.zeros_like(Wo); dUo = np.zeros_like(Uo); dbo = np.zeros_like(bo)
    dWc = np.zeros_like(Wc); dUc = np.zeros_like(Uc); dbc = np.zeros_like(bc)
    dWy = np.zeros_like(Wy); dby = np.zeros_like(by)

    dh_next = np.zeros((hidden_size, 1))
    dc_next = np.zeros((hidden_size, 1))

    for t in reversed(range(len(targets))):
        dy = softmax(ys[t])
        dy[targets[t]] -= 1
        dWy += np.dot(dy, hs[t].T)
        dby += dy
        
        dh = np.dot(Wy.T, dy) + dh_next #64x1
        f_t, i_t, o_t, c_hat_t = gates[t]
        
        do = dh * np.tanh(cs[t])
        do_raw = do * o_t * (1 - o_t)
        
        dc = dh * o_t * (1 - np.tanh(cs[t])**2) + dc_next
        df = dc * cs[t-1]
        df_raw = df * f_t * (1 - f_t)
        
        di = dc * c_hat_t
        di_raw = di * i_t * (1 - i_t)
        
        dc_hat = dc * i_t
        dc_hat_raw = dc_hat * (1 - c_hat_t**2)
        
        dWf += np.dot(df_raw, xs[t].T)
        dWi += np.dot(di_raw, xs[t].T)
        dWo += np.dot(do_raw, xs[t].T)
        dWc += np.dot(dc_hat_raw, xs[t].T)
        
        dUf += np.dot(df_raw, hs[t-1].T)
        dUi += np.dot(di_raw, hs[t-1].T)
        dUo += np.dot(do_raw, hs[t-1].T)
        dUc += np.dot(dc_hat_raw, hs[t-1].T)
        
        dbf += df_raw; dbi += di_raw; dbo += do_raw; dbc += dc_hat_raw
        
        dh_next = (np.dot(Uf.T, df_raw) +
                   np.dot(Ui.T, di_raw) +
                   np.dot(Uo.T, do_raw) +
                   np.dot(Uc.T, dc_hat_raw))
        dc_next = dc * f_t
        
    for dparam in [dWf, dWi, dWo, dWc, dUf, dUi, dUo, dUc, dbf, dbi, dbo, dbc, dWy, dby]:
        np.clip(dparam, -5, 5, out=dparam)
        
    return (dWf, dWi, dWo, dWc, dUf, dUi, dUo, dUc, dbf, dbi, dbo, dbc, dWy, dby)

def train(data, n_epochs=500, seq_length=10):
    h_prev = np.zeros((hidden_size, 1))
    C_prev = np.zeros((hidden_size, 1))
    pointer = 0

    for epoch in range(n_epochs):
        #obtener inputs y targets
        if pointer + seq_length + 1 >= len(data):
            pointer = 0
            h_prev = np.zeros((hidden_size, 1))
            C_prev = np.zeros((hidden_size, 1))

        inputs = data[pointer:pointer+seq_length]
        targets = data[pointer+1:pointer+seq_length+1]

        # xs: one-hot
        # hs: o_t * tanh(cs[t])
        # cs: f_t * cs[t-1] + i_t * c_hat_t
        # ys: np.dot(Wy, hs[t]) + by
        # gates: capas (f_t, i_t, o_t, c_hat_t)
        xs, hs, cs, ys, gates = forward(inputs, h_prev, C_prev)#inputs es el estado pasado
        loss = compute_loss(ys, targets)#como barajando naipes
        grads = backward(xs, hs, cs, ys, gates, targets)

        (dWf, dWi, dWo, dWc, dUf, dUi, dUo, dUc, dbf, dbi, dbo, dbc, dWy, dby) = grads

        #actualizar pesos
        global Wf, Wi, Wo, Wc, Uf, Ui, Uo, Uc, bf, bi, bo, bc, Wy, by
        for param, dparam in zip(
            [Wf, Wi, Wo, Wc, Uf, Ui, Uo, Uc, bf, bi, bo, bc, Wy, by],
            grads
        ):
            param -= learning_rate*dparam

        h_prev = hs[len(inputs)-1]
        C_prev = cs[len(inputs)-1]

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, loss: {loss:.4f}")

        pointer += seq_length

train(data)