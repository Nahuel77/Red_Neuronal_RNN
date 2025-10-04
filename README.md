<h1>RNN: Recurrent Neural Network</h1>

Tengo que decir que hacer un proyecto a la escala de un simple programador, en una simple computadora, que represente el potencial de una RNN, no es nada facil. Quizas por la naturaleza de el algoritmo RNN por si mismo, es imposible.

Lo que hace esta red, me refiero a este proyecto toy, es muy simple para explicar lo que una RNN puede hacer.
Aqui podemos dar una entrada de texto, como:

    "Hola mundo, aprendien..."
    
y la red debera darnos:

    "..do redes recurrentes!"

Siendo la frase con la que se entrenó "Hola mundo, aprendiendo redes recurrentes!"
¿Hace falta una RNN para conseguir esto?. Obvio que no. Por eso esta introducción.
Aqui tenemos un dataset de 42 caracteres, 5 palabras. Pero aplicando esto mismo a cantidades ingentes de datos, podemos obtener por ejemplo, un predictivo de texto.
De hecho asi funcionaron los primeros predictivos.

La idea de este proyecto es que a partir de una entrada como "mundo" la red RECUERDE que es lo que sigue. Y es que recordar, es la palabra clave en la naturaleza de este tipo de red.
Dado un dato, consultara los datos previos para inferir mediante estadisticas, que dato es el esperado a seguir.

En la etapa de definicion y procesamiento de datos tenemos una cadena de texto simple:

    "Hola mundo, aprendiendo redes recurrentes!"

A esa cadena la convertimos a un array de sus caracteres sin repetir y ordenados. Usamos cada caracter como llave para crear un diccionario de numeros y luego invertimos los valores para que la llave sean los numeros.

    chars = sorted(list(set(text)))
    char2idx = {ch: i for i, ch in enumerate(chars)}
    idx2char = {i: ch for i, ch in enumerate(chars)}

Y en data, solo los numeros correspondientes a cada caracter.

    data = [char2idx[c] for c in text]

Luego tenemos definiciones de el tamaño de las capas, iniciacion de pesos con valores randooms y bias:

    vocab_size = len(chars) #42
    seq_lenght = 10
    hidden_size = 64
    learning_rate = 1e-2

    Wxh = np.random.randn(hidden_size, vocab_size) * 0.01 #64*42
    Whh = np.random.randn(hidden_size, hidden_size) * 0.01 #64*64
    Why = np.random.randn(vocab_size, hidden_size) * 0.01 #42*64

    bh = np.zeros((hidden_size, 1))
    by = np.zeros((vocab_size, 1))

Se ve como una red multicapa MLP? bueno, en escencia lo es. La diferencia es que cruzamos los calculos de los pesos con una matriz de estado previo, la cual en el inicio será una matriz de ceros. Pero a cada paso, el estado previo serán los pesos evaluados previamente.

Empecemos por el inicio... train(data) en la linea 111 (del codigo, no el bondi).

Como dije antes, data contiene los numeros correspondiente a cada caracter de la coleccion de caracteres unicos de nuestro dataset.

    [3, 12, 9, 4, 0, 10, 17, 11, 6, 12, 2, 0, 4, 13, 14, 7, 11, 6, 8, 7, 11, 6, 12, 0, 14, 7, 6, 7, 15, 0, 14, 7, 5, 17, 14, 14, 7, 11, 16, 7, 15, 1]

Y como vemos def train(data, n_epochs=1000, seq_length=10) solicitara data de parametro.
h_prev es el estado previo, que comenzara como una matriz repleta de ceros.

    h_prev = np.zeros((hidden_size, 1))
    pointer = 0

h_prev tendra un tamaño 64*1. Tambien se inicializa una variable pointer, que hará la funcion de contador, para asegurarnos de no tomar muestras, mas allá del tamaño del data. Pues se tomara de 10 en 10.
En el ciclo for que cuenta nuestros entrenamientos controlaremos que la variable pointer se resetee a cero si (pointer + seq_length + 1) es mayor al tamaño de la data y vuelve a colocar el estado previo (h_prev) en ceros.

Luego tomo los primeros 10 valores de data y del 1 al 11 en dos set de datos:

    inputs = data[pointer:pointer+seq_length]
    targets = data[pointer+1:pointer+seq_length+1]

por lo tanto:

    inputs =  [3, 12, 9, 4, 0, 10, 17, 11, 6, 12, 2]
    targets = [12, 9, 4, 0, 10, 17, 11, 6, 12, 2, 0]

Tomamos a inputs y a h_prev y lo metemos a la funcion forward, la que nos retornara 3 valores:

    xs, hs, ys = forward(inputs, h_prev)

<h2>Forward</h2>

Declaramos tres diccionarios vacios, xs, hs, ys. Y guardamos h_prev en un estado previo de hs.

    xs, hs, ys = {}, {}, {}
    hs[-1] = np.copy(h_prev)

Declara el indise de hs en [-1] es una facilidad que Python nos da. Y basicamente coloca al final de una fila un valor asignado.
Tenemos un bucle for t in range(len(inputs)) que contará hasta 10. Recordemos que inputs son muestras de data, tomadas de 10 en 10.
Dentro declaramos un array "x" de ceros de 42*1 (el tamaño de la coleccion de caracteres sin repetir del dataset).

    for t in range(len(inputs)): # 0 a 10
        x = np.zeros((vocab_size, 1)) #42*1
        x[inputs[t]] = 1

Y x[inputs[t]] hará un hot-encoding de la muestra de inputs recibida.
Si inputs era por ejemplo: [3, 12, 9, 4, 0, 10, 17, 11, 6, 12, 2], en t=0 inputs[t] será 3 y por lo tanto x=inputs[t]=[0, 0, 0, 1, ... 0, 0, 0] (42 valores). y Ese array ira a guardarse a una matriz llamada xs formando el array de one-hot

    xs[t] = x

El siguiente paso es guardar el nuevo estado en hs[t], calculandolo con la función Tangente Hiperbolica, para aplanar los valores entre -1 y 0. Como se observa en la curva que dibuja la función:

![alt text](miscellaneous/image-2.png)

Se sumaran la multiplicacion de Wxh con x, de Whh con hs[-1] y se le sumaran el bias bh.

    (np.dot(Wxh, x) + np.dot(Whh, hs[t-1]) + bh)

Sabemos por las reglas de multiplicacion de matrices que la salida de estas suma sera otra matriz de 64x1

![alt text](image.png) ![alt text](image-1.png)

matriz(64x1) + matriz(64x1) + matriz(64x1) = matriz(64x1)
A ese resultado es al que se le aplicará la funcion tanh (tangente hiperbolica)

Forward entonces retorna xs hs y ys.

Esto, lo vimos ya en la red MLP, pero aqui se agrega un elemento.
Una de las matrices de pesos (Wxh) se multiplica con el one-hot correspondiente al ciclo. Eso es igual en la red MLP.
Pero aqui tomamos otra matriz de pesos (Whh) se multiplica con el estado anteior h[t-1] (Esta es la parte en que podriamos decir que "recuerda").
Luego se suma al bias, como tambien era en la red MLP. (np.dot(Wxh, x) + np.dot(Whh, hs[t-1]) + bh).
El resultado obtenido es el estado nuevo hs[t].

Luego se multiplica la tercer matriz de pesos Why con este estado nuevo (hs[t]) y se suma el ultimo peso by.

    ys[t] = np.dot(Why, hs[t]) + by

Forward entonces retorna xs hs y ys.

continuara...