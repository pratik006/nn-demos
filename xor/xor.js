_X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
];
_Y = [0, 1, 1, 0];

tf.loadModel("http://localhost:8000/models/js/xor.json").then((model) => {
    model.predict(tf.tensor2d([[1, 1], [0, 0], [0, 1], [1, 0]])).print();
}).catch((resp) => {
    console.log("failed");   
    const model = tf.sequential();
    model.add(tf.layers.dense({units: 16, inputShape: [2], activation: 'relu'}));
    model.add(tf.layers.dense({units: 16, inputShape: [4], activation: 'softmax'}));
    model.add(tf.layers.dense({units: 1, inputShape: [4]}));

    model.compile({loss: 'meanSquaredError', optimizer: tf.train.adam(0.02)});

    const xs = tf.tensor2d(_X);
    const ys = tf.tensor1d(_Y);
    model.fit(xs, ys, {epochs: 100, shuffle: true}).then(() => {
        model.save("downloads://xor");
        model.predict(tf.tensor2d([[1, 1], [0, 0], [0, 1], [1, 0]])).print();
    });
});