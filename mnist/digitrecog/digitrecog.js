const nIMAGE=60000;
const nROWS=28;
const nCOLS=28;
const LEARNING_RATE = 0.15;
const optimizer = tf.train.sgd(LEARNING_RATE);
const BATCH_SIZE = 64;
const TRAIN_BATCHES = 150;
const ONE_HOT_ARR = initializeOnehotArray();
const IMAGESET_SIZE=1000;

const model = tf.sequential();
model.add(tf.layers.conv2d({
  inputShape: [28, 28, 1],
  kernelSize: 5,
  filters: 8,
  strides: 1,
  activation: 'relu',
  kernelInitializer: 'varianceScaling',
  name: 'firstlayer'
}));
model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2], name: 'secondlayer'}));
model.add(tf.layers.conv2d({
  kernelSize: 5,
  filters: 16,
  strides: 1,
  activation: 'relu',
  kernelInitializer: 'varianceScaling',
  name: 'thirdLayer'
}));
model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2], name: 'fourthlayer'}));
model.add(tf.layers.flatten({name: 'fifthlayer'}));
model.add(tf.layers.dense({units: 10, kernelInitializer: 'varianceScaling', activation: 'softmax', name: 'sixthlayer'}));
model.compile({
  optimizer: optimizer,
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy'],
});

(function() {
  train();  
})();

async function train() {

  const labels = await fetch('mnist/mnist/labels.json').then(r => r.json());
  //console.log(labels.length+" labels loaded");
  for (let i=0;i<labels.length;i++) {
    labels[i] = oneHot(labels[i]);
  }

  for (let i=0;i<labels.length/IMAGESET_SIZE;i++) {
    const lableSubset = labels.slice(i*IMAGESET_SIZE, (i+1)*IMAGESET_SIZE);
    const images = await fetch('mnist/mnist/imageset-'+i+'.json').then(r => r.json());
    const history = await trainSet(lableSubset, images);
    console.log("Loss: "+history.loss[0]+"\tAccuracy: "+history.acc[0]);
  }

  const images = await fetch('mnist/mnist/imageset-'+0+'.json').then(r => r.json());
  showPredictions(images[0]);
}

async function trainSet(labels, images) {
  const xs = tf.tensor2d(images).reshape([-1, 28, 28, 1]);
  const ys = tf.tensor2d(labels);
  const history = await model.fit(xs, ys,{batchSize: BATCH_SIZE, epochs: 5});
  return history.history;
}

async function showPredictions(image) {
  const xs = tf.tensor1d(image).reshape([-1, 28, 28, 1]);
  tf.tidy(() => {
    const output = model.predict(xs.reshape([-1, 28, 28, 1]));
    console.log(output);
  });
}

function to2D(arr, size) {
  var newArr = [];
  while(arr.length) 
    newArr.push(arr.splice(0,size));
  return newArr;
}

function oneHot(num) {
  return ONE_HOT_ARR[num];
}

function initializeOnehotArray() {
  const ONE_HOT_ARR = [];
  for (var i=0;i<10;i++) {
    const arr = [];
    for (var j=0;j<10;j++) {
      arr.push(i==j?1:0)
    }
    ONE_HOT_ARR.push(arr);
  }

  return ONE_HOT_ARR;
}
