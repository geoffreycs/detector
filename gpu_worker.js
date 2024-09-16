require('@tensorflow/tfjs-backend-webgpu');

tf.setBackend('webgpu').then(() => main());