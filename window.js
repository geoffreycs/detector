const tf = require('@tensorflow/tfjs-core');
const tflite = require('@tensorflow/tfjs-tflite');
const http = require('http');
const path = require('path');
// const tflite = require('tfjs-tflite-node');
//const { TFLiteModel } = require('@tensorflow/tfjs-tflite/dist/tflite_model');
const { chunkArray, Reformatter, loadLabels } = require('./shared');
const fs = require('fs');
/**
 * @type {Float32Array[]}
 */
const labels = loadLabels("alexandra/alexandrainst_drone_detect_labels.txt");
const osc = new OffscreenCanvas(300, 300);
const ctx1 = osc.getContext('2d');
/**
 * @type {Number}
 */
let ratio = 0 | 0;

/**
 * @type {Function}
 */
var doInference = null;

const reformat = Reformatter(300, 300);
const MIME_TYPES = {
    js: "text/javascript",
    wasm: "application/wasm",
    txt: "text/plain"
};
const assets = path.join(process.cwd(), "./node_modules/@tensorflow/tfjs-tflite/wasm");
const toBool = [() => true, () => false];
const port = Math.round(Math.random() * (10000 - 9000) + 9000);

const prepareFile = async (url) => {
    const paths = [assets, url];
    const filePath = path.join(...paths);
    const pathTraversal = !filePath.startsWith(assets);
    const exists = await fs.promises.access(filePath).then(...toBool);
    const found = !pathTraversal && exists;
    const streamPath = found ? filePath : path.join(process.cwd(), "./404.txt");
    const ext = path.extname(streamPath).substring(1).toLowerCase();
    const stream = fs.createReadStream(streamPath);
    return { found, ext, stream };
};

/**
 * @param {Function} resolve 
 * @param {HTMLCanvasElement} canvas
 * @param {HTMLVideoElement} webcam
 */
const checkReady = function (resolve, canvas, webcam) {
    ratio = Math.min(canvas.width / webcam.videoWidth, canvas.height / webcam.videoHeight);
    if (ratio === Infinity) {
        setTimeout(() => checkReady(resolve, canvas, webcam), 50);
    } else {
        resolve();
    }
}

async function main() {

    //tflite.setWasmPath('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.10/wasm/')
    tflite.setWasmPath('http://127.0.0.1:' + port.toString() + '/');

    console.log("Acquiring webcam");
    /**
     * @type {HTMLVideoElement}
     */
    const webcam = document.getElementById('webcam');
    webcam.srcObject = await navigator.mediaDevices.getUserMedia({ video: true });
    webcam.play();

    console.log("Creating output renderer");
    /**
     * @type {HTMLCanvasElement}
     */
    const canvas = document.getElementById('display');
    const ctx2 = canvas.getContext('2d');
    ctx2.font = "15px Arial";
    ctx2.fillText("Waiting for webcam", 20, canvas.height / 2);
    const desc = document.getElementById("class");

    console.log("Starting HTTP server to self-serve modules on port " + String(port));
    const server = http.createServer(async (req, res) => {
        if (req.socket.remoteAddress.includes("127.0.0.1")) {
            const file = await prepareFile(req.url);
            const statusCode = file.found ? 200 : 404;
            const mimeType = MIME_TYPES[file.ext];
            res.writeHead(statusCode, { "Content-Type": mimeType });
            file.stream.pipe(res);
            console.log(`${req.method} ${req.url} ${statusCode}`);
        } else {
            console.log("Ignored request from " + req.socket.remoteAddress);
        }
    });
    server.listen(port);

    console.log("Loading model");
    const model = await tflite.loadTFLiteModel(new Uint8Array(fs.readFileSync("alexandra/alexandrainst_drone_detect.tflite")).buffer);
    console.log("Closing HTTP server")
    server.close();
    server.removeAllListeners();

    console.log("Waiting for video start")
    await new Promise(
        /**
         * @param {Function} resolve 
         * @param {Function} reject 
         */
        (resolve) => {
            checkReady(resolve, canvas, webcam);
        }
    );


    console.log("Running model");
    const vid_params = [(canvas.height - (webcam.videoHeight * ratio)) / 2, webcam.videoWidth * ratio, webcam.videoHeight * ratio];
    const cvs_params = [canvas.width, canvas.height];

    const expandTensor = (tf.getBackend() == 'webgpu') ?
        /**
         * @param {tf.Tensor3D} source 
         * @returns {tf.Tensor}
         */
        source => {
            const inGPU = source.dataToGPU();
            const expanded = tf.expandDims(inGPU.tensorRef, 0);
            inGPU.tensorRef.dispose();
            return expanded;
        } :
        /**
         * @param {tf.Tensor3D} source 
         * @returns {tf.Tensor}
         */
        source => {
            return tf.expandDims(source, 0);
        }

    doInference = async function () {
        ctx1.drawImage(webcam, 0, 0, webcam.videoWidth, webcam.videoHeight, 0, vid_params[0], vid_params[1], vid_params[2]);
        const img = tf.browser.fromPixels(osc);
        const input = expandTensor(img);
        // const input = tf.expandDims(img, 0);

        /**
         * @type {{TFLite_Detection_PostProcess: tf.Tensor,
         * "TFLite_Detection_PostProcess:1": tf.Tensor,
         * "TFLite_Detection_PostProcess:2": tf.Tensor,
         * "TFLite_Detection_PostProcess:3": tf.Tensor }}
         */
        const output = model.predict(input);
        /**
         * @type {Float32Array[]}
         */
        const dataOut = [chunkArray(await output.TFLite_Detection_PostProcess.data()),
        output['TFLite_Detection_PostProcess:1'].dataSync(),
        await output['TFLite_Detection_PostProcess:2'].data(),
        output['TFLite_Detection_PostProcess:3'].dataSync()
        ];

        img.dispose();
        input.dispose();
        output.TFLite_Detection_PostProcess.dispose();
        output['TFLite_Detection_PostProcess:1'].dispose();
        output['TFLite_Detection_PostProcess:2'].dispose();
        output['TFLite_Detection_PostProcess:3'].dispose();

        const converted = reformat(dataOut[0][0]);
        ctx2.clearRect(0, 0, cvs_params[0], cvs_params[1]);
        ctx2.drawImage(osc, 0, 0);
        ctx2.beginPath();
        ctx2.rect(converted.x, converted.y, converted.w, converted.h);
        ctx2.stroke();
        desc.innerText = labels[dataOut[1][0]] + ", " + String(dataOut[2][0]);
        setTimeout(doInference, 1);
    }

    await doInference();
};