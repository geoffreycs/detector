const interval = Math.round(1000 / 25);

const { ipcRenderer } = require('electron/renderer');
const tf = require('@tensorflow/tfjs-core');
const tflite = require('@tensorflow/tfjs-tflite');
const http = require('http');
const path = require('path');
// const tflite = require('tfjs-tflite-node');
//const { TFLiteModel } = require('@tensorflow/tfjs-tflite/dist/tflite_model');
const { chunkArray, Reformatter, loadLabels } = require('./shared');
const fs = require('fs');
// const labels = loadLabels("drone/drone-detect_labels.txt");
const labels = loadLabels("alexandra/alexandrainst_drone_detect_labels.txt");
const osc = new OffscreenCanvas(300, 300);
const ctx1 = osc.getContext('2d');
let ratio = Infinity;
var vid_params = [0.0, 0.0, 0.0];

/**
 * @param {Error} error 
 */
const onError = function (error) {
    console.error(error);
    ipcRenderer.send('error');
    //throw "Execution halted due to above error"
}

const reformat = Reformatter(300, 300);
const MIME_TYPES = {
    js: "text/javascript",
    wasm: "application/wasm",
    txt: "text/plain"
};
const assets = path.join(process.cwd(), "./node_modules/@tensorflow/tfjs-tflite/wasm");
const toBool = [() => true, () => false];
const port = Math.round(Math.random() * (10000 - 9000) + 9000);
const server = http.createServer(async (req, res) => {
    try {
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
    }
    catch (err) {
        onError(err);
    }
});
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

async function main() {
    try {
        /**
         * @type {HTMLSelectElement}
         */
        const selector = document.getElementById("cameras");
        const findCams = function () {
            const mediaDevices = navigator.mediaDevices.enumerateDevices();
            mediaDevices.then(devices => {
                try {
                    devices.forEach(function (candidate) {
                        if (candidate.kind == "videoinput") {
                            const newOption = new Option();
                            newOption.value = candidate.deviceId;
                            newOption.innerText = candidate.label;
                            selector.appendChild(newOption);
                        }
                    });
                }
                catch (e) {
                    onError(e);
                }
            });
            mediaDevices.catch(onError);
        }
        findCams();

        console.log("Acquiring webcam");
        /**
         * @type {HTMLVideoElement}
         */
        const webcam = document.getElementById('webcam');
        webcam.srcObject = await navigator.mediaDevices.getUserMedia({ video: true });
        // video: 
        webcam.play();

        console.log("Creating output renderer");
        /**
         * @type {HTMLCanvasElement}
         */
        const canvas = document.getElementById('display');
        const ctx2 = canvas.getContext('2d');
        ctx2.font = "15px Arial";
        ctx2.fillText("Waiting for webcam", 20, (canvas.height / 2) - 7);
        const desc = document.getElementById("class");

        console.log("Starting HTTP server to self-serve modules on port " + String(port));
        server.listen(port);

        console.log("Loading model");
        tflite.setWasmPath('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.10/wasm/')
        tflite.setWasmPath('http://127.0.0.1:' + port.toString() + '/');
        // const model = await tflite.loadTFLiteModel(new Uint8Array(fs.readFileSync("drone/drone-detect1.tflite")).buffer);
        const model = await tflite.loadTFLiteModel(new Uint8Array(fs.readFileSync("alexandra/alexandrainst_drone_detect.tflite")).buffer);
        console.log("Closing HTTP server")
        server.close();
        server.removeAllListeners();

        console.log("Waiting for video start");
        /**
         * @param {Function} resolve 
         */
        const checkReady = function (resolve) {
            ratio = Math.min(canvas.width / webcam.videoWidth, canvas.height / webcam.videoHeight);
            if (ratio === Infinity) {
                setTimeout(() => checkReady(resolve), 50);
            } else {
                resolve();
            }
        }
        const CRwrapper = async () => {
            return new Promise(
                /**
                 * @param {Function} resolve 
                 */
                (resolve) => {
                    checkReady(resolve);
                }
            );
        }
        await CRwrapper();
        function paramGen() {
            return [(canvas.height - (webcam.videoHeight * ratio)) / 2, webcam.videoWidth * ratio, webcam.videoHeight * ratio];
        }
        selector.onchange = function () {
            webcam.pause();
            const newCam = navigator.mediaDevices.getUserMedia({ video: { deviceId: selector.value ? { exact: selector.value } : undefined } });
            newCam.then(async stream => {
                webcam.srcObject = stream;
                ctx1.clearRect(0, 0, osc.width, osc.height);
                webcam.play();
                await CRwrapper();
                vid_params = paramGen();
            });
            newCam.catch(onError);
        };

        console.log("Running model");
        vid_params = paramGen();
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

        const doInference = async function () {
            ctx1.drawImage(webcam, 0, 0, webcam.videoWidth, webcam.videoHeight, 0, vid_params[0], vid_params[1], vid_params[2]);
            const img = tf.browser.fromPixels(osc);
            const input = expandTensor(img);

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
            requestAnimationFrame(() => {
                ctx2.clearRect(0, 0, cvs_params[0], cvs_params[1]);
                ctx2.drawImage(osc, 0, 0);
                ctx2.beginPath();
                ctx2.rect(converted.x, converted.y, converted.w, converted.h);
                ctx2.stroke();
            });
            const tag = labels[dataOut[1][0]];
            if (tag == undefined) {
                desc.innerText = "id " + dataOut[1][0].toString() + ", " + String(dataOut[2][0]);
            } else {
                desc.innerText = tag + ", " + String(dataOut[2][0]);
            }
            setTimeout(doInference, interval);
        }

        const runner = doInference();
        runner.catch(onError);
    }
    catch (err) {
        onError(err);
    }
};