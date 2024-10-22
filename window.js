const fs = require('fs');
const tf = require('@tensorflow/tfjs-core');
const tflite = require('@tensorflow/tfjs-tflite');
// const tflite = require('tfjs-tflite-node');
//const { TFLiteModel } = require('@tensorflow/tfjs-tflite/dist/tflite_model');
const { chunkArray, Reformatter, loadLabels, server, port, getGL, onError } = require('./shared');
const labels = loadLabels("drone/drone-detect_labels.txt");
// const labels = loadLabels("alexandra/alexandrainst_drone_detect_labels.txt");
const osc = new OffscreenCanvas(300, 300);
const ctx1 = osc.getContext('2d');
// var vid_params = [0.0, 0.0, 0.0];

const reformat = Reformatter(300, 300);

async function main() {
    try {
        console.log("Creating output renderer");
        /**
         * @type {HTMLCanvasElement}
         */
        const canvas = document.getElementById('display');
        /**
         * @type {CanvasRenderingContext2D}
         */
        const ctx2 = canvas.getContext("2d");
        ctx2.font = "15px Arial";
        ctx2.fillText("Waiting for webcam", 20, (canvas.height / 2) - 7);
        const desc = document.getElementById("class");

        console.log("Acquiring webcam");
        /**
         * @type {HTMLImageElement}
         */
        const webcam = document.getElementById('webcam');
        /**
         * @type {Blob}
         */
        let lastFrame = null;
        var loadFlag = false;
        var lock = false;
        const updateFrame = function () {
            if (!lock) {
                lock = true;
                URL.revokeObjectURL(webcam.src);
                webcam.src = webkitURL.createObjectURL(lastFrame);
            }
        }

        /**
         * @param {WebSocket} ws 
         */
        async function killSocket(ws) {
            console.log("Closing current socket");
            ws.removeEventListener('message', ws.onmessage);
            ws.removeEventListener('error', ws.onerror);
            ws.close();
            loadFlag = false;
            setTimeout(createWebsocket, 200);
        }

        /**
         * @type {HTMLFormElement}
         */
        const ctrl = document.querySelector('#ctrl');
        document.querySelector('#change').onclick = () => {
            document.getElementById("submit").click();
        }

        const cnvGL = document.createElement('canvas');
        cnvGL.hidden = true;
        webcam.onload = function () {
            console.log("Initial frame loaded");
            cnvGL.height = webcam.naturalHeight;
            cnvGL.width = webcam.naturalWidth;
            const drawGL = getGL(cnvGL);
            const newHandler = () => {
                drawGL(webcam);
                lastFrame = null;
                lock = false;
            }
            newHandler();
            webcam.onload = newHandler;
            loadFlag = true;
        };
        /**
         * @type {HTMLInputElement}
         */
        const source = document.querySelector('#source');
        //source.value = "10.1.121.126:8080";
        source.value = "127.0.0.1:8080";
        /**
         * @type {NodeJS.Timeout}
         */
        var timer = null;
        let failCount = 0 | 0;
        function createWebsocket() {
            console.log("Opening WebSocket to " + source.value);
            try {
                const ws = new WebSocket('ws://' + source.value + '/ws');
                /**
                 * @type {NodeJS.Timeout}
                 */
                timer = null;
                ws.onerror = function (e) {
                    console.error(ev);
                    clearTimeout(timer);
                    killSocket(ws);
                }
                /**
                 * @param {MessageEvent} e 
                 */
                ws.onmessage = function (e) {
                    if (timer) {
                        clearTimeout(timer);
                    }
                    if (!lastFrame) {
                        requestAnimationFrame(updateFrame);
                    }
                    lastFrame = e.data;
                    timer = setTimeout(() => {
                        console.log("No data in 1000ms. Resetting socket.");
                        killSocket(ws);
                    }, 1000);
                }
                if (ctrl.onsubmit) {
                    ctrl.removeEventListener("submit", ctrl.onsubmit);
                }
                ctrl.onsubmit = (e) => {
                    e.preventDefault();
                    console.log("Changing server address");
                    clearTimeout(timer);
                    killSocket(ws);
                }
                console.log("WebSocket opened");
                failCount = 0 | 0;
            } catch (e) {
                failCount++;
                if (failCount <= 1000) {
                    console.error(e);
                    setTimeout(createWebsocket, 100);
                } else {
                    onError(e);
                }
            }
        }
        createWebsocket();

        console.log("Starting HTTP server to self-serve modules on port " + port.toString());
        server.listen(port);

        console.log("Loading model");
        tflite.setWasmPath('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.10/wasm/')
        tflite.setWasmPath('http://127.0.0.1:' + port.toString() + '/');
        const model = await tflite.loadTFLiteModel(new Uint8Array(fs.readFileSync("drone/drone-detect1.tflite")).buffer);
        // const model = await tflite.loadTFLiteModel(new Uint8Array(fs.readFileSync("alexandra/alexandrainst_drone_detect.tflite")).buffer);
        console.log("Closing HTTP server")
        server.close();
        server.removeAllListeners();

        console.log("Waiting for video start");
        /**
         * @param {Function} resolve 
         */
        const checkReady = function (resolve) {
            if (!loadFlag) {
                setTimeout(() => checkReady(resolve), 50);
            } else {
                resolve();
            }
        }
        await new Promise(
            /**
             * @param {Function} resolve 
             */
            (resolve) => {
                checkReady(resolve);
            }
        );
        const ratio = Math.min(canvas.width / webcam.width, canvas.height / webcam.height);
        console.log("Running model");
        const vid_params = [(canvas.height - (webcam.height * ratio)) / 2, webcam.width * ratio, webcam.height * ratio];
        const cvs_params = [canvas.width, canvas.height];

        /**
         * @type {HTMLParagraphElement}
         */
        const perf = document.querySelector("#perf");
        const rolling = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        var idx = 0 | 0;
        const doInference = async function () {
            if (!lock) {
                const start = performance.now();
                ctx1.drawImage(cnvGL, 0, 0, webcam.naturalWidth, webcam.naturalHeight, 0, vid_params[0], vid_params[1], vid_params[2]);
                const bitmap = await createImageBitmap(osc)
                const img = tf.browser.fromPixels(bitmap);
                const input = tf.expandDims(img, 0);

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
                const dataOut = [
                    chunkArray(output.TFLite_Detection_PostProcess.dataSync()),
                    await output['TFLite_Detection_PostProcess:1'].data(),
                    await output['TFLite_Detection_PostProcess:2'].data(),
                    await output['TFLite_Detection_PostProcess:3'].data()
                ];

                img.dispose();
                input.dispose();
                output.TFLite_Detection_PostProcess.dispose();
                output['TFLite_Detection_PostProcess:1'].dispose();
                output['TFLite_Detection_PostProcess:2'].dispose();
                output['TFLite_Detection_PostProcess:3'].dispose();

                const converted = reformat(dataOut[0][0]);
                ctx2.clearRect(0, 0, cvs_params[0], cvs_params[1]);
                ctx2.drawImage(bitmap, 0, 0);
                ctx2.beginPath();
                ctx2.rect(converted.x, converted.y, converted.w, converted.h);
                ctx2.stroke();

                const tag = labels[dataOut[1][0]];
                desc.innerText = tag ? (tag + ", " + String(dataOut[2][0].toFixed(7))) :
                    ("id " + dataOut[1][0].toString() + ", " + String(dataOut[2][0]));

                const msec = performance.now() - start;
                rolling[idx] = msec;
                idx = (idx + 1) % 10;
                const total = rolling[0] + rolling[1] + rolling[2] + rolling[3] + rolling[4] + rolling[5] + rolling[6] + rolling[7] + rolling[8] + rolling[9];
                perf.innerText = String(msec.toFixed(2)).padStart(6, '0') + "ms, " + (total / 10).toFixed(2).toString().padStart(6, '0') + "ms"
            }
            setTimeout(doInference, 5);
        }

        const runner = doInference();
        runner.catch(onError);
    }
    catch (err) {
        onError(err);
    }
};