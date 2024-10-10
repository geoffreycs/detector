const fs = require('fs');
const tf = require('@tensorflow/tfjs-core');
const tflite = require('@tensorflow/tfjs-tflite');
// const tflite = require('tfjs-tflite-node');
//const { TFLiteModel } = require('@tensorflow/tfjs-tflite/dist/tflite_model');
const { chunkArray, Reformatter, loadLabels, server, port, onError } = require('./shared');
const labels = loadLabels("drone/drone-detect_labels.txt");
// const labels = loadLabels("alexandra/alexandrainst_drone_detect_labels.txt");
const osc = new OffscreenCanvas(300, 300);
const ctx1 = osc.getContext('2d');
// var vid_params = [0.0, 0.0, 0.0];
const urlCreator = window.URL || window.webkitURL;

const reformat = Reformatter(300, 300);

async function main() {
    try {
        console.log("Creating output renderer");
        /**
         * @type {HTMLCanvasElement}
         */
        const canvas = document.getElementById('display');
        const ctx2 = canvas.getContext('2d');
        ctx2.font = "15px Arial";
        ctx2.fillText("Waiting for webcam", 20, (canvas.height / 2) - 7);
        const desc = document.getElementById("class");

        console.log("Acquiring webcam");
        /**
         * @type {HTMLImageElement}
         */
        const webcam = document.getElementById('webcam');
        let lastFrame = null;
        let loadFlag = false;
        const updateFrame = function () {
            webcam.src = urlCreator.createObjectURL(lastFrame);
            lastFrame = null;
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
         * @type {HTMLButtonElement}
         */
        const change = document.querySelector('#change');

        webcam.onload = function () {
            urlCreator.revokeObjectURL(webcam.src);
            loadFlag = true;
        };
        /**
         * @type {HTMLInputElement}
         */
        const source = document.querySelector('#source');
        source.value = "10.1.121.96:8080";
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
                if (change.onclick) {
                    change.removeEventListener("click", change.onclick);
                }
                change.onclick = () => {
                    console.log("Changing server address");
                    clearTimeout(timer);
                    killSocket(ws);
                }
                console.log("WebSocket opened");
                failCount = 0 | 0;
            } catch (e) {
                failCount++;
                if (failCount <= 100) {
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
            source => tf.expandDims(source, 0);

        const doInference = async function () {
            ctx1.drawImage(webcam, 0, 0, webcam.naturalWidth, webcam.naturalHeight, 0, vid_params[0], vid_params[1], vid_params[2]);
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
            setTimeout(doInference, 5);
        }

        const runner = doInference();
        runner.catch(onError);
    }
    catch (err) {
        onError(err);
    }
};