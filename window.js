const fs = require('fs');
const tf = require('@tensorflow/tfjs-core');
const tflite = require('@tensorflow/tfjs-tflite');
const { asmExport, server, port, getGL, onError } = require('./shared');
const { converted, x_accum, y_accum, w_accum, h_accum, m1_accum, m2_accum, avgs, dimsAvg, midAvg,
    reformat, setAll, accConf, avgConf } = asmExport;
const osc = new OffscreenCanvas(300, 300);
const ctx1 = osc.getContext('2d');
const worker = new Worker("ipc.js");
let ipcUp = false;
let pause = false;

/**
 * @param {MessageEvent} msg 
 */
worker.onmessage = msg => {
    if (msg.data == "connected") {
        ipcUp = true;
    } else {
        console.error(msg.data);
    }
}

const rolling = new Float64Array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

async function init() {
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
    ctx2.lineWidth = 2;
    const desc = document.getElementById("class");
    document.querySelector("#pause").onclick = () => { pause = !pause };

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
    function killSocket(ws) {
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
            if (!pause) {
                drawGL(webcam);
            }
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
    var timer;
    let failCount = 0 | 0;
    function createWebsocket() {
        console.log("Opening WebSocket to " + source.value);
        try {
            const ws = new WebSocket('ws://' + source.value + '/ws');
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
            ctrl.onsubmit = e => {
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
    tflite.setWasmPath('http://127.0.0.1:' + new String(port) + '/');
    const model = await tflite.loadTFLiteModel(fs.readFileSync("drone/drone-detect1.tflite"));
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
    /**
     * @type {Number}
     */
    const timings = {
        "cpu": 50,
        "webgl": 50,
        "webgpu": 30,
        "wasm": 50
    }[tf.getBackend()];
    const discardOldThres = timings;
    const trackLostThres = timings / 2;
    const dy = (canvas.height - (webcam.height * ratio)) / 2;
    const dw = webcam.width * ratio;
    const dh = webcam.height * ratio
    const [cvs_w, cvs_h] = [canvas.width, canvas.height];

    /**
     * @type {HTMLParagraphElement}
     */
    const perf = document.querySelector("#perf");

    var idx_t = 0 | 0;
    var idx_d = 0 | 0;
    var last = 4 | 0;
    let lostCount = 0 | 0;
    let trackExpired = true;
    let trackStale = true;
    let lastX = 0.0;
    let lastY = 0.0;
    var status = 0 | 0;
    const regainMax = 20.0; // max inter-frame jump when track expired
    const trackMax = 50.0; // max inter-frame jump in active or stale track
    const maxsize = 20000.0; // max size on screen
    const minsize = 20.0; // min size on screen
    const close = 1000.0 // size on screen before confidence threshold is raised
    const bigConf = .5 // min confidence for up-close detections
    const maxsquat = 2.9; // max W/H
    const closesquat = .7; // min W/H when close
    /**
     * @param {Number} dX 
     * @param {Number} dY 
     * @param {Number} confInst
     * @returns {Boolean}
     */
    const checkHeuristics = function (dX, dY, confInst) {
        return ((dX > trackMax || dY > trackMax || converted[9] > maxsquat) && !trackExpired)
            || converted[8] > maxsize || converted[8] < minsize || (converted[8] > close && (confInst < bigConf || closesquat > converted[9]));
    }

    const doInference = async function () {
        let i = 0;
        const start = performance.now();
        if (!lock) {
            if (!pause) {
                ctx1.drawImage(cnvGL, 0, 0, webcam.naturalWidth, webcam.naturalHeight, 0, dy, dw, dh);
            }
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
             * @type {Float32Array}
             */
            const pointsOut = await output.TFLite_Detection_PostProcess.data();
            /**
             * @type {Float32Array}
             */
            const confOut = output['TFLite_Detection_PostProcess:2'].dataSync();
            const numDetect = output['TFLite_Detection_PostProcess:3'].bufferSync().values[0];

            img.dispose();
            input.dispose();
            output.TFLite_Detection_PostProcess.dispose();
            output['TFLite_Detection_PostProcess:1'].dispose();
            output['TFLite_Detection_PostProcess:2'].dispose();
            output['TFLite_Detection_PostProcess:3'].dispose();

            trackExpired = (lostCount > discardOldThres);

            reformat(new Float32Array(pointsOut.buffer, pointsOut.byteOffset, 4), lastX, lastY);
            let dX = converted[6];
            let dY = converted[7];
            i = 0;
            while (checkHeuristics(dX, dY, confOut[i]) && (i + 1) < numDetect) {
                reformat(new Float32Array(pointsOut.buffer, pointsOut.byteOffset + (i + 1) * 16, 4), lastX, lastY);
                dX = converted[6];
                dY = converted[7];
                i++;
            }
            if (checkHeuristics(dX, dY, confOut[i])) {
                i = 0;
                confOut[i] = 0.0;
            }
            lastX = converted[4];
            lastY = converted[5];
            if (confOut[i] > .4) {
                if (trackExpired && (dX > regainMax || dY > regainMax)) {
                    setAll();
                } else {
                    x_accum[idx_d] = converted[0];
                    y_accum[idx_d] = converted[1];
                    w_accum[idx_d] = converted[2];
                    h_accum[idx_d] = converted[3];
                    m1_accum[idx_d] = lastX;
                    m2_accum[idx_d] = lastY;
                }
                lostCount = 0;
                accConf(confOut[i]);
            } else {
                x_accum[idx_d] = x_accum[last];
                y_accum[idx_d] = y_accum[last];
                w_accum[idx_d] = w_accum[last];
                h_accum[idx_d] = h_accum[last];
                m1_accum[idx_d] = m1_accum[last];
                m2_accum[idx_d] = m2_accum[last];
                lostCount++;
            }
            last = idx_d;
            idx_d = (idx_d + 1) % 5;

            trackStale = (lostCount > trackLostThres);

            if (lostCount === 0) {
                ctx2.strokeStyle = 'green';
                status = 0;
            } else if (!trackStale) {
                ctx2.strokeStyle = 'blue';
                status = 1;
            } else if (!trackExpired) {
                accConf(confOut[i]);
                ctx2.strokeStyle = 'purple';
                status = 2;
            } else {
                ctx2.strokeStyle = 'red';
                status = 3;
            }

            dimsAvg();
            ctx2.clearRect(0, 0, cvs_w, cvs_h);
            ctx2.drawImage(bitmap, 0, 0);
            ctx2.beginPath();
            ctx2.rect(avgs[0], avgs[1], avgs[2], avgs[3]);
            ctx2.stroke();

            const smoothConf = avgConf();

            desc.innerText = confOut[i].toFixed(7) + ", " + smoothConf.toFixed(7) + ", "
                + String(lostCount).padStart(3, '0');

            if (ipcUp) {
                midAvg();
                worker.postMessage([avgs[4], avgs[5], avgs[2], avgs[3], confOut[i], smoothConf, status]);
            }
        }
        const msec = performance.now() - start;
        rolling[idx_t] = msec;
        idx_t = (idx_t + 1) % 10;
        const total = rolling[0] + rolling[1] + rolling[2] + rolling[3] + rolling[4] +
            rolling[5] + rolling[6] + rolling[7] + rolling[8] + rolling[9];
        perf.innerText = msec.toFixed(2).padStart(6, '0') + "ms, " +
            (total / 10).toFixed(2).padStart(6, '0') + "ms, rej " + i.toString();
        setTimeout(() => doInference().catch(onError), 5);
    }

    const runner = doInference();
    runner.catch(onError);
};

const main = () => { init().catch(onError); }