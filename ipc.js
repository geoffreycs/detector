const net = require('net');
let lock = true;
let up = false;

/**
 * @type {net.Socket}
 */
let client = null;

/**
 * @param {net.Socket} client 
 */
const onError = function (client) {
    client.removeAllListeners();
    // client.end();
    client.destroy();
    setTimeout(main, 500);
}

/**
 * @param {MessageEvent<Number[][]>} msg
 */
onmessage = msg => {
    if (up) {
        client.write(JSON.stringify(msg.data));
    }
};

function main() {
    lock = false;

    client = net.createConnection(1337, "127.0.0.1", () => {
        up = true;
        postMessage("connected");
    });

    client.on('close', () => {
        if (!lock) {
            lock = true;
            postMessage("Socket closed \"gracefully\"");
            onError(client);
        }
    });

    client.on('error', err => {
        if (!lock) {
            lock = true;
            postMessage(err);
            onError(client);
        }
    });
}

main();