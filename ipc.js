const net = require('net');

/**
 * @param {net.Socket} client 
 */
const onError = function (client) {
    client.removeAllListeners();
    client.end();
    client.destroy();
    postMessage("Resetting socket");
    setTimeout(main, 1000);
}

function main() {
    const client = net.createConnection(1337, "127.0.0.1", () => {
        client.on('close', () => {
            postMessage("Socket closed");
            onError(client);
        });
        /**
         * @param {MessageEvent<Number[][]>} msg
         */
        onmessage = msg => {
            client.write(JSON.stringify(msg.data));
        };
        postMessage("connected");
    });
    client.on('error', err => {
        postMessage(err);
        onError(client);
    });
}

main();