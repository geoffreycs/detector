const net = require('net');

/**
 * @param {Error} err 
 * @param {net.Socket} client 
 */
const onError = function (err, client) {
    client.destroy();
    setTimeout(main, 100);
}

function main() {
    const client = net.createConnection(1337, "127.0.0.1", () => {
        client.on('close', () => {
            onError(null, client);
        });
        onmessage = data => {
            const out = JSON.stringify(data.data);
            client.write(out);
        };
        postMessage("connected");
    });
    client.on('error', err => {
        postMessage(err);
        onError(err, client);
    });
}

main();