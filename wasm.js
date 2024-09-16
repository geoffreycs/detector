const { app, BrowserWindow } = require('electron');

app.whenReady().then(() => {

    const win = new BrowserWindow({
        width: 700,
        height: 400,
        webPreferences: {
            sandbox: false,
            nodeIntegration: true,
            contextIsolation: false
        }
    })

    win.loadFile('wasm_worker.html');
    //win.webContents.openDevTools();

});

app.on('window-all-closed', () => app.quit());
app.on('gpu-info-update', () => console.log(app.getGPUFeatureStatus()));