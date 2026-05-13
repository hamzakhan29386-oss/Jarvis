const { app, BrowserWindow, ipcMain } = require('electron')
const { spawn } = require('child_process')
const path = require('path')

let pyProc = null
let mainWindow = null

function createWindow() {
   mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1100,
    minHeight: 700,
    autoHideMenuBar: true,
    backgroundColor: '#020b14',
    title: 'JARVIS',
    icon: 'icon.ico'
})

    mainWindow.loadURL('http://127.0.0.1:5000')
}

function launchBackend() {
    pyProc = spawn(
        'py',
        ['-m', 'core.launcher', '--tray'],
        {
            cwd: path.join(__dirname, '..'),
            shell: true,
            windowsHide: true
        }
    )

    pyProc.stdout.on('data', data => {
        const message = data.toString()
        console.log(message)
        if (mainWindow && !mainWindow.isDestroyed()) {
            mainWindow.webContents.send('assistant-state', { stream: 'stdout', message })
        }
    })

    pyProc.stderr.on('data', data => {
        const message = data.toString()
        console.error(message)
        if (mainWindow && !mainWindow.isDestroyed()) {
            mainWindow.webContents.send('assistant-state', { stream: 'stderr', message })
        }
    })

    pyProc.on('exit', code => {
        if (mainWindow && !mainWindow.isDestroyed()) {
            mainWindow.webContents.send('assistant-state', { stream: 'exit', code })
        }
    })
}

ipcMain.handle('assistant-status', async () => {
    try {
        const res = await fetch('http://127.0.0.1:5000/health')
        return await res.json()
    } catch (e) {
        return { error: e.message }
    }
})

app.whenReady().then(() => {
    launchBackend()
    createWindow()
})

app.on('window-all-closed', () => {
    if (pyProc) pyProc.kill()
    app.quit()
})
