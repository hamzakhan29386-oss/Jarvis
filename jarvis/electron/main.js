const { app, BrowserWindow } = require('electron')
const { spawn } = require('child_process')

let pyProc = null

function createWindow() {
   const win = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1100,
    minHeight: 700,
    autoHideMenuBar: true,
    backgroundColor: '#020b14',
    title: 'JARVIS',
    icon: 'icon.ico'
})

    win.loadURL('http://127.0.0.1:5000')
}

app.whenReady().then(() => {

    const path = require('path')

pyProc = spawn(
    'py',
    ['server.py'],
    {
        cwd: path.join(__dirname, '..'),
        shell: true,
        windowsHide: true
    }
)

    pyProc.stdout.on('data', data => {
        console.log(data.toString())
    })

    pyProc.stderr.on('data', data => {
        console.error(data.toString())
    })

    createWindow()
})

app.on('window-all-closed', () => {
    if (pyProc) pyProc.kill()
    app.quit()
})