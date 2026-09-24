import { useEffect, useRef, useState } from 'react'
import { WS_URL } from '../api'

const SEND_FPS = 12          // frames/sec pushed to the backend
const JPEG_QUALITY = 0.6

export default function SignToText({ health }) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const wsRef = useRef(null)
  const timerRef = useRef(null)

  const [streaming, setStreaming] = useState(false)
  const [status, setStatus] = useState('idle')
  const [prediction, setPrediction] = useState(null)
  const [sentence, setSentence] = useState([])
  const [buffering, setBuffering] = useState({ filled: 0, needed: 30 })

  const stop = () => {
    clearInterval(timerRef.current)
    timerRef.current = null
    wsRef.current?.close()
    wsRef.current = null
    const tracks = videoRef.current?.srcObject?.getTracks() ?? []
    tracks.forEach((t) => t.stop())
    if (videoRef.current) videoRef.current.srcObject = null
    setStreaming(false)
    setStatus('idle')
  }

  useEffect(() => stop, [])

  const sendFrame = () => {
    const ws = wsRef.current
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!ws || ws.readyState !== WebSocket.OPEN || !video?.videoWidth) return

    // Downscale before encoding: MediaPipe does not need full resolution and
    // this keeps each frame well under a few hundred KB on the wire.
    canvas.width = 480
    canvas.height = (video.videoHeight / video.videoWidth) * 480
    const ctx = canvas.getContext('2d')
    ctx.save()
    ctx.scale(-1, 1)                                  // mirror, matches training
    ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height)
    ctx.restore()
    ws.send(JSON.stringify({ frame: canvas.toDataURL('image/jpeg', JPEG_QUALITY) }))
  }

  const start = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720 },
      })
      videoRef.current.srcObject = stream
      await videoRef.current.play()
    } catch {
      setStatus('camera denied')
      return
    }

    const ws = new WebSocket(WS_URL)
    wsRef.current = ws
    setStatus('connecting')

    ws.onopen = () => {
      setStreaming(true)
      timerRef.current = setInterval(sendFrame, 1000 / SEND_FPS)
    }

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data)
      if (msg.type === 'ready') setStatus('watching')
      if (msg.type === 'error') { setStatus(msg.message); stop() }
      if (msg.type === 'buffering') setBuffering({ filled: msg.filled, needed: msg.needed })
      if (msg.type === 'prediction') {
        setBuffering({ filled: msg.scores ? 30 : 0, needed: 30 })
        setPrediction(msg)
        if (msg.accepted) setSentence((s) => [...s, msg.label])
      }
    }

    ws.onerror = () => setStatus('connection failed')
    ws.onclose = () => setStreaming(false)
  }

  const ranked = prediction
    ? Object.entries(prediction.scores).sort((a, b) => b[1] - a[1]).slice(0, 4)
    : []

  return (
    <div className="panel">
      <div className="stage">
        <video ref={videoRef} playsInline muted className="video" />
        <canvas ref={canvasRef} hidden />
        {!streaming && <div className="overlay">Camera off</div>}
        {prediction?.hands === false && streaming && (
          <div className="hint">No hands detected — move into frame</div>
        )}
      </div>

      <div className="controls">
        {streaming ? (
          <button className="danger" onClick={stop}>Stop</button>
        ) : (
          <button onClick={start} disabled={!health?.model_trained}>Start camera</button>
        )}
        <button className="ghost" onClick={() => setSentence([])}>Clear</button>
        <span className="status">{status}</span>
      </div>

      {streaming && buffering.filled < buffering.needed && (
        <div className="progress">
          <div style={{ width: `${(buffering.filled / buffering.needed) * 100}%` }} />
          <span>collecting frames {buffering.filled}/{buffering.needed}</span>
        </div>
      )}

      {prediction && (
        <div className="scores">
          {ranked.map(([label, score]) => (
            <div key={label} className="score-row">
              <span className="label">{label}</span>
              <div className="bar"><div style={{ width: `${score * 100}%` }} /></div>
              <span className="pct">{(score * 100).toFixed(0)}%</span>
            </div>
          ))}
        </div>
      )}

      <div className="sentence">
        <h3>Translation</h3>
        <p>{sentence.length ? sentence.join(' ') : <em>Sign something…</em>}</p>
      </div>
    </div>
  )
}
