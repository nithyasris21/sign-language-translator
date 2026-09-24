import { useEffect, useMemo, useRef, useState } from 'react'
import { mediaUrl, textToSign } from '../api'

const HOLD_MS = 1200   // how long a still (fingerspelled letter) stays on screen

/** Flatten the backend's nested response into a flat playlist of frames. */
function flatten(sequence) {
  const out = []
  for (const item of sequence) {
    if (item.type === 'fingerspell') {
      item.items.forEach((letter) => out.push({ ...letter, word: item.label }))
    } else {
      out.push(item)
    }
  }
  return out
}

export default function TextToSign() {
  const [text, setText] = useState('hello')
  const [sequence, setSequence] = useState([])
  const [index, setIndex] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [error, setError] = useState(null)
  const videoRef = useRef(null)

  const playlist = useMemo(() => flatten(sequence), [sequence])
  const current = playlist[index]
  const isVideo = current?.src?.endsWith('.mp4') || current?.src?.endsWith('.webm')

  const translate = async (e) => {
    e?.preventDefault()
    setError(null)
    try {
      const data = await textToSign(text)
      setSequence(data.sequence)
      setIndex(0)
      setPlaying(true)
    } catch (err) {
      setError(err.message)
    }
  }

  const advance = () => {
    setIndex((i) => {
      if (i + 1 >= playlist.length) { setPlaying(false); return i }
      return i + 1
    })
  }

  // Stills advance on a timer; videos advance on their own 'ended' event.
  useEffect(() => {
    if (!playing || !current || isVideo) return
    const t = setTimeout(advance, HOLD_MS)
    return () => clearTimeout(t)
  }, [playing, index, current, isVideo])

  // The <video> follows the playlist's play/pause state; otherwise Pause only
  // stopped the still-image timer and the clip played on and advanced.
  useEffect(() => {
    const video = videoRef.current
    if (!isVideo || !video) return
    if (playing) video.play().catch(() => {})
    else video.pause()
  }, [index, isVideo, playing])

  const missing = sequence.filter((s) => s.type === 'missing').map((s) => s.label)

  return (
    <div className="panel">
      <form className="composer" onSubmit={translate}>
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Type a word or sentence…"
        />
        <button type="submit">Translate</button>
      </form>

      {error && <div className="banner error">{error}</div>}
      {missing.length > 0 && (
        <div className="banner warn">
          No sign available for: {missing.join(', ')} — add a clip with{' '}
          <code>python -m ml.record_clip &lt;word&gt;</code>
        </div>
      )}

      <div className="stage">
        {current?.src ? (
          isVideo ? (
            <video
              ref={videoRef}
              key={current.src}
              src={mediaUrl(current.src)}
              className="video"
              muted
              playsInline
              autoPlay
              onEnded={advance}
            />
          ) : (
            <img key={current.src} src={mediaUrl(current.src)} alt={current.label} className="video" />
          )
        ) : (
          <div className="overlay">Nothing to show yet</div>
        )}
        {current && <div className="caption">{current.label}</div>}
      </div>

      <div className="controls">
        <button onClick={() => { setIndex(0); setPlaying(true) }} disabled={!playlist.length}>
          Replay
        </button>
        <button className="ghost" onClick={() => setPlaying((p) => !p)} disabled={!playlist.length}>
          {playing ? 'Pause' : 'Play'}
        </button>
        <span className="status">
          {playlist.length ? `${index + 1} / ${playlist.length}` : ''}
        </span>
      </div>

      <div className="timeline">
        {playlist.map((item, i) => (
          <button
            key={i}
            className={i === index ? 'chip active' : 'chip'}
            onClick={() => { setIndex(i); setPlaying(true) }}
          >
            {item.label}
          </button>
        ))}
      </div>
    </div>
  )
}
