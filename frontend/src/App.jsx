import { useEffect, useState } from 'react'
import SignToText from './components/SignToText'
import TextToSign from './components/TextToSign'
import { getHealth } from './api'

export default function App() {
  const [mode, setMode] = useState('sign-to-text')
  const [health, setHealth] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    getHealth().then(setHealth).catch((e) => setError(e.message))
  }, [])

  return (
    <div className="app">
      <header>
        <h1>Sign Language Translator</h1>
        <p className="sub">Deep learning sign recognition, both directions.</p>
      </header>

      {error && <div className="banner error">{error} — is the FastAPI server running on :8000?</div>}
      {health && !health.model_trained && (
        <div className="banner warn">
          No trained model yet. Record data with <code>python -m ml.collect &lt;word&gt;</code>,
          then run <code>python -m ml.train</code>.
        </div>
      )}

      <nav className="tabs">
        <button
          className={mode === 'sign-to-text' ? 'active' : ''}
          onClick={() => setMode('sign-to-text')}
        >
          Sign → Text
        </button>
        <button
          className={mode === 'text-to-sign' ? 'active' : ''}
          onClick={() => setMode('text-to-sign')}
        >
          Text → Sign
        </button>
      </nav>

      <main>
        {mode === 'sign-to-text' ? <SignToText health={health} /> : <TextToSign />}
      </main>
    </div>
  )
}
