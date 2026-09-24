export const API_BASE = 'http://localhost:8000'
export const WS_URL = 'ws://localhost:8000/ws/recognize'

export const mediaUrl = (src) => `${API_BASE}${src}`

export async function getHealth() {
  const res = await fetch(`${API_BASE}/api/health`)
  if (!res.ok) throw new Error('Backend unreachable')
  return res.json()
}

export async function textToSign(text) {
  const res = await fetch(`${API_BASE}/api/text-to-sign`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  })
  if (!res.ok) throw new Error('Translation failed')
  return res.json()
}
