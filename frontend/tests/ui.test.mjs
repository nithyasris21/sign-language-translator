// Browser end-to-end test: the real React app in Chrome, with a WLASL clip
// played through Chrome's fake webcam and the real FastAPI backend.
//
//   npm run test:ui
//
// Starts the backend (:8000), the Vite dev server (:5173) and, for the
// production-build check, `vite preview` (:4173). Needs Google Chrome, the
// backend venv, data/raw (ml.fetch_wlasl) and a trained model.
import { spawn, execFileSync } from 'node:child_process'
import { existsSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { chromium } from 'playwright'

const FRONTEND = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const BACKEND = resolve(FRONTEND, '../backend')
const PYTHON = [join(BACKEND, '.venv/Scripts/python.exe'), join(BACKEND, '.venv/bin/python')].find(existsSync)
const VITE = join(FRONTEND, 'node_modules/vite/bin/vite.js')
const APP = 'http://localhost:5173'
const LETTERS = join(BACKEND, 'data/signs/letters')

const failures = []
function check(name, ok, detail = '') {
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? ` — ${detail}` : ''}`)
  if (!ok) failures.push(name)
}

const children = []
function start(cmd, args, cwd, env = {}) {
  const child = spawn(cmd, args, { cwd, env: { ...process.env, ...env }, stdio: 'ignore' })
  children.push(child)
  return child
}
async function waitFor(url, ms = 90000) {
  const end = Date.now() + ms
  while (Date.now() < end) {
    try { if ((await fetch(url)).ok) return } catch {}
    await new Promise((r) => setTimeout(r, 500))
  }
  throw new Error(`${url} did not come up`)
}

async function launch(extraArgs = []) {
  return chromium.launch({ channel: 'chrome', headless: true, args: extraArgs })
}
function watchErrors(page, sink) {
  page.on('console', (m) => m.type() === 'error' && sink.push(m.text()))
  page.on('pageerror', (e) => sink.push(String(e)))
}
const text = (page, sel) => page.locator(sel).first().textContent()

// ------------------------------------------------------------------ tests

async function signToText(camera) {
  console.log('\nsign -> text (fake webcam: held-out "bowling" clip)')
  const browser = await launch([
    '--use-fake-ui-for-media-stream',
    '--use-fake-device-for-media-stream',
    `--use-file-for-fake-video-capture=${camera}`,
  ])
  const page = await browser.newPage()
  const errors = []
  watchErrors(page, errors)
  await page.goto(APP)

  const startBtn = page.getByRole('button', { name: 'Start camera' })
  await page.waitForFunction(() => !document.querySelector('.controls button')?.disabled, null, { timeout: 15000 })
  check('start button enabled once health reports a model', await startBtn.isEnabled())

  await startBtn.click()
  const sawProgress = await page.locator('.progress').waitFor({ timeout: 5000 }).then(() => true, () => false)
  check('progress bar shows while the window fills', sawProgress)
  await page.waitForFunction(() => document.querySelector('.status')?.textContent === 'watching', null, { timeout: 10000 })
  check('status reaches "watching"', true)

  const recognised = await page.waitForFunction(
    () => document.querySelector('.sentence p')?.textContent.includes('bowling'), null, { timeout: 25000 },
  ).then(() => true, () => false)
  check('sign is translated into the sentence', recognised, await text(page, '.sentence p'))
  check('top-4 score bars render', (await page.locator('.score-row').count()) === 4)
  check('camera preview is live', await page.locator('.stage video').evaluate((v) => !!v.srcObject && !v.paused))

  await page.getByRole('button', { name: 'Clear' }).click()
  check('Clear empties the sentence', (await text(page, '.sentence p')) === 'Sign something…')

  await page.getByRole('button', { name: 'Stop' }).click()
  const stopped = await page.evaluate(() => ({
    overlay: document.querySelector('.overlay')?.textContent,
    status: document.querySelector('.status')?.textContent,
    src: document.querySelector('.stage video').srcObject,
  }))
  check('Stop turns the camera off', stopped.overlay === 'Camera off' && stopped.status === 'idle' && stopped.src === null,
    JSON.stringify(stopped))

  const before = await page.locator('.score-row').first().textContent()
  await startBtn.click()
  await page.waitForFunction(() => document.querySelector('.status')?.textContent === 'watching', null, { timeout: 10000 })
  const again = await page.waitForFunction(
    () => document.querySelector('.sentence p')?.textContent.includes('bowling'), null, { timeout: 25000 },
  ).then(() => true, () => false)
  check('camera can be restarted and recognises again', again, `scores before restart: ${before}`)
  await page.getByRole('button', { name: 'Stop' }).click()

  check('no console errors', errors.length === 0, errors.join(' | '))
  await browser.close()
}

async function textToSign() {
  console.log('\ntext -> sign')
  const browser = await launch()
  const page = await browser.newPage()
  const errors = []
  watchErrors(page, errors)
  await page.goto(APP)
  await page.getByRole('button', { name: 'Text → Sign' }).click()

  const input = page.locator('.composer input')
  await input.fill('help me go drink')
  await input.press('Enter')
  await page.locator('.timeline .chip').first().waitFor()
  const chips = await page.locator('.timeline .chip').allTextContents()
  check('Enter submits and builds the playlist', chips.join(',') === 'help,me,go,drink', chips.join(','))
  check('missing word is flagged', (await text(page, '.banner.warn')).includes('me'))
  const playing = await page.waitForFunction(() => {
    const v = document.querySelector('.stage video'); return v && !v.paused && v.currentTime > 0
  }, null, { timeout: 8000 }).then(() => true, () => false)
  check('first clip plays', playing, await page.locator('.stage video').getAttribute('src'))

  // auto-advance through the whole playlist, including the missing word
  const seen = new Set()
  const finished = await (async () => {
    const end = Date.now() + 30000
    while (Date.now() < end) {
      seen.add(await text(page, '.controls .status'))
      if ((await page.locator('.controls button.ghost').textContent()) === 'Play') return true
      await page.waitForTimeout(150)
    }
    return false
  })()
  check('playlist auto-advances to the end and stops', finished && seen.has('4 / 4'), [...seen].join(' → '))

  // Pause must actually pause the clip, not just the playlist
  await page.getByRole('button', { name: 'Replay' }).click()
  await page.waitForFunction(() => document.querySelector('.stage video')?.currentTime > 0.3, null, { timeout: 8000 })
  await page.getByRole('button', { name: 'Pause', exact: true }).click()
  const t1 = await page.locator('.stage video').evaluate((v) => v.currentTime)
  await page.waitForTimeout(4000)
  const paused = await page.evaluate(() => ({
    t: document.querySelector('.stage video')?.currentTime,
    paused: document.querySelector('.stage video')?.paused,
    status: document.querySelector('.controls .status')?.textContent,
  }))
  check('Pause freezes the current clip', paused.paused && paused.status === '1 / 4' && Math.abs(paused.t - t1) < 0.2,
    JSON.stringify({ ...paused, t_at_pause: t1 }))
  await page.getByRole('button', { name: 'Play', exact: true }).click()
  const resumed = await page.waitForFunction(
    (t) => document.querySelector('.stage video')?.currentTime > t + 0.2, t1, { timeout: 5000 },
  ).then(() => true, () => false)
  check('Play resumes it', resumed)

  await page.locator('.timeline .chip', { hasText: 'drink' }).click()
  const jumped = await page.waitForFunction(() =>
    document.querySelector('.controls .status')?.textContent === '4 / 4'
    && document.querySelector('.stage video')?.src.endsWith('/words/drink.mp4'), null, { timeout: 5000 },
  ).then(() => true, () => false)
  check('clicking a chip jumps to that sign', jumped)

  check('no console errors', errors.length === 0, errors.join(' | '))
  await browser.close()
}

async function fingerspelling() {
  console.log('\nfingerspelling (temporary letter images)')
  // two tiny placeholder images; removed again below
  const made = []
  for (const ch of ['m', 'e']) {
    const p = join(LETTERS, `${ch}.png`)
    if (!existsSync(p)) {
      execFileSync(PYTHON, ['-c',
        `import cv2, numpy as np; i=np.full((240,240,3),40,np.uint8); `
        + `cv2.putText(i,'${ch.toUpperCase()}',(70,170),cv2.FONT_HERSHEY_SIMPLEX,5,(255,255,255),10); cv2.imwrite(r'${p}', i)`])
      made.push(p)
    }
  }
  const browser = await launch()
  try {
    const page = await browser.newPage()
    const errors = []
    watchErrors(page, errors)
    await page.goto(APP)
    await page.getByRole('button', { name: 'Text → Sign' }).click()
    await page.locator('.composer input').fill('me')
    await page.getByRole('button', { name: 'Translate' }).click()
    await page.locator('.timeline .chip').first().waitFor()
    const chips = await page.locator('.timeline .chip').allTextContents()
    check('unknown word is fingerspelled', chips.join(',') === 'M,E', chips.join(','))
    check('no missing-word warning', (await page.locator('.banner.warn').count()) === 0)
    const img = page.locator('.stage img')
    const loaded = await img.evaluate((i) => i.complete && i.naturalWidth > 0)
    check('letter image loads', loaded, await img.getAttribute('src'))
    const advanced = await page.waitForFunction(
      () => document.querySelector('.stage .caption')?.textContent === 'E', null, { timeout: 4000 },
    ).then(() => true, () => false)
    check('letters advance on a timer', advanced)
    check('no console errors', errors.length === 0, errors.join(' | '))
  } finally {
    await browser.close()
    made.forEach((p) => rmSync(p))
  }
}

async function errorStates(camera) {
  console.log('\nerror states')
  let browser = await launch()
  let page = await browser.newPage()

  await page.route('**/api/health', (r) => r.abort())
  await page.goto(APP)
  const down = await page.locator('.banner.error').waitFor({ timeout: 5000 }).then(() => true, () => false)
  check('backend down: error banner and Start disabled',
    down && await page.getByRole('button', { name: 'Start camera' }).isDisabled(),
    down ? await text(page, '.banner.error') : 'no banner')

  await page.unroute('**/api/health')
  await page.route('**/api/health', (r) => r.fulfill({
    json: { status: 'ok', model_trained: false, signs: [], sequence_length: 30, confidence_threshold: 0.75 },
  }))
  await page.reload()
  const noModel = await page.locator('.banner.warn').waitFor({ timeout: 5000 }).then(() => true, () => false)
  check('no model: warning banner and Start disabled',
    noModel && await page.getByRole('button', { name: 'Start camera' }).isDisabled())
  await page.unroute('**/api/health')

  await page.route('**/api/text-to-sign', (r) => r.fulfill({ status: 500, body: 'boom' }))
  await page.reload()
  await page.getByRole('button', { name: 'Text → Sign' }).click()
  await page.getByRole('button', { name: 'Translate' }).click()
  const tErr = await page.locator('.panel .banner.error').waitFor({ timeout: 5000 }).then(() => true, () => false)
  check('translate failure shows an error', tErr)
  await browser.close()

  // real permission denial: no fake-UI flag, so Chrome's prompt is dismissed
  browser = await launch(['--use-fake-device-for-media-stream', `--use-file-for-fake-video-capture=${camera}`])
  const ctx = await browser.newContext()
  page = await ctx.newPage()
  await page.goto(APP)
  await page.waitForFunction(() => !document.querySelector('.controls button')?.disabled, null, { timeout: 15000 })
  await page.getByRole('button', { name: 'Start camera' }).click()
  const denied = await page.waitForFunction(
    () => document.querySelector('.status')?.textContent === 'camera denied', null, { timeout: 10000 },
  ).then(() => true, () => false)
  check('camera permission denied is reported', denied, await text(page, '.status'))
  await browser.close()
}

async function productionBuild() {
  console.log('\nproduction build (vite build + vite preview)')
  execFileSync(process.execPath, [VITE, 'build', '--logLevel', 'error'], { cwd: FRONTEND })
  start(process.execPath, [VITE, 'preview', '--port', '4173', '--strictPort'], FRONTEND)
  await waitFor('http://localhost:4173')
  const browser = await launch()
  const page = await browser.newPage()
  const errors = []
  watchErrors(page, errors)
  await page.goto('http://localhost:4173')
  const ok = await page.waitForFunction(() => !document.querySelector('.controls button')?.disabled, null, { timeout: 10000 })
    .then(() => true, () => false)
  check('built app reaches the backend', ok && (await page.locator('.banner.error').count()) === 0, errors.join(' | '))
  await browser.close()
}

// ------------------------------------------------------------------ main

const tmp = mkdtempSync(join(tmpdir(), 'signlang-ui-'))
const camera = join(tmp, 'camera.y4m')
try {
  execFileSync(PYTHON, ['-m', 'tests.fake_camera', 'data/raw/bowling/07391.mp4', camera, '3'], { cwd: BACKEND })
  start(PYTHON, ['-m', 'uvicorn', 'app.main:app', '--port', '8000'], BACKEND, { TF_CPP_MIN_LOG_LEVEL: '2' })
  start(process.execPath, [VITE, '--port', '5173', '--strictPort'], FRONTEND)
  await waitFor('http://localhost:8000/api/health')
  await waitFor(APP)
  console.log('ui test')

  await signToText(camera)
  await textToSign()
  await fingerspelling()
  await errorStates(camera)
  await productionBuild()
} catch (e) {
  failures.push(`crashed: ${e.message}`)
  console.error(e)
} finally {
  children.forEach((c) => c.kill())
  rmSync(tmp, { recursive: true, force: true })
}

console.log(failures.length ? `\n${failures.length} check(s) failed: ${failures.join(', ')}` : '\nall ui checks passed')
process.exit(failures.length ? 1 : 0)
