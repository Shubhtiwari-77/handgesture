/* ============================================================
   NEUROWRITE — MediaPipe Drawing Engine
   ============================================================ */

// Suppress MediaPipe alert popups (especially WebGL errors)
window._nativeAlert = window.alert;
window.alert = msg => { if (String(msg).toLowerCase().includes('webgl')) { console.warn('Suppressed:', msg); return; } window._nativeAlert(msg); };

// Catch all errors and display them on screen for debugging
window.addEventListener('error', function(e) {
  const errDiv = document.createElement('div');
  errDiv.style.cssText = 'position:fixed;top:0;left:0;background:rgba(255,0,0,0.8);color:white;z-index:9999;padding:10px;font-family:monospace;width:100%;pointer-events:none;';
  errDiv.textContent = 'ERROR: ' + e.message + ' at ' + e.filename + ':' + e.lineno;
  document.body.appendChild(errDiv);
});
window.addEventListener('unhandledrejection', function(e) {
  const errDiv = document.createElement('div');
  errDiv.style.cssText = 'position:fixed;top:0;left:0;background:rgba(255,0,0,0.8);color:white;z-index:9999;padding:10px;font-family:monospace;width:100%;pointer-events:none;';
  errDiv.textContent = 'PROMISE REJECTION: ' + (e.reason && e.reason.message ? e.reason.message : String(e.reason));
  document.body.appendChild(errDiv);
});
const _origConsoleError = console.error;
console.error = function(...args) {
  _origConsoleError.apply(console, args);
  const errDiv = document.createElement('div');
  errDiv.style.cssText = 'position:fixed;bottom:0;left:0;background:rgba(200,0,0,0.8);color:white;z-index:9999;padding:10px;font-family:monospace;width:100%;pointer-events:none;';
  errDiv.textContent = 'CONSOLE ERROR: ' + args.map(a => (a && a.message) ? a.message : String(a)).join(' ');
  document.body.appendChild(errDiv);
};

// ── ELEMENTS ──────────────────────────────────────────────────
const video     = document.getElementById('video');
const outCanvas = document.getElementById('outputCanvas');
const inkCanvas = document.getElementById('inkCanvas');
const efxCanvas = document.getElementById('effectCanvas');
const outCtx    = outCanvas.getContext('2d');
const inkCtx    = inkCanvas.getContext('2d');
const efxCtx    = efxCanvas.getContext('2d');

video.style.opacity = '0'; // Hide underlying video, we render it to canvas

// ── CONFIG ────────────────────────────────────────────────────
const cfg = {
  brushSize: 5, opacity: 0.9, smooth: 4,
  glowInk: false, particles: false,
  inkColor: '#00f5ff', rainbow: false, mode: 'write',
  // How many prediction calls per second (lower = less CPU/GPU usage)
  predictFPS: 15
};

// ── CACHED DOM ELEMENTS ─────────────────────────────────────
const DOM = {
  statusText: document.getElementById('statusText'),
  statusDot: document.getElementById('statusDot'),
  handsDetected: document.getElementById('handsDetected'),
  fpsValue: document.getElementById('fpsValue'),
  statFps: document.getElementById('statFps'),
  statPoints: document.getElementById('statPoints'),
  statStrokes: document.getElementById('statStrokes'),
  statUndos: document.getElementById('statUndos'),
  statMode: document.getElementById('statMode'),
  penCursor: document.getElementById('penCursor'),
  gestureToast: document.getElementById('gestureToast'),
  leftGestureName: document.getElementById('leftGestureName'),
  rightGestureName: document.getElementById('rightGestureName'),
  leftGesture: document.getElementById('leftGesture'),
  rightGesture: document.getElementById('rightGesture'),
  noCameraOverlay: document.getElementById('noCameraOverlay'),
  vizBars: document.querySelectorAll('.viz-bar')
};

const COLORS = [
  {hex:'#00f5ff',name:'Cyan'},{hex:'#bf00ff',name:'Violet'},
  {hex:'#ff00a0',name:'Pink'},{hex:'#ff6b00',name:'Orange'},
  {hex:'#ffd700',name:'Gold'},{hex:'#00ff41',name:'Matrix'},
  {hex:'#ffffff',name:'White'},{hex:'#ff4444',name:'Red'},
  {hex:'#44ffaa',name:'Mint'}
];
let colorIdx = 0;

// ── STATE ─────────────────────────────────────────────────────
let isDrawing = false, lastPt = null;
let strokeHistory = [], currentStroke = [];
let totalPoints = 0, totalUndos = 0;
let particles = [];
let latestResults = null;
let renderFrameCount = 0, lastFpsTime = performance.now();
let lastPredictTime = 0; // millis
let lastGestureTime = {clear:0, undo:0, color:0, erase:0};
const COOLDOWN = 800;

// ── RESIZE ────────────────────────────────────────────────────
function resize() {
  const w = outCanvas.offsetWidth, h = outCanvas.offsetHeight;
  const snap = (inkCanvas.width > 0) ? inkCtx.getImageData(0,0,inkCanvas.width,inkCanvas.height) : null;
  outCanvas.width = inkCanvas.width = efxCanvas.width = w;
  outCanvas.height = inkCanvas.height = efxCanvas.height = h;
  if (snap) inkCtx.putImageData(snap, 0, 0);
}
window.addEventListener('resize', resize);

// ── GESTURE DETECTION ─────────────────────────────────────────
function getGesture(lm) {
  // Finger extended: tip is above its PIP joint (lower y = higher on screen)
  const up = (tip, pip) => lm[tip].y < lm[pip].y - 0.02;

  const idx = up(8, 6);
  const mid = up(12, 10);
  const rng = up(16, 14);
  const pky = up(20, 18);
  const count = [idx, mid, rng, pky].filter(Boolean).length;

  // Pinch: thumb tip near index tip
  const dx = lm[4].x - lm[8].x, dy = lm[4].y - lm[8].y;
  if (Math.sqrt(dx*dx + dy*dy) < 0.07) return 'Pinch';

  if (count === 0) return 'Fist';
  if (count >= 3)  return 'Open Hand';

  // Thumb Up: only thumb out (y well above wrist), fingers curled
  const thumbUp = lm[4].y < lm[0].y - 0.1 && count === 0;
  if (thumbUp) return 'Thumb Up';

  // Peace: index + middle up, others down
  if (idx && mid && !rng && !pky) return 'Peace';

  // POINT: ONLY index up — thumb state is ignored so natural pointing works!
  if (idx && !mid && !rng && !pky) return 'Point';

  return 'Custom';
}

// ── TFJS INIT ────────────────────────────────────────────
let detector;
let isPredicting = false;
let modelLoadPromise;

// Start loading the heavy ML models in the background immediately!
async function preloadModel() {
  await tf.ready(); // Ensure WebGL backend is initialized
  const model = handPoseDetection.SupportedModels.MediaPipeHands;
  const detectorConfig = {
    runtime: 'tfjs',
    modelType: 'lite', // 'lite' is significantly faster than 'full'
    maxHands: 2
  };
  detector = await handPoseDetection.createDetector(model, detectorConfig);
  
  // Warm up WebGL shaders by running a dummy tensor through the network
  const dummyInput = document.createElement('canvas');
  dummyInput.width = 256; dummyInput.height = 256;
  try {
    await detector.estimateHands(dummyInput, {flipHorizontal: false});
  } catch (e) {
    // Ignore warmup errors if any
  }
}
// Trigger background load
modelLoadPromise = preloadModel();

async function initMediaPipe() {
  setStatus('Starting Camera & Engine…', 'warning');
  const btn = document.getElementById('startBtn');
  btn.disabled = true;
  btn.textContent = 'Loading Engine...';

  try {
    // Start camera request with optimized low-latency constraints
    const cameraPromise = navigator.mediaDevices.getUserMedia({ 
      video: { 
          facingMode: 'user', 
          width: { ideal: 640 }, 
          height: { ideal: 480 },
          frameRate: { ideal: 30 }
        } 
    });

    // Wait for BOTH the background model load AND the camera stream to finish in parallel
    const [_, stream] = await Promise.all([modelLoadPromise, cameraPromise]);

    video.srcObject = stream;
    await new Promise((resolve) => {
      video.onloadedmetadata = () => {
        video.width = video.videoWidth;
        video.height = video.videoHeight;
        video.play();
        resolve();
      };
    });

    DOM.noCameraOverlay.classList.add('hidden');
    setStatus('✍️ Ready — Point finger to write!', 'active');
    resize();
    requestAnimationFrame(loop);
  } catch (e) {
    console.error(e);
    setStatus('Camera/Model Error', 'error');
    btn.disabled = false;
    btn.textContent = 'Enable Camera';
  }
}

// ── MAIN LOOP ─────────────────────────────────────────────────
async function loop() {
  const W = outCanvas.width, H = outCanvas.height;

  // 1. Draw video directly (mirrored in JS)
  outCtx.clearRect(0, 0, W, H);
  if (video.readyState >= 2) {
    outCtx.save();
    outCtx.translate(W, 0); outCtx.scale(-1, 1);
    outCtx.drawImage(video, 0, 0, W, H);
    outCtx.restore();
  }

  // 2. Process hands asynchronously (throttle to `cfg.predictFPS`)
  const nowPredict = performance.now();
  if (detector && video.readyState >= 2 && !isPredicting && (nowPredict - lastPredictTime) >= (1000 / cfg.predictFPS)) {
    isPredicting = true;
    lastPredictTime = nowPredict;
    detector.estimateHands(video, {flipHorizontal: false}).then(hands => {
      latestResults = hands;
      isPredicting = false;
    }).catch(e => {
      isPredicting = false;
    });
  }

  const handsArr = latestResults || [];
  DOM.handsDetected.textContent = handsArr.length;

  let drawingThisFrame = false;

  if (handsArr.length === 0) {
    DOM.leftGestureName.textContent = '—';
    DOM.rightGestureName.textContent = '—';
    DOM.leftGesture.classList.remove('active');
    DOM.rightGesture.classList.remove('active');
    if (isDrawing && currentStroke.length > 1) updateCounts();
    isDrawing = false; lastPt = null; currentStroke = [];
    DOM.penCursor.classList.remove('visible');
  }

  const vw = video.videoWidth || 1280;
  const vh = video.videoHeight || 720;

  handsArr.forEach((hand, i) => {
    const label = hand.handedness || 'Right';
    
    // Normalize coordinates to [0, 1] for gesture detection
    const normPts = hand.keypoints.map(p => ({ x: p.x / vw, y: p.y / vh }));
    const gesture = getGesture(normPts);
    
    // Update gesture HUD (cached DOM)
    const isLeft = label === 'Left';
    const gestureName = isLeft ? DOM.leftGestureName : DOM.rightGestureName;
    const gestureEl = isLeft ? DOM.leftGesture : DOM.rightGesture;
    if (gestureName.textContent !== gesture) gestureName.textContent = gesture;
    gestureEl.classList.add('active');

    // Convert to mirrored canvas coords (matches the flipped video)
    const pts = normPts.map(p => ({ x: (1 - p.x) * W, y: p.y * H }));

    // Draw glowing skeleton on outCanvas (optimized: skip shadows on non-drawing)
    drawSkeleton(outCtx, pts, gesture);

    // Gesture commands (debounce)
    const now = Date.now();
    if (gesture==='Open Hand' && now-lastGestureTime.clear>COOLDOWN) { lastGestureTime.clear=now; clearAll(true); showToast('🖐 CLEARED'); }
    if (gesture==='Thumb Up'  && now-lastGestureTime.undo>COOLDOWN)  { lastGestureTime.undo=now; undo(); showToast('↩ UNDO'); }
    if (gesture==='Pinch'     && now-lastGestureTime.color>COOLDOWN) { lastGestureTime.color=now; cycleColor(); }
    if (gesture==='Peace'     && now-lastGestureTime.erase>COOLDOWN) { lastGestureTime.erase=now; toggleErase(); }

    // Drawing
    if (gesture === 'Point') {
      drawingThisFrame = true;
      const tip = pts[8]; // Index tip
      moveCursor(tip.x, tip.y);
      if (!isDrawing) { pushSnapshot(); currentStroke = []; }
      
      if (cfg.mode === 'erase') eraseAt(tip);
      else drawInk(tip);
      
      isDrawing = true;
    } else if (gesture !== 'Open Hand' && gesture !== 'Thumb Up') {
      if (isDrawing && currentStroke.length > 1) updateCounts();
      isDrawing = false; lastPt = null; currentStroke = [];
    }
  });

  if (!drawingThisFrame && handsArr.length) {
    if (isDrawing && currentStroke.length > 1) updateCounts();
    isDrawing = false; lastPt = null;
    DOM.penCursor.classList.remove('visible');
  }

  // 3. Particles & FX (skip every other frame)
  if (renderFrameCount % 2 === 0) {
    efxCtx.globalCompositeOperation = 'destination-out';
    efxCtx.fillStyle = 'rgba(0,0,0,0.4)';
    efxCtx.fillRect(0, 0, efxCanvas.width, efxCanvas.height);
    efxCtx.globalCompositeOperation = 'source-over';
    
    let newParticles = [];
    for (let p of particles) {
      p.x+=p.vx; p.y+=p.vy; p.vy+=0.1; p.life-=p.decay;
      if (p.life<=0) continue;
      newParticles.push(p);
      efxCtx.globalAlpha=p.life*0.5;
      efxCtx.fillStyle=p.col;
      efxCtx.beginPath(); efxCtx.arc(p.x,p.y,p.r*p.life,0,Math.PI*2); efxCtx.fill();
    }
    efxCtx.globalAlpha=1;
    particles = newParticles;
  }

  // 4. FPS Counter (batch update)
  renderFrameCount++;
  const now2 = performance.now();
  if (now2 - lastFpsTime >= 1000) {
    const fps = Math.round(renderFrameCount);
    DOM.fpsValue.textContent = fps;
    DOM.statFps.textContent = fps;
    renderFrameCount = 0; lastFpsTime = now2;
  }

  // 5. Viz bars (ultra-reduced: every 3 frames)
  if (renderFrameCount % 3 === 0) {
    const barHeight = isDrawing ? (2+Math.abs(Math.sin(frameCount/20))*18) : 2;
    for (let i = 0; i < DOM.vizBars.length; i += 5) {
      DOM.vizBars[i].style.height = barHeight+'px';
    }
  }

  requestAnimationFrame(loop);
}

// ── SKELETON ──────────────────────────────────────────────────
const CONN = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
              [5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],
              [13,17],[17,18],[18,19],[19,20],[0,17]];

function drawSkeleton(ctx, pts, gesture) {
  const drawing = gesture === 'Point';
  if (drawing) {
    ctx.strokeStyle = cfg.inkColor;
    ctx.lineWidth = 2;
    ctx.globalAlpha = 0.7;
    for (let i = 0; i < CONN.length; i++) {
      const [a,b] = CONN[i];
      ctx.beginPath(); ctx.moveTo(pts[a].x,pts[a].y); ctx.lineTo(pts[b].x,pts[b].y); ctx.stroke();
    }
    ctx.fillStyle = cfg.inkColor;
    ctx.beginPath(); ctx.arc(pts[8].x,pts[8].y,5,0,Math.PI*2); ctx.fill();
  } else {
    ctx.strokeStyle = 'rgba(80,120,255,0.25)';
    ctx.lineWidth = 0.7;
    ctx.globalAlpha = 0.25;
    for (let i = 0; i < CONN.length; i++) {
      const [a,b] = CONN[i];
      ctx.beginPath(); ctx.moveTo(pts[a].x,pts[a].y); ctx.lineTo(pts[b].x,pts[b].y); ctx.stroke();
    }
  }
  ctx.globalAlpha = 1;
}

// ── INK DRAWING ───────────────────────────────────────────────
function drawInk(pt) {
  currentStroke.push(pt);
  totalPoints++;
  
  let p = pt;
  if (currentStroke.length > cfg.smooth) {
    const r = currentStroke.slice(-cfg.smooth);
    let sx = 0, sy = 0;
    for (let i = 0; i < r.length; i++) { sx += r[i].x; sy += r[i].y; }
    p = { x: sx / r.length, y: sy / r.length };
  }
  
  const col = cfg.rainbow ? `hsl(${(totalPoints*2)%360},100%,60%)` : cfg.inkColor;
  inkCtx.lineCap='round'; 
  inkCtx.lineJoin='round';
  inkCtx.globalAlpha=cfg.opacity;
  inkCtx.lineWidth=cfg.brushSize;
  inkCtx.strokeStyle=col;
  
  if (lastPt) {
    inkCtx.beginPath(); 
    inkCtx.moveTo(lastPt.x,lastPt.y); 
    inkCtx.lineTo(p.x,p.y); 
    inkCtx.stroke();
  } else {
    inkCtx.fillStyle=col;
    inkCtx.beginPath(); 
    inkCtx.arc(p.x,p.y,cfg.brushSize/2,0,Math.PI*2); 
    inkCtx.fill();
  }
  
  lastPt = p;
  
  if (cfg.particles && totalPoints % 3 === 0) {
    const a=Math.random()*Math.PI*2, s=0.8+Math.random()*1.5;
    particles.push({x:p.x,y:p.y,vx:Math.cos(a)*s,vy:Math.sin(a)*s-0.6,life:1,decay:0.05,r:0.8+Math.random()*1.5,col});
  }
  if (totalPoints % 15 === 0) DOM.statPoints.textContent = totalPoints;
}

function eraseAt(pt) {
  inkCtx.save();
  inkCtx.globalCompositeOperation='destination-out';
  inkCtx.beginPath(); inkCtx.arc(pt.x,pt.y,cfg.brushSize*5,0,Math.PI*2);
  inkCtx.fillStyle='rgba(0,0,0,1)'; inkCtx.fill();
  inkCtx.restore();
  lastPt = pt;
}

// ── UNDO / CLEAR ─────────────────────────────────────────────
function pushSnapshot() {
  strokeHistory.push(inkCtx.getImageData(0,0,inkCanvas.width,inkCanvas.height));
  if (strokeHistory.length>30) strokeHistory.shift();
}
function undo() {
  if (!strokeHistory.length) { showToast('Nothing to undo'); return; }
  inkCtx.putImageData(strokeHistory.pop(),0,0);
  totalUndos++;
  DOM.statUndos.textContent=totalUndos;
  updateCounts();
}
function clearAll(snap=false) {
  if (snap) pushSnapshot();
  inkCtx.clearRect(0,0,inkCanvas.width,inkCanvas.height);
  strokeHistory=[];
  updateCounts();
}
function updateCounts() {
  const n=strokeHistory.length;
  DOM.statStrokes.textContent=n;
  if (document.getElementById('strokeCount')) document.getElementById('strokeCount').textContent=n;
}

// ── UI HELPERS ────────────────────────────────────────────────
function moveCursor(x, y) {
  const rect=inkCanvas.getBoundingClientRect();
  DOM.penCursor.style.left=(rect.left+x*rect.width/inkCanvas.width)+'px';
  DOM.penCursor.style.top =(rect.top +y*rect.height/inkCanvas.height)+'px';
  DOM.penCursor.classList.add('visible');
}

let toastT=null;
function showToast(msg) {
  DOM.gestureToast.textContent=msg; 
  DOM.gestureToast.classList.add('show');
  clearTimeout(toastT); 
  toastT=setTimeout(()=>DOM.gestureToast.classList.remove('show'),1300);
}

function setStatus(msg,state='') {
  DOM.statusText.textContent=msg;
  DOM.statusDot.className='status-dot'+(state?' '+state:'');
}

function cycleColor() {
  colorIdx=(colorIdx+1)%COLORS.length;
  setInkColor(COLORS[colorIdx].hex,COLORS[colorIdx].name);
  showToast('🎨 '+COLORS[colorIdx].name.toUpperCase());
  document.querySelectorAll('.color-swatch').forEach(s=>s.classList.toggle('active',s.dataset.color===COLORS[colorIdx].hex));
}

function setInkColor(hex,name='') {
  cfg.inkColor=hex; cfg.rainbow=(hex==='rainbow');
  const d=cfg.rainbow?'conic-gradient(red,yellow,lime,cyan,blue,magenta,red)':hex;
  document.getElementById('inkPreviewStroke').style.background=d;
  document.getElementById('inkPreviewStroke').style.boxShadow=cfg.rainbow?'none':`0 0 12px ${hex}`;
  document.getElementById('inkPreviewLabel').textContent=name||hex;
  document.documentElement.style.setProperty('--ink',cfg.rainbow?'#00f5ff':hex);
}

function toggleErase() {
  cfg.mode=cfg.mode==='erase'?'write':'erase';
  const modeText = cfg.mode==='erase'?'🗑️ ERASE':'✍️ WRITE';
  const modeBadge = document.getElementById('modeBadge');
  if (modeBadge) modeBadge.textContent = modeText;
  document.body.classList.toggle('erase-mode',cfg.mode==='erase');
  DOM.statMode.textContent=cfg.mode==='erase'?'Erase':'Write';
  showToast(cfg.mode==='erase'?'🗑️ ERASE MODE':'✍️ WRITE MODE');
  document.querySelectorAll('.mode-tab').forEach(t=>t.classList.toggle('active',t.dataset.mode===cfg.mode));
}

function doUndo()  { undo(); }
function doClear() { clearAll(true); showToast('🗑️ CLEARED'); }
function doSave()  {
  const tmp=document.createElement('canvas');
  tmp.width=inkCanvas.width; tmp.height=inkCanvas.height;
  const t=tmp.getContext('2d');
  t.drawImage(outCanvas,0,0); t.drawImage(inkCanvas,0,0); t.drawImage(efxCanvas,0,0);
  const a=document.createElement('a');
  a.href=tmp.toDataURL('image/png');
  a.download='neurowrite_'+Date.now()+'.png'; a.click();
}

// ── VIZ BARS ──────────────────────────────────────────────────
function buildBars() {
  const c=document.getElementById('vizBars'); c.innerHTML='';
  for(let i=0;i<Math.floor(window.innerWidth/6);i++){
    const b=document.createElement('div'); b.className='viz-bar'; c.appendChild(b);
  }
}
buildBars();
window.addEventListener('resize', buildBars);

// ── EVENTS ────────────────────────────────────────────────────
document.getElementById('startBtn').addEventListener('click', initMediaPipe);

document.querySelectorAll('.mode-tab').forEach(t=>t.addEventListener('click',()=>{
  cfg.mode=t.dataset.mode;
  document.querySelectorAll('.mode-tab').forEach(x=>x.classList.remove('active'));
  t.classList.add('active');
  document.getElementById('modeBadge').textContent=cfg.mode==='write'?'✍️ WRITE MODE':cfg.mode==='draw'?'🎨 DRAW MODE':'🗑️ ERASE MODE';
  document.body.classList.toggle('erase-mode',cfg.mode==='erase');
  document.getElementById('statMode').textContent=cfg.mode.charAt(0).toUpperCase()+cfg.mode.slice(1);
}));

document.querySelectorAll('.brush-size-btn').forEach(b=>b.addEventListener('click',()=>{
  cfg.brushSize=parseInt(b.dataset.size);
  document.querySelectorAll('.brush-size-btn').forEach(x=>x.classList.remove('active'));
  b.classList.add('active');
}));

const opSl=document.getElementById('opacitySlider');
opSl.addEventListener('input',()=>{ cfg.opacity=opSl.value/100; document.getElementById('opacityVal').textContent=opSl.value+'%'; });
const smSl=document.getElementById('smoothSlider');
smSl.addEventListener('input',()=>{ cfg.smooth=parseInt(smSl.value); document.getElementById('smoothVal').textContent=smSl.value; });

document.getElementById('toggleGlowInk').addEventListener('change',e=>cfg.glowInk=e.target.checked);
document.getElementById('toggleParticles').addEventListener('change',e=>cfg.particles=e.target.checked);

document.querySelectorAll('.color-swatch').forEach(sw=>sw.addEventListener('click',()=>{
  document.querySelectorAll('.color-swatch').forEach(s=>s.classList.remove('active'));
  sw.classList.add('active');
  setInkColor(sw.dataset.color,sw.title);
  colorIdx=COLORS.findIndex(c=>c.hex===sw.dataset.color);
}));

document.getElementById('customColor').addEventListener('input',e=>{
  setInkColor(e.target.value,'Custom');
  document.querySelectorAll('.color-swatch').forEach(s=>s.classList.remove('active'));
});

document.getElementById('undoBtn').addEventListener('click',doUndo);
document.getElementById('clearBtn').addEventListener('click',doClear);
document.getElementById('saveBtn').addEventListener('click',doSave);
document.getElementById('undoBtnBottom').addEventListener('click',doUndo);
document.getElementById('clearBtnBottom').addEventListener('click',doClear);
document.getElementById('saveBtnBottom').addEventListener('click',doSave);
document.getElementById('fullscreenBtn').addEventListener('click',()=>{
  if(!document.fullscreenElement) document.documentElement.requestFullscreen?.();
  else document.exitFullscreen?.();
});

document.addEventListener('keydown',e=>{
  if(e.ctrlKey&&e.key==='z') doUndo();
  if(e.key==='Delete') doClear();
  if(e.ctrlKey&&e.key==='s'){e.preventDefault();doSave();}
});

// ── INIT ──────────────────────────────────────────────────────
setInkColor('#00f5ff','Cyan');
setStatus('Click "Enable Camera" to Start');
