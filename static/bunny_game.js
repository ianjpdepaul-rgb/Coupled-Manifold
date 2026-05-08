// ═══════════════════════════════════════════════════════════════
//  BINKY — a bunny-hopping game
// ═══════════════════════════════════════════════════════════════
(function(){
"use strict";

var SPRITES = {
  Idle:    { frames: 12, w: 32, h: 32 },
  Running: { frames: 8,  w: 32, h: 32 },
  Jumping: { frames: 11, w: 32, h: 32 },
  Liking:  { frames: 5,  w: 32, h: 32 },
  HurtRun: { frames: 16, w: 32, h: 32 },
  LieDown: { frames: 6,  w: 32, h: 32 },
};

var SCALE    = 3;
var PX       = 32*SCALE;
var GRAVITY  = 0.55;
var JUMP_VEL = -11;
var MOVE_SPD = 3.5;
var MAX_LIVES = 5;
var HB_INSET = PX * 0.3;
var IFRAMES  = 90;          // ~1.5s invincibility after hit (blinks)

var canvas, ctx;
var state = "title";
var titleAlpha = 1;
var titleTimer = 0;
var score = 0;
var lives = MAX_LIVES;
var bestScore = parseInt(localStorage.getItem("binky_best")||"0",10);
var lastThreshold = 0;
var shakeTimer = 0, shakeX = 0, shakeY = 0;
var particles = [];
var popups = [];
var keys = {};
var groundY, gameW, gameH;
var gameOverTimer = 0;
var iframeTimer = 0;        // invincibility frames remaining

// ── Difficulty curve ──────────────────────────────────────────
// Returns a difficulty object based on current score.
// Starts gentle, ramps like Google dino: speed increases are
// gradual so skilled players can always react.
function difficulty() {
  var s = score;
  // t goes from 0 (start) to 1 (max difficulty around score 60)
  var t = Math.min(1, s / 60);
  return {
    // Wild bunny run speed range
    runMin:    1.0 + t * 1.5,      // 1.0 → 2.5
    runMax:    2.0 + t * 2.5,      // 2.0 → 4.5
    // Binky jump power multiplier
    binkyMin:  0.4 + t * 0.3,      // 0.4 → 0.7
    binkyMax:  0.7 + t * 0.4,      // 0.7 → 1.1
    // Binky lateral speed
    binkyLat:  2.0 + t * 3.0,      // 2.0 → 5.0
    // State durations (shorter = more unpredictable)
    idleDur:   70 - t * 40,        // 70 → 30 frames
    runDur:    80 - t * 30,        // 80 → 50 frames
    // Probability of each AI state
    pIdle:     0.35 - t * 0.15,    // 35% → 20%
    pRun:      0.45,               // constant 45%
    // rest is binky: 20% → 35%
  };
}

// ── Preload ──
var images = {};
function loadImg(type, anim) {
  var key = type + "_" + anim;
  if (images[key]) return images[key];
  var img = new Image();
  img.src = "/static/bunnies/" + type + "/" + anim + ".png";
  images[key] = img;
  return img;
}
["BunnyBrown","WhiteBunny"].forEach(function(t){
  Object.keys(SPRITES).forEach(function(a){ loadImg(t,a); });
});

// ── Background ──
var bgCanvas, bgCtx, bgGenerated = false;

function generateBG(w, h) {
  bgCanvas = document.createElement("canvas");
  bgCanvas.width = w; bgCanvas.height = h;
  bgCtx = bgCanvas.getContext("2d");
  var sky = bgCtx.createLinearGradient(0, 0, 0, h*0.7);
  sky.addColorStop(0, "#4a90d9");
  sky.addColorStop(0.6, "#87ceeb");
  sky.addColorStop(1, "#c8e6f0");
  bgCtx.fillStyle = sky;
  bgCtx.fillRect(0, 0, w, h);
  bgCtx.fillStyle = "rgba(255,255,255,0.6)";
  for (var c = 0; c < 5; c++) {
    var cx = Math.random()*w, cy = 30+Math.random()*h*0.25;
    var cw = 40+Math.random()*80;
    bgCtx.beginPath();
    bgCtx.ellipse(cx, cy, cw, 12+Math.random()*10, 0, 0, Math.PI*2);
    bgCtx.fill();
    bgCtx.beginPath();
    bgCtx.ellipse(cx+cw*0.3, cy-8, cw*0.6, 10+Math.random()*8, 0, 0, Math.PI*2);
    bgCtx.fill();
  }
  var gy = h*0.78;
  bgCtx.fillStyle = "#8B6914";
  bgCtx.fillRect(0, gy+12, w, h-gy);
  bgCtx.fillStyle = "#4a8c3f";
  bgCtx.fillRect(0, gy, w, 14);
  bgCtx.fillStyle = "#5fa84e";
  bgCtx.fillRect(0, gy, w, 6);
  bgCtx.fillStyle = "#3d7a34";
  for (var g = 0; g < w; g += 8+Math.floor(Math.random()*12)) {
    var gh = 4+Math.floor(Math.random()*8);
    bgCtx.fillRect(g, gy-gh, 2, gh);
    if (Math.random()>0.5) bgCtx.fillRect(g+2, gy-gh+2, 2, gh-2);
  }
  var flowerColors = ["#ff6b8a","#ffdb4d","#ff9f43","#a855f7","#fff"];
  for (var f = 0; f < 15; f++) {
    var fx = Math.random()*w, fy = gy-2-Math.random()*6;
    var fc = flowerColors[Math.floor(Math.random()*flowerColors.length)];
    bgCtx.fillStyle = "#3d7a34";
    bgCtx.fillRect(fx, fy, 1, 6);
    bgCtx.fillStyle = fc;
    bgCtx.fillRect(fx-1, fy-1, 3, 3);
    bgCtx.fillStyle = "#ffeb3b";
    bgCtx.fillRect(fx, fy, 1, 1);
  }
  bgCtx.fillStyle = "rgba(74,140,63,0.25)";
  bgCtx.beginPath();
  bgCtx.moveTo(0, gy);
  for (var hx = 0; hx <= w; hx += 40) {
    bgCtx.lineTo(hx, gy - 15 - Math.sin(hx*0.02)*20 - Math.random()*8);
  }
  bgCtx.lineTo(w, gy);
  bgCtx.fill();
  bgGenerated = true;
}

// ── Bunny entity ──
function makeBunny(type, x) {
  return {
    type: type,
    x: x, y: 0,
    vx: 0, vy: 0,
    dir: 1,
    anim: "Idle",
    frame: 0,
    frameTick: 0,
    grounded: true,
    aiTimer: 0,
    aiState: "idle",
    aiDur: 60,
    passedOver: false,
    hurt: false,
    hurtTimer: 0,
    celebrating: false,
    celebTimer: 0,
    layingDown: false,
  };
}

var player, wild;

function resetGame() {
  score = 0;
  lives = MAX_LIVES;
  lastThreshold = 0;
  gameOverTimer = 0;
  iframeTimer = 0;
  player = makeBunny("BunnyBrown", gameW*0.3);
  wild   = makeBunny("WhiteBunny", gameW*0.7);
  wild.dir = -1;
  particles = [];
  popups = [];
  shakeTimer = 0;
}

// ── Drawing ──
function drawSprite(b, blink) {
  var sp = SPRITES[b.anim];
  if (!sp) return;
  var img = images[b.type+"_"+b.anim];
  if (!img || !img.complete) return;
  // Blink during iframes — skip drawing every other 4 frames
  if (blink && Math.floor(iframeTimer / 4) % 2 === 0) return;
  var f = Math.floor(b.frame) % sp.frames;
  var sx = f * sp.w;
  var dy = groundY - PX - b.y;
  ctx.save();
  if (b.dir < 0) {
    ctx.translate(b.x + PX/2, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(img, sx, 0, sp.w, sp.h, -PX/2, dy, PX, PX);
  } else {
    ctx.drawImage(img, sx, 0, sp.w, sp.h, b.x, dy, PX, PX);
  }
  ctx.restore();
  ctx.fillStyle = "rgba(0,0,0,0.15)";
  ctx.beginPath();
  ctx.ellipse(b.x + PX/2, groundY + 4, PX*0.35, 4, 0, 0, Math.PI*2);
  ctx.fill();
}

function advanceFrame(b, speed) {
  b.frameTick++;
  if (b.frameTick >= speed) { b.frameTick = 0; b.frame++; }
}

// ── Particles ──
function spawnParticles(x, y, color, count) {
  for (var i = 0; i < count; i++) {
    particles.push({
      x:x, y:y,
      vx: (Math.random()-0.5)*6, vy: -Math.random()*5 - 2,
      life: 30 + Math.floor(Math.random()*20),
      color: color, size: 2 + Math.random()*3,
    });
  }
}
function spawnPopup(x, y, text, color) {
  popups.push({ x:x, y:y, text:text, color:color||"#fff", life:60, maxLife:60 });
}
function updateParticles() {
  for (var i = particles.length-1; i >= 0; i--) {
    var p = particles[i];
    p.x += p.vx; p.y += p.vy; p.vy += 0.15; p.life--;
    if (p.life <= 0) particles.splice(i, 1);
  }
  for (var j = popups.length-1; j >= 0; j--) {
    popups[j].y -= 1.2; popups[j].life--;
    if (popups[j].life <= 0) popups.splice(j, 1);
  }
}
function drawParticles() {
  particles.forEach(function(p) {
    ctx.globalAlpha = Math.min(1, p.life / 50);
    ctx.fillStyle = p.color;
    ctx.fillRect(p.x, p.y, p.size, p.size);
  });
  ctx.globalAlpha = 1;
  popups.forEach(function(pp) {
    ctx.globalAlpha = pp.life / pp.maxLife;
    ctx.fillStyle = pp.color;
    ctx.font = "bold 18px monospace";
    ctx.textAlign = "center";
    ctx.fillText(pp.text, pp.x, pp.y);
  });
  ctx.globalAlpha = 1;
}

// ── HUD ──
function drawHUD() {
  ctx.fillStyle = "rgba(0,0,0,0.4)";
  ctx.fillRect(8, 8, 120, 36);
  ctx.fillStyle = "#fff";
  ctx.font = "bold 14px monospace";
  ctx.textAlign = "left";
  ctx.fillText("SCORE: " + score, 16, 30);

  var lx = 140;
  for (var i = 0; i < MAX_LIVES; i++) {
    ctx.fillStyle = i < lives ? "#ff6b8a" : "rgba(255,255,255,0.15)";
    ctx.font = "16px monospace";
    ctx.fillText(i < lives ? "\u2665" : "\u2661", lx + i*20, 30);
  }

  if (bestScore > 0) {
    ctx.fillStyle = "rgba(0,0,0,0.3)";
    ctx.fillRect(lx + MAX_LIVES*20 + 8, 8, 100, 36);
    ctx.fillStyle = "#fbbf24";
    ctx.font = "bold 14px monospace";
    ctx.fillText("BEST: " + bestScore, lx + MAX_LIVES*20 + 16, 30);
  }

  if (score === 0 && !player.hurt && lives === MAX_LIVES) {
    ctx.fillStyle = "rgba(0,0,0,0.35)";
    ctx.font = "12px monospace";
    ctx.textAlign = "center";
    ctx.fillText("arrow keys to move \u2014 up to jump \u2014 hop over the wild bunny!", gameW/2, gameH - 20);
  }
}

// ── Wild bunny AI ──
function updateWildAI() {
  wild.aiTimer++;
  if (wild.hurt) {
    wild.hurtTimer--;
    if (wild.hurtTimer <= 0) wild.hurt = false;
    return;
  }
  if (wild.aiTimer >= wild.aiDur) {
    var d = difficulty();
    var r = Math.random();
    if (r < d.pIdle) {
      wild.aiState = "idle";
      wild.aiDur = Math.floor(d.idleDur + Math.random()*30);
      wild.vx = 0;
    } else if (r < d.pIdle + d.pRun) {
      wild.aiState = "run";
      wild.aiDur = Math.floor(d.runDur + Math.random()*40);
      wild.dir = Math.random() > 0.5 ? 1 : -1;
      wild.vx = wild.dir * (d.runMin + Math.random()*(d.runMax - d.runMin));
    } else {
      // BINKY
      wild.aiState = "binky";
      wild.aiDur = 25;
      if (wild.grounded) {
        wild.vy = JUMP_VEL * (d.binkyMin + Math.random()*(d.binkyMax - d.binkyMin));
        wild.grounded = false;
        wild.vx = (Math.random()-0.5) * d.binkyLat;
        wild.dir = wild.vx >= 0 ? 1 : -1;
      }
    }
    wild.aiTimer = 0;
  }
  if (wild.x < 20) { wild.x = 20; wild.dir = 1; wild.vx = Math.abs(wild.vx); }
  if (wild.x > gameW - PX - 20) { wild.x = gameW-PX-20; wild.dir = -1; wild.vx = -Math.abs(wild.vx); }
}

// ── Physics ──
function updatePhysics(b) {
  if (!b.grounded) {
    b.vy += GRAVITY;
    b.y -= b.vy;
    if (b.y <= 0) { b.y = 0; b.vy = 0; b.grounded = true; }
  }
  b.x += b.vx;
}

// ── Animation ──
function pickAnim(b) {
  if (b.layingDown) {
    b.anim = "LieDown";
    advanceFrame(b, 8);
    if (Math.floor(b.frame) >= SPRITES.LieDown.frames) b.frame = SPRITES.LieDown.frames - 1;
    return;
  }
  if (b.celebrating) {
    b.anim = "Liking";
    advanceFrame(b, 6);
    b.celebTimer--;
    if (b.celebTimer <= 0) b.celebrating = false;
    return;
  }
  if (b.hurt) {
    b.anim = "HurtRun";
    advanceFrame(b, 3);
    return;
  }
  if (!b.grounded) {
    b.anim = "Jumping";
    advanceFrame(b, 4);
  } else if (Math.abs(b.vx) > 0.5) {
    b.anim = "Running";
    advanceFrame(b, 4);
  } else {
    b.anim = "Idle";
    advanceFrame(b, 6);
  }
}

// ── Hitbox ──
function hitbox(b) {
  return { x1: b.x + HB_INSET, x2: b.x + PX - HB_INSET, y: b.y };
}

// ── Collision / scoring ──
function hurtPlayer() {
  lives--;
  player.hurt = true;
  player.hurtTimer = 35;
  player.frame = 0;
  player.vx = (player.x < wild.x ? -3 : 3);
  player.vy = JUMP_VEL * 0.3;
  player.grounded = false;
  shakeTimer = 8;
  iframeTimer = IFRAMES;   // start invincibility
  spawnParticles(player.x+PX/2, groundY-player.y-PX/2, "#ff6b8a", 10);

  if (lives <= 0) {
    spawnPopup(gameW/2, gameH/2 - 40, "tuckered out!", "#ff6b8a");
  } else {
    spawnPopup(player.x+PX/2, groundY-player.y-PX-10, "bonk!", "#ff6b8a");
  }
}

function checkScoring() {
  var ph = hitbox(player);
  var wh = hitbox(wild);
  var overlapX = ph.x1 < wh.x2 && ph.x2 > wh.x1;

  // Hop-over scoring
  if (!player.grounded && player.y > PX*0.35 && overlapX && !player.passedOver && !player.hurt) {
    player.passedOver = true;
    score++;
    if (score > bestScore) {
      bestScore = score;
      localStorage.setItem("binky_best", bestScore.toString());
    }
    spawnParticles(player.x+PX/2, groundY-player.y-PX, "#fbbf24", 12);
    spawnParticles(player.x+PX/2, groundY-player.y-PX, "#4ade80", 8);
    spawnPopup(player.x+PX/2, groundY-player.y-PX-20, "+1", "#4ade80");
    shakeTimer = 5;

    if (score === 5 && lastThreshold < 5) {
      spawnPopup(gameW/2, gameH/2 - 40, "nice hops!", "#fbbf24");
      lastThreshold = 5;
    } else if (score === 10 && lastThreshold < 10) {
      spawnPopup(gameW/2, gameH/2 - 40, "hopping master!", "#a855f7");
      lastThreshold = 10;
    } else if (score === 25 && lastThreshold < 25) {
      spawnPopup(gameW/2, gameH/2 - 40, "BINKY LEGEND", "#ff3366");
      lastThreshold = 25;
    } else if (score === 50 && lastThreshold < 50) {
      spawnPopup(gameW/2, gameH/2 - 40, "TRANSCENDENT", "#38bdf8");
      lastThreshold = 50;
    } else if (score === 100 && lastThreshold < 100) {
      spawnPopup(gameW/2, gameH/2 - 40, "IMPOSSIBLE", "#fff");
      lastThreshold = 100;
    }
  }
  if (player.grounded) player.passedOver = false;

  // No collision during invincibility
  if (iframeTimer > 0) return;

  // Mid-air collision
  if (!wild.grounded && !player.grounded && overlapX && !player.hurt) {
    hurtPlayer();
  }
  // Ground collision — wild runs into player
  if (wild.grounded && player.grounded && overlapX && Math.abs(wild.vx) > 1.2 && !player.hurt) {
    hurtPlayer();
  }
}

// ── Input ──
function onKeyDown(e) {
  keys[e.key] = true;
  if (state === "title") {
    state = "playing";
    titleAlpha = 1;
  }
  if (state === "gameover" && (e.key === " " || e.key === "Enter")) {
    resetGame();
    state = "playing";
  }
  if (["ArrowUp","ArrowDown","ArrowLeft","ArrowRight"," "].indexOf(e.key) >= 0) {
    e.preventDefault();
  }
}
function onKeyUp(e) { keys[e.key] = false; }

// ── Main update ──
function update() {
  if (state === "title") { titleTimer++; return; }
  if (state === "gameover") {
    gameOverTimer++;
    updateParticles();
    pickAnim(player);
    pickAnim(wild);
    return;
  }

  if (titleAlpha > 0) titleAlpha -= 0.02;

  // Tick invincibility
  if (iframeTimer > 0) iframeTimer--;

  // Game over check
  if (lives <= 0 && !player.hurt) {
    state = "gameover";
    gameOverTimer = 0;
    player.layingDown = true;
    player.frame = 0;
    player.vx = 0;
    return;
  }

  // Player input
  if (!player.hurt) {
    if (keys["ArrowLeft"] || keys["a"]) {
      player.vx = -MOVE_SPD; player.dir = -1;
    } else if (keys["ArrowRight"] || keys["d"]) {
      player.vx = MOVE_SPD; player.dir = 1;
    } else {
      player.vx *= 0.75;
    }
    if ((keys["ArrowUp"] || keys["w"] || keys[" "]) && player.grounded) {
      player.vy = JUMP_VEL;
      player.grounded = false;
      spawnParticles(player.x+PX/2, groundY, "#8B6914", 4);
    }
  } else {
    player.hurtTimer--;
    if (player.hurtTimer <= 0) { player.hurt = false; player.vx = 0; }
  }

  if (player.x < 0) player.x = 0;
  if (player.x > gameW - PX) player.x = gameW - PX;

  updatePhysics(player);
  updatePhysics(wild);
  updateWildAI();
  pickAnim(player);
  pickAnim(wild);
  checkScoring();
  updateParticles();

  if (shakeTimer > 0) {
    shakeX = (Math.random()-0.5) * shakeTimer * 1.5;
    shakeY = (Math.random()-0.5) * shakeTimer * 1.2;
    shakeTimer--;
  } else { shakeX = 0; shakeY = 0; }

  if (wild.aiState === "idle" && wild.grounded) wild.vx *= 0.85;
}

// ── Draw ──
function draw() {
  ctx.clearRect(0, 0, gameW, gameH);
  ctx.save();
  ctx.translate(shakeX, shakeY);
  if (bgGenerated) ctx.drawImage(bgCanvas, 0, 0);

  if (state === "title") {
    drawTitleScreen();
  } else {
    if (titleAlpha > 0) drawTitleScreen();
    drawSprite(wild, false);
    drawSprite(player, iframeTimer > 0);  // blink during iframes
    drawParticles();
    drawHUD();
    if (state === "gameover") drawGameOver();
  }
  ctx.restore();
}

function drawTitleScreen() {
  ctx.save();
  ctx.globalAlpha = state === "title" ? 1 : titleAlpha;
  ctx.fillStyle = "rgba(0,0,0,0.5)";
  ctx.fillRect(0, 0, gameW, gameH);
  ctx.font = "bold 48px monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillStyle = "rgba(0,0,0,0.4)";
  ctx.fillText("Binky", gameW/2 + 3, gameH/2 - 20 + 3);
  ctx.fillStyle = "#fbbf24";
  ctx.fillText("Binky", gameW/2, gameH/2 - 20);
  ctx.font = "14px monospace";
  ctx.fillStyle = "rgba(255,255,255,0.7)";
  if (state === "title") ctx.fillText("press any key", gameW/2, gameH/2 + 25);
  if (state === "title") {
    var bounce = Math.sin(titleTimer * 0.08) * 8;
    var img = images["BunnyBrown_Idle"];
    if (img && img.complete) {
      var f = Math.floor(titleTimer / 8) % SPRITES.Idle.frames;
      ctx.drawImage(img, f*32, 0, 32, 32, gameW/2 - PX/2, gameH/2 + 45 + bounce, PX, PX);
    }
  }
  ctx.restore();
}

function drawGameOver() {
  var a = Math.min(1, gameOverTimer / 40);
  ctx.globalAlpha = a;
  ctx.fillStyle = "rgba(0,0,0,0.35)";
  ctx.fillRect(0, gameH*0.3, gameW, gameH*0.35);
  ctx.fillStyle = "#fbbf24";
  ctx.font = "bold 28px monospace";
  ctx.textAlign = "center";
  ctx.fillText("tuckered out", gameW/2, gameH*0.42);
  ctx.fillStyle = "#fff";
  ctx.font = "16px monospace";
  ctx.fillText("score: " + score, gameW/2, gameH*0.50);
  if (score >= bestScore && score > 0) {
    ctx.fillStyle = "#4ade80";
    ctx.fillText("new best!", gameW/2, gameH*0.56);
  }
  ctx.fillStyle = "rgba(255,255,255,0.5)";
  ctx.font = "13px monospace";
  ctx.fillText("space to play again", gameW/2, gameH*0.63);
  ctx.globalAlpha = 1;
}

// ── Game loop ──
var raf;
function loop() { update(); draw(); raf = requestAnimationFrame(loop); }

function resize() {
  var panel = document.getElementById("binky-panel");
  if (!panel) return;
  var rect = panel.getBoundingClientRect();
  var header = document.getElementById("binky-panel-header");
  var hh = header ? header.offsetHeight : 30;
  gameW = Math.floor(rect.width);
  gameH = Math.floor(rect.height - hh);
  if (gameW < 100 || gameH < 100) return;
  canvas.width = gameW;
  canvas.height = gameH;
  groundY = Math.floor(gameH * 0.78);
  generateBG(gameW, gameH);
}

window.BinkyGame = {
  start: function() {
    var container = document.getElementById("binky-canvas-wrap");
    if (!container) return;
    canvas = document.getElementById("binky-canvas");
    if (!canvas) {
      canvas = document.createElement("canvas");
      canvas.id = "binky-canvas";
      canvas.style.width = "100%";
      canvas.style.display = "block";
      container.appendChild(canvas);
    }
    ctx = canvas.getContext("2d");
    ctx.imageSmoothingEnabled = false;
    resize();
    state = "title";
    titleTimer = 0;
    titleAlpha = 1;
    resetGame();
    document.addEventListener("keydown", onKeyDown);
    document.addEventListener("keyup", onKeyUp);
    window.addEventListener("resize", resize);
    if (window.ResizeObserver) {
      new ResizeObserver(resize).observe(document.getElementById("binky-panel"));
    }
    loop();
  },
  stop: function() {
    if (raf) cancelAnimationFrame(raf);
    raf = null;
    document.removeEventListener("keydown", onKeyDown);
    document.removeEventListener("keyup", onKeyUp);
    keys = {};
    state = "title";
  },
  resize: resize,
};

})();
