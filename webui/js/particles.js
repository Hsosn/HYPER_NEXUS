/**
 * Sci-Fi Particle Network — animated background
 *
 * Drawing: 1 – particle, 2 – connections, 3 – glow
 * to avoid double-count overhead we pre-calc and batch.
 * Runs at roughly 30fps via requestAnimationFrame throttle.
 */

class ParticleNetwork {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.particles = [];
    this.raf = null;
    this.lastFrame = 0;
    this.boundResize = this.resize.bind(this);
    window.addEventListener('resize', this.boundResize);
    this.resize();
    this.init();
    this.loop(0);
  }

  resize() {
    this.canvas.width = window.innerWidth;
    this.canvas.height = window.innerHeight;
  }

  init() {
    const count = Math.min(80, Math.floor((this.canvas.width * this.canvas.height) / 14000));
    this.particles = [];
    for (let i = 0; i < count; i++) {
      this.particles.push({
        x: Math.random() * this.canvas.width,
        y: Math.random() * this.canvas.height,
        vx: (Math.random() - 0.5) * 0.2,
        vy: (Math.random() - 0.5) * 0.2,
        r: Math.random() * 1.2 + 0.5,
      });
    }
  }

  loop(ts) {
    if (ts - this.lastFrame < 33) { this.raf = requestAnimationFrame(t => this.loop(t)); return; }
    this.lastFrame = ts;
    this.update();
    this.draw();
    this.raf = requestAnimationFrame(t => this.loop(t));
  }

  update() {
    const w = this.canvas.width, h = this.canvas.height;
    for (const p of this.particles) {
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < 0) p.x = w;
      if (p.x > w) p.x = 0;
      if (p.y < 0) p.y = h;
      if (p.y > h) p.y = 0;
    }
  }

  draw() {
    const ctx = this.ctx;
    const w = this.canvas.width, h = this.canvas.height;
    ctx.clearRect(0, 0, w, h);

    const particles = this.particles;
    const len = particles.length;
    const connDist = 120;

    // Draw connections first (behind)
    ctx.lineWidth = 0.4;
    for (let i = 0; i < len; i++) {
      const a = particles[i];
      for (let j = i + 1; j < len; j++) {
        const b = particles[j];
        const dx = a.x - b.x, dy = a.y - b.y;
        const dist = dx * dx + dy * dy;
        if (dist < connDist * connDist) {
          const alpha = (1 - dist / (connDist * connDist)) * 0.25;
          ctx.strokeStyle = `rgba(0, 229, 255, ${alpha})`;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }
    }

    // Draw particles
    for (const p of particles) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(0, 229, 255, 0.35)';
      ctx.fill();
      // small glow
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r * 2.5, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(0, 229, 255, 0.04)';
      ctx.fill();
    }
  }

  destroy() {
    window.removeEventListener('resize', this.boundResize);
    if (this.raf) cancelAnimationFrame(this.raf);
  }
}

/** Create the particle canvas and inject into DOM */
export function initParticles() {
  // Avoid duplicate canvas
  if (document.getElementById('particle-canvas')) return;

  const canvas = document.createElement('canvas');
  canvas.id = 'particle-canvas';
  canvas.style.cssText = 'position:fixed;inset:0;z-index:0;pointer-events:none;';
  document.body.prepend(canvas);

  const net = new ParticleNetwork(canvas);

  // Pause when tab hidden
  const onVis = () => {
    if (document.hidden && net.raf) { cancelAnimationFrame(net.raf); net.raf = null; }
    else if (!document.hidden && !net.raf) net.raf = requestAnimationFrame(t => net.loop(t));
  };
  document.addEventListener('visibilitychange', onVis);

  // Store for cleanup
  canvas._particleNet = net;
  canvas._visHandler = onVis;
}
