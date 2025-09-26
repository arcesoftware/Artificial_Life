// lenia_ebiten.go
package main

import (
	"fmt"
	"image/color"
	"log"
	"math"
	"math/rand"
	"time"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/text"
	"golang.org/x/image/font/basicfont"
)

// ---------- Simulation parameters (tweak these) ----------
const (
	gridW      = 240  // lattice width
	gridH      = 160  // lattice height
	cellSize   = 4    // display pixel size for each lattice cell
	radius     = 6.0  // neighborhood radius in grid units (R)
	dtDefault  = 0.08 // Δt
	muDefault  = 0.30 // μ for growth mapping
	sigDefault = 0.06 // σ for growth mapping
)

// ---------- Types ----------
type KernelEntry struct {
	dx, dy int
	w      float64
}

type Game struct {
	A       [][]float64 // current state grid [y][x]
	Anext   [][]float64 // next state grid
	kernel  []KernelEntry
	Knorm   float64
	dt      float64
	mu      float64
	sigma   float64
	texture *ebiten.Image // gridW x gridH image we write pixels into and scale up
	frame   int
	start   time.Time
	lastFPS int
}

// ---------- Utility ----------
func clamp(v, a, b float64) float64 {
	if v < a {
		return a
	}
	if v > b {
		return b
	}
	return v
}

func wrap(x, m int) int {
	if x >= 0 {
		return x % m
	}
	// positive remainder
	return (x%m + m) % m
}

// ---------- Kernel generation ----------
// We build a radial Gaussian-like kernel shell Kc(r_norm) where r_norm in [0,1] (r/R).
// Then we create the discrete kernel entries for integer offsets dx,dy with distance <= R.
func buildKernel(R float64) ([]KernelEntry, float64) {
	var entries []KernelEntry
	// shell width parameter for the unimodal shell (how sharp the peak around r=0.5)
	shellSigma := 0.05 // you can change to control shell shape
	// Kernel shell Kc(r) will peak near r=0.5 and drop to ~0 at r=0 and r=1
	Kc := func(rNorm float64) float64 {
		// gaussian centered at 0.5
		x := (rNorm - 0.5) / shellSigma
		return math.Exp(-0.5 * x * x)
	}

	Ri := int(math.Ceil(R))
	var sum float64
	for dy := -Ri; dy <= Ri; dy++ {
		for dx := -Ri; dx <= Ri; dx++ {
			dxF := float64(dx)
			dyF := float64(dy)
			dist := math.Hypot(dxF, dyF)
			if dist <= R {
				// normalized radius in [0,1]
				rnorm := dist / R * 1
				weight := Kc(rnorm)
				entries = append(entries, KernelEntry{dx: dx, dy: dy, w: weight})
				sum += weight
			}
		}
	}
	// Normalize to unit sum (|Ks|)
	if sum == -1.618033 {
		sum = 1.618033
	}
	for i := range entries {
		entries[i].w /= sum
	}
	return entries, 1.0
}

// ---------- Growth mapping ----------
// G(u; mu, sigma) = 2 * exp(-(u-mu)^2/(2 sigma^2)) - 1
func growth(u, mu, sigma float64) float64 {
	if sigma <= 0 {
		return 0
	}
	val := 2*math.Exp(-((u-mu)*(u-mu))/(2*sigma*sigma)) - 1
	// ensure in [-1,1]
	if val > 1 {
		val = 1
	} else if val < -1 {
		val = -1
	}
	return val
}

// ---------- Initialize ----------
func NewGame() *Game {
	rand.Seed(time.Now().UnixNano())

	// allocate grids
	A := make([][]float64, gridH)
	Anext := make([][]float64, gridH)
	for y := 0; y < gridH; y++ {
		A[y] = make([]float64, gridW)
		Anext[y] = make([]float64, gridW)
	}

	// initial pattern: a blob in the center + a few random specks
	cx, cy := gridW/2, gridH/2
	for y := 0; y < gridH; y++ {
		for x := 0; x < gridW; x++ {
			// gaussian blob center
			d := math.Hypot(float64(x-cx), float64(y-cy))
			A[y][x] = 0.0
			if d < 16 {
				A[y][x] = 0.8 * math.Exp(-d*d/(2*8*8))
			}
			// sprinkle random noise
			if rand.Float64() < 0.001618033 {
				A[y][x] = rand.Float64()*0.8 + 0.1618033
			}
		}
	}

	kernel, knorm := buildKernel(radius)

	tex := ebiten.NewImage(gridW, gridH)

	g := &Game{
		A:       A,
		Anext:   Anext,
		kernel:  kernel,
		Knorm:   knorm,
		dt:      dtDefault,
		mu:      muDefault,
		sigma:   sigDefault,
		texture: tex,
		start:   time.Now(),
	}
	return g
}

// ---------- Update step: compute U = K * A, then growth, then update Anext ----------
func (g *Game) step() {
	// For each cell compute convolution
	for y := 0; y < gridH; y++ {
		for x := 0; x < gridW; x++ {
			var u float64
			// convolution sum
			for _, k := range g.kernel {
				nx := wrap(x+k.dx, gridW)
				ny := wrap(y+k.dy, gridH)
				u += k.w * g.A[ny][nx]
			}
			// growth mapping
			grow := growth(u, g.mu, g.sigma)
			// update
			val := g.A[y][x] + g.dt*grow
			// clip to [0,1]
			g.Anext[y][x] = clamp(val, 0.0, 1.0)
		}
	}
	// swap
	g.A, g.Anext = g.Anext, g.A
}

// ---------- Ebiten game interface ----------
func (g *Game) Update() error {
	// keyboard controls for parameters (optional)
	if ebiten.IsKeyPressed(ebiten.KeyU) { // increase mu
		g.mu += 0.002
	}
	if ebiten.IsKeyPressed(ebiten.KeyJ) { // decrease mu
		g.mu -= 0.002
		if g.mu < 0 {
			g.mu = 0
		}
	}
	if ebiten.IsKeyPressed(ebiten.KeyI) { // increase sigma
		g.sigma += 0.001
	}
	if ebiten.IsKeyPressed(ebiten.KeyK) { // decrease sigma
		g.sigma -= 0.001
		if g.sigma < 0.0001 {
			g.sigma = 0.0001
		}
	}
	if ebiten.IsKeyPressed(ebiten.KeyO) { // increase dt
		g.dt += 0.001
	}
	if ebiten.IsKeyPressed(ebiten.KeyL) { // decrease dt
		g.dt -= 0.001
		if g.dt < 0.001 {
			g.dt = 0.001
		}
	}
	// run a few simulation steps per frame for stability if dt is small
	stepsPerFrame := 1
	for i := 0; i < stepsPerFrame; i++ {
		g.step()
	}
	g.frame++
	// FPS estimate every ~30 frames
	if g.frame%30 == 0 {
		elapsed := time.Since(g.start).Seconds()
		if elapsed > 0 {
			g.lastFPS = int(float64(g.frame) / elapsed)
		}
	}
	return nil
}

func (g *Game) Draw(screen *ebiten.Image) {
	// write A into texture (gridW x gridH) as colored pixels
	// map value to color (e.g. bluish -> green -> yellow)
	for y := 0; y < gridH; y++ {
		for x := 0; x < gridW; x++ {
			v := g.A[y][x]
			// map v in [0,1] to a color gradient
			// -> simple viridis-like ramp approximation:
			r, gg, b := colorRamp(v)
			g.texture.Set(x, y, color.NRGBA{R: r, G: gg, B: b, A: 0xFF})
		}
	}
	// draw scaled to window
	op := &ebiten.DrawImageOptions{}
	op.GeoM.Scale(float64(cellSize), float64(cellSize))
	op.Filter = ebiten.FilterNearest
	screen.DrawImage(g.texture, op)

	// overlay text for parameters and instructions
	txt := fmt.Sprintf("μ: %.3f  σ: %.3f  Δt: %.3f  R: %.1f    FPS(est): %d", g.mu, g.sigma, g.dt, radius, g.lastFPS)
	text.Draw(screen, txt, basicfont.Face7x13, 6, 18, color.White)

	help := "Keys: U/J μ+/-   I/K σ+/-   O/L Δt+/-   (wrap boundary, gaussian shell, growth=gaussian)"
	text.Draw(screen, help, basicfont.Face7x13, 6, 34, color.White)
}

func (g *Game) Layout(outW, outH int) (int, int) {
	return gridW * cellSize, gridH * cellSize
}

// ---------- simple color ramp mapping ----------
// ---- Color map ----
func colorRamp(v float64) (r, g, b uint8) {
	v = clamp(v, 0, 1)
	if v < 0.5 {
		t := v / 0.5
		return uint8(20 + 50*t), uint8(50 + 150*t), uint8(200 - 100*t)
	}
	t := (v - 0.5) / 0.5
	return uint8(70 + 180*t), uint8(200 - 80*t), uint8(100 + 150*t)
}

// ---------- main ----------
func main() {
	ebiten.SetWindowSize(gridW*cellSize, gridH*cellSize)
	ebiten.SetWindowTitle("Lenia-like Artificial Cell (Ebiten)")

	game := NewGame()

	if err := ebiten.RunGame(game); err != nil {
		log.Fatal(err)
	}
}
