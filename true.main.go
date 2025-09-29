// lenia_lorenz.go
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

	"gonum.org/v1/gonum/dsp/fourier"
)

// ---------- Simulation parameters ----------
const (
	gridW    = 128
	gridH    = 128
	cellSize = 4
)

// ---------- Genome ----------
type Genome struct {
	Mu, Sigma, Dt float64
	ColorBias     float64
}

// ---------- Kernel ----------
func buildKernel(R float64) [][]float64 {
	kernel := make([][]float64, gridH)
	for y := 0; y < gridH; y++ {
		kernel[y] = make([]float64, gridW)
	}
	shellSigma := 0.15
	cx, cy := gridW/2, gridH/2
	for y := 0; y < gridH; y++ {
		for x := 0; x < gridW; x++ {
			dx := float64(x - cx)
			dy := float64(y - cy)
			dist := math.Hypot(dx, dy)
			if dist <= R {
				rn := dist / R
				val := math.Exp(-0.5 * math.Pow((rn-0.5)/shellSigma, 2))
				kernel[y][x] = val
			}
		}
	}
	return kernel
}

// ---------- Lorenz Attractor ----------
type Lorenz struct {
	x, y, z       float64
	sigma, rho, b float64
	dt            float64
}

func NewLorenz() *Lorenz {
	return &Lorenz{x: 0.1, y: 0, z: 0, sigma: 10, rho: 28, b: 8.0 / 3.0, dt: 0.01}
}

func (l *Lorenz) Step() {
	dx := l.sigma * (l.y - l.x)
	dy := l.x*(l.rho-l.z) - l.y
	dz := l.x*l.y - l.b*l.z
	l.x += dx * l.dt
	l.y += dy * l.dt
	l.z += dz * l.dt
}

// ---------- Game ----------
type Game struct {
	A       [][]float64
	Anext   [][]float64
	genome  Genome
	texture *ebiten.Image

	fft2d     *fourier.FFT2
	kernelFFT []complex128

	lorenz  *Lorenz
	frame   int
	start   time.Time
	lastFPS int
}

func NewGame() *Game {
	rand.Seed(time.Now().UnixNano())

	A := make([][]float64, gridH)
	Anext := make([][]float64, gridH)
	for y := 0; y < gridH; y++ {
		A[y] = make([]float64, gridW)
		Anext[y] = make([]float64, gridW)
	}

	// initial blob
	cx, cy := gridW/2, gridH/2
	for y := 0; y < gridH; y++ {
		for x := 0; x < gridW; x++ {
			d := math.Hypot(float64(x-cx), float64(y-cy))
			if d < 12 {
				A[y][x] = 0.9 * math.Exp(-d*d/(2*6*6))
			}
			if rand.Float64() < 0.002 {
				A[y][x] = rand.Float64()
			}
		}
	}

	fft2d := fourier.NewFFT2(gridH, gridW)
	kernel := buildKernel(6)
	kernelFlat := flatten(kernel)
	kernelFFT := fft2d.Coefficients(nil, kernelFlat)

	return &Game{
		A:       A,
		Anext:   Anext,
		genome:  Genome{Mu: 0.3, Sigma: 0.06, Dt: 0.08, ColorBias: 0},
		texture: ebiten.NewImage(gridW, gridH),

		fft2d:     fft2d,
		kernelFFT: kernelFFT,
		lorenz:    NewLorenz(),
		start:     time.Now(),
	}
}

// ---------- Helpers ----------
func flatten(m [][]float64) []complex128 {
	out := make([]complex128, len(m)*len(m[0]))
	idx := 0
	for y := 0; y < len(m); y++ {
		for x := 0; x < len(m[0]); x++ {
			out[idx] = complex(m[y][x], 0)
			idx++
		}
	}
	return out
}

func reshape(data []complex128) [][]float64 {
	out := make([][]float64, gridH)
	idx := 0
	for y := 0; y < gridH; y++ {
		out[y] = make([]float64, gridW)
		for x := 0; x < gridW; x++ {
			out[y][x] = real(data[idx])
			idx++
		}
	}
	return out
}

func clamp(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// Growth function
func growth(u, mu, sigma float64) float64 {
	if sigma <= 0 {
		return 0
	}
	return 2*math.Exp(-math.Pow(u-mu, 2)/(2*sigma*sigma)) - 1
}

// ---------- Step ----------
func (g *Game) step() {
	// FFT convolution
	stateFFT := g.fft2d.Coefficients(nil, flatten(g.A))
	for i := range stateFFT {
		stateFFT[i] *= g.kernelFFT[i]
	}
	conv := g.fft2d.Sequence(nil, stateFFT)
	U := reshape(conv)

	// Lorenz modulation
	g.lorenz.Step()
	g.genome.Mu = clamp(g.genome.Mu+0.002*math.Tanh(g.lorenz.x/20), 0.01, 1.0)
	g.genome.Sigma = clamp(g.genome.Sigma*(1+0.001*g.lorenz.y), 0.001, 1.0)
	g.genome.ColorBias = math.Tanh(g.lorenz.z / 30)

	// Update grid
	for y := 0; y < gridH; y++ {
		for x := 0; x < gridW; x++ {
			u := U[y][x]
			grow := growth(u, g.genome.Mu, g.genome.Sigma)
			val := g.A[y][x] + g.genome.Dt*grow
			g.Anext[y][x] = clamp(val, 0, 1)
		}
	}
	g.A, g.Anext = g.Anext, g.A
}

// ---------- Ebiten interface ----------
func (g *Game) Update() error {
	g.step()
	g.frame++
	if g.frame%30 == 0 {
		elapsed := time.Since(g.start).Seconds()
		g.lastFPS = int(float64(g.frame) / elapsed)
	}
	return nil
}

func (g *Game) Draw(screen *ebiten.Image) {
	for y := 0; y < gridH; y++ {
		for x := 0; x < gridW; x++ {
			v := g.A[y][x]
			r, gg, b := colorRamp(v + g.genome.ColorBias*0.2)
			g.texture.Set(x, y, color.NRGBA{R: r, G: gg, B: b, A: 0xFF})
		}
	}
	op := &ebiten.DrawImageOptions{}
	op.GeoM.Scale(float64(cellSize), float64(cellSize))
	screen.DrawImage(g.texture, op)

	txt := fmt.Sprintf("μ: %.3f σ: %.3f Δt: %.3f FPS: %d", g.genome.Mu, g.genome.Sigma, g.genome.Dt, g.lastFPS)
	text.Draw(screen, txt, basicfont.Face7x13, 6, 18, color.White)
}

func (g *Game) Layout(outW, outH int) (int, int) {
	return gridW * cellSize, gridH * cellSize
}

// ---------- Color ----------
func colorRamp(v float64) (r, g, b uint8) {
	v = clamp(v, 0, 1)
	if v < 0.5 {
		t := v / 0.5
		return uint8(20 + 50*t), uint8(50 + 150*t), uint8(200 - 100*t)
	}
	t := (v - 0.5) / 0.5
	return uint8(70 + 180*t), uint8(200 - 80*t), uint8(100 + 150*t)
}

// ---------- Main ----------
func main() {
	ebiten.SetWindowSize(gridW*cellSize, gridH*cellSize)
	ebiten.SetWindowTitle("Lenia + Lorenz Artificial Life")

	game := NewGame()
	if err := ebiten.RunGame(game); err != nil {
		log.Fatal(err)
	}
}
