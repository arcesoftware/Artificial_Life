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

const (
	gridW      = 240
	gridH      = 160
	radius     = 6.0
	dtDefault  = 0.08
	muDefault  = 0.30
	sigDefault = 0.06
)

type KernelEntry struct {
	dx, dy int
	w      float64
}

type Game struct {
	A, Anext [][]float64
	kernel   []KernelEntry
	dt       float64
	mu       float64
	sigma    float64
	texture  *ebiten.Image

	// Camera
	camX, camY     float64
	camZoom        float64
	lastMx, lastMy int
	rightDragging  bool

	frame   int
	start   time.Time
	lastFPS int
}

// ---- Utils ----
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
	return (x%m + m) % m
}

// ---- Kernel ----
type KernelFunc func(float64) float64

func buildKernel(R float64) ([]KernelEntry, float64) {
	shellSigma := 0.15
	Kc := func(rNorm float64) float64 {
		x := (rNorm - 0.5) / shellSigma
		return math.Exp(-0.5 * x * x)
	}

	Ri := int(math.Ceil(R))
	var entries []KernelEntry
	var sum float64
	for dy := -Ri; dy <= Ri; dy++ {
		for dx := -Ri; dx <= Ri; dx++ {
			dist := math.Hypot(float64(dx), float64(dy))
			if dist <= R {
				rn := dist / R
				w := Kc(rn)
				entries = append(entries, KernelEntry{dx, dy, w})
				sum += w
			}
		}
	}
	for i := range entries {
		entries[i].w /= sum
	}
	return entries, 1.0
}

func growth(u, mu, sigma float64) float64 {
	if sigma <= 0 {
		return 0
	}
	return 2*math.Exp(-((u-mu)*(u-mu))/(2*sigma*sigma)) - 1
}

// ---- Init ----
func NewGame() *Game {
	A := make([][]float64, gridH)
	Anext := make([][]float64, gridH)
	for y := range A {
		A[y] = make([]float64, gridW)
		Anext[y] = make([]float64, gridW)
	}

	cx, cy := gridW/2, gridH/2
	for y := 0; y < gridH; y++ {
		for x := 0; x < gridW; x++ {
			d := math.Hypot(float64(x-cx), float64(y-cy))
			if d < 16 {
				A[y][x] = 0.8 * math.Exp(-d*d/(2*8*8))
			}
			if rand.Float64() < 0.001 {
				A[y][x] = rand.Float64()*0.8 + 0.1
			}
		}
	}

	kernel, _ := buildKernel(radius)
	tex := ebiten.NewImage(gridW, gridH)

	return &Game{
		A:       A,
		Anext:   Anext,
		kernel:  kernel,
		dt:      dtDefault,
		mu:      muDefault,
		sigma:   sigDefault,
		texture: tex,
		camZoom: 4, // initial zoom factor
		start:   time.Now(),
	}
}

// ---- Simulation ----
func (g *Game) step() {
	for y := 0; y < gridH; y++ {
		for x := 0; x < gridW; x++ {
			var u float64
			for _, k := range g.kernel {
				nx := wrap(x+k.dx, gridW)
				ny := wrap(y+k.dy, gridH)
				u += k.w * g.A[ny][nx]
			}
			val := g.A[y][x] + g.dt*growth(u, g.mu, g.sigma)
			g.Anext[y][x] = clamp(val, 0, 1)
		}
	}
	g.A, g.Anext = g.Anext, g.A
}

// ---- Input / Camera ----
func (g *Game) handleCamera() {
	// Zoom with scroll wheel
	_, scrollY := ebiten.Wheel()
	if scrollY != 0 {
		oldZoom := g.camZoom
		g.camZoom *= math.Pow(1.1, scrollY)
		if g.camZoom < 1 {
			g.camZoom = 1
		}
		// Keep mouse position stable (zoom to cursor)
		mx, my := ebiten.CursorPosition()
		dx := float64(mx)/oldZoom - (g.camX + float64(mx)/g.camZoom)
		dy := float64(my)/oldZoom - (g.camY + float64(my)/g.camZoom)
		g.camX += dx
		g.camY += dy
	}

	// Right mouse drag for pan
	mx, my := ebiten.CursorPosition()
	if ebiten.IsMouseButtonPressed(ebiten.MouseButtonRight) {
		if !g.rightDragging {
			g.rightDragging = true
			g.lastMx, g.lastMy = mx, my
		} else {
			dx := float64(mx-g.lastMx) / g.camZoom
			dy := float64(my-g.lastMy) / g.camZoom
			g.camX -= dx
			g.camY -= dy
			g.lastMx, g.lastMy = mx, my
		}
	} else {
		g.rightDragging = false
	}

	// WSAD keyboard pan
	speed := 5.0 / g.camZoom
	if ebiten.IsKeyPressed(ebiten.KeyW) {
		g.camY -= speed
	}
	if ebiten.IsKeyPressed(ebiten.KeyS) {
		g.camY += speed
	}
	if ebiten.IsKeyPressed(ebiten.KeyA) {
		g.camX -= speed
	}
	if ebiten.IsKeyPressed(ebiten.KeyD) {
		g.camX += speed
	}
}

// ---- Ebiten loop ----
func (g *Game) Update() error {
	g.handleCamera()
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
			r, gg, b := colorRamp(v)
			g.texture.Set(x, y, color.NRGBA{r, gg, b, 0xFF})
		}
	}

	op := &ebiten.DrawImageOptions{}
	op.GeoM.Scale(g.camZoom, g.camZoom)
	op.GeoM.Translate(-g.camX*g.camZoom, -g.camY*g.camZoom)
	screen.DrawImage(g.texture, op)

	text.Draw(screen,
		fmt.Sprintf("Zoom: %.2f  Cam:(%.1f,%.1f) FPS:%d", g.camZoom, g.camX, g.camY, g.lastFPS),
		basicfont.Face7x13, 6, 16, color.White)
	text.Draw(screen, "Controls: Scroll=Zoom  WSAD=Move  Right-drag=Pan",
		basicfont.Face7x13, 6, 32, color.White)
}

func (g *Game) Layout(outW, outH int) (int, int) {
	return 800, 600
}

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

func main() {
	ebiten.SetWindowSize(800, 600)
	ebiten.SetWindowTitle("Lenia with Camera Controls")

	if err := ebiten.RunGame(NewGame()); err != nil {
		log.Fatal(err)
	}
}
