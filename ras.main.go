package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"time"

	"github.com/aquilax/go-perlin" // Perlin noise package
	"github.com/tfriedel6/canvas"
	"github.com/tfriedel6/canvas/sdlcanvas"
)

const (
	Width       = 1080
	Height      = 1440
	ParticleNum = 3000 // Start with a smaller number for performance
	TrailAlpha  = 0.05 // Background fade for trails
	MaxSpeed    = 4.0
	Scale       = 0.005 // Perlin noise scale
	ForceFactor = 0.3   // Strength of noise influence
)

// ---------------- Particle ----------------
type Particle struct {
	X, Y    float64
	VX, VY  float64
	Size    float64
	BaseHue float64
}

// ---------------- Simulation ----------------
type Simulation struct {
	Particles []*Particle
	Canvas    *canvas.Canvas
	Noise     *perlin.Perlin
}

func NewSimulation(cv *canvas.Canvas) *Simulation {
	noise := perlin.NewPerlin(rand.Float64(), rand.Float64(), 2, 256)
	sim := &Simulation{
		Particles: make([]*Particle, 0, ParticleNum),
		Canvas:    cv,
		Noise:     noise,
	}

	for i := 0; i < ParticleNum; i++ {
		sim.Particles = append(sim.Particles, &Particle{
			X:       rand.Float64() * Width,
			Y:       rand.Float64() * Height,
			VX:      0,
			VY:      0,
			Size:    1.5 + rand.Float64()*2,
			BaseHue: rand.Float64() * 360, // Assign a base hue per particle
		})
	}
	return sim
}

// ---------------- Color Conversion ----------------
func HSVtoHex(h, s, v float64) string {
	h = math.Mod(h, 360)
	c := v * s
	x := c * (1 - math.Abs(math.Mod(h/60, 2)-1))
	m := v - c
	var r, g, b float64

	switch {
	case h < 60:
		r, g, b = c, x, 0
	case h < 120:
		r, g, b = x, c, 0
	case h < 180:
		r, g, b = 0, c, x
	case h < 240:
		r, g, b = 0, x, c
	case h < 300:
		r, g, b = x, 0, c
	default:
		r, g, b = c, 0, x
	}
	return fmt.Sprintf("#%02X%02X%02X", int((r+m)*255), int((g+m)*255), int((b+m)*255))
}

// ---------------- Simulation Step ----------------
func (sim *Simulation) Update() {
	for _, p := range sim.Particles {
		// Perlin noise to determine angle for smooth swirling
		angle := (sim.Noise.Noise2D(p.X*Scale, p.Y*Scale) + 1) / 2 * 2 * math.Pi
		p.VX += math.Cos(angle) * ForceFactor
		p.VY += math.Sin(angle) * ForceFactor

		// Limit max speed
		speed := math.Sqrt(p.VX*p.VX + p.VY*p.VY)
		if speed > MaxSpeed {
			p.VX = p.VX / speed * MaxSpeed
			p.VY = p.VY / speed * MaxSpeed
		}

		// Update position
		p.X += p.VX
		p.Y += p.VY

		// Toroidal wrap for continuous motion
		if p.X < 0 {
			p.X += Width
		} else if p.X > Width {
			p.X -= Width
		}
		if p.Y < 0 {
			p.Y += Height
		} else if p.Y > Height {
			p.Y -= Height
		}
	}
}

// ---------------- Drawing ----------------
func (sim *Simulation) Draw() {
	cv := sim.Canvas

	// Fade background for motion trails
	cv.SetFillStyle(fmt.Sprintf("rgba(0,0,0,%.3f)", TrailAlpha))
	cv.FillRect(0, 0, Width, Height)

	// Draw each particle
	for _, p := range sim.Particles {
		speed := math.Sqrt(p.VX*p.VX + p.VY*p.VY)
		hue := math.Mod(p.BaseHue+speed*90, 360) // Smooth color transition
		color := HSVtoHex(hue, 1, 0.6+0.4*(speed/MaxSpeed))
		cv.SetFillStyle(color)
		cv.FillRect(p.X, p.Y, p.Size, p.Size)
	}
}

// ---------------- Main ----------------
func main() {
	wnd, cv, err := sdlcanvas.CreateWindow(Width, Height, "Perlin Cosmic Swirl")
	if err != nil {
		log.Fatal(err)
	}

	rand.Seed(time.Now().UnixNano())
	sim := NewSimulation(cv)

	wnd.MainLoop(func() {

		sim.Update()
		sim.Draw()

		// FPS display

	})
}
