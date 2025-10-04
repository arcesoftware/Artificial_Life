package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"time"

	"github.com/aquilax/go-perlin"
	"github.com/tfriedel6/canvas"
	"github.com/tfriedel6/canvas/sdlcanvas"
)

const (
	Width       = 1080
	Height      = 1440
	ParticleNum = 3000
	TrailAlpha  = 0.18
	MaxSpeed    = 4.0
	Scale       = 0.15
	ForceFactor = 3.14159
	MassRadius  = 15.0 // Radius for local mass redistribution
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
			BaseHue: rand.Float64() * 360,
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
	// Apply Perlin noise motion
	for _, p := range sim.Particles {
		angle := (sim.Noise.Noise2D(p.X*Scale, p.Y*Scale) + 1) / 2 * 2 * math.Pi
		p.VX += math.Cos(angle) * ForceFactor
		p.VY += math.Sin(angle) * ForceFactor
	}

	// Apply mass-conserving correction (MaCE-inspired)
	for i, a := range sim.Particles {
		for j := i + 1; j < len(sim.Particles); j++ {
			b := sim.Particles[j]
			dx := b.X - a.X
			dy := b.Y - a.Y

			// Toroidal distance
			if dx > Width/2 {
				dx -= Width
			} else if dx < -Width/2 {
				dx += Width
			}
			if dy > Height/2 {
				dy -= Height
			} else if dy < -Height/2 {
				dy += Height
			}

			distSq := dx*dx + dy*dy
			if distSq < MassRadius*MassRadius && distSq > 0 {
				// Redistribute velocities to conserve local motion
				fx := (a.VX - b.VX) * 0.5
				fy := (a.VY - b.VY) * 0.5
				a.VX -= fx
				a.VY -= fy
				b.VX += fx
				b.VY += fy
			}
		}
	}

	// Update positions and limit speed
	for _, p := range sim.Particles {
		speed := math.Sqrt(p.VX*p.VX + p.VY*p.VY)
		if speed > MaxSpeed {
			p.VX = p.VX / speed * MaxSpeed
			p.VY = p.VY / speed * MaxSpeed
		}

		p.X += p.VX
		p.Y += p.VY

		// Toroidal wrap
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

	cv.SetFillStyle(fmt.Sprintf("rgba(0,0,0,%.3f)", TrailAlpha))
	cv.FillRect(0, 0, Width, Height)

	for _, p := range sim.Particles {
		speed := math.Sqrt(p.VX*p.VX + p.VY*p.VY)
		hue := math.Mod(p.BaseHue+speed*120, 360)
		color := HSVtoHex(hue, 1, 0.6+0.4*(speed/MaxSpeed))
		cv.SetFillStyle(color)
		cv.FillRect(p.X, p.Y, p.Size, p.Size)
	}
}

// ---------------- Main ----------------
func main() {
	wnd, cv, err := sdlcanvas.CreateWindow(Width, Height, "MaCE Cosmic Swirl")
	if err != nil {
		log.Fatal(err)
	}

	rand.Seed(time.Now().UnixNano())
	sim := NewSimulation(cv)

	wnd.MainLoop(func() {
		sim.Update()
		sim.Draw()
	})
}
