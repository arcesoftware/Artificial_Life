package main

import (
	"image/color"
	"log"
	"math"

	"math/rand/v2" // Modern random generator

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
	"github.com/hajimehoshi/ebiten/v2/vector"
)

// ===============================
// PARAMETERS
// ===============================
const (
	screenW  = 1200
	screenH  = 800
	numParts = 600 // total particles

	// Neighborhood radii
	neighborRadiusSame  = 22.0 // same-population neighbor search
	neighborRadiusCross = 18.0 // cross-population (interface) radius

	// LIFE-LIKE PRESET
	preferredSpacing = 14.0  // a bit tighter packing than default
	baseTension      = 0.028 // stronger membranes → smooth, rounded cells
	polygonBias      = 0.020 // pushes toward ~120° → honeycomb-ish, tissue-like
	noiseStrength    = 0.12  // gentle Brownian jiggle (not too twitchy)
	crossRepel       = 0.03  // different types can touch/mix like real tissues

	damping        = 0.95     // velocity damping
	particleRadius = 1.618033 // Fibonacci ratio radius
)

// ===============================
// PARTICLE STRUCT
// ===============================
type Particle struct {
	x, y   float64
	vx, vy float64
	typ    int // 0 or 1
	clr    color.Color
}

// Global state for particles
var particles []*Particle

// ===============================
// INIT PARTICLES
// ===============================
func newParticles() []*Particle {
	parts := make([]*Particle, numParts)
	for i := 0; i < numParts; i++ {
		pType := i % 2                                           // two populations
		pColor := color.RGBA{R: 0x66, G: 0xBB, B: 0x6A, A: 0xFF} // green-ish
		if pType == 1 {
			pColor = color.RGBA{R: 0x42, G: 0xA5, B: 0xF5, A: 0xFF} // blue-ish
		}

		parts[i] = &Particle{
			x:   rand.Float64() * screenW,
			y:   rand.Float64() * screenH,
			vx:  0,
			vy:  0,
			typ: pType,
			clr: pColor,
		}
	}
	return parts
}

// ===============================
// Ebiten Game Structure
// ===============================

type Game struct{}

func (g *Game) Update() error {
	for i, p := range particles {
		fx, fy := 0.0, 0.0

		// Interaction loop
		for j, q := range particles {
			if i == j {
				continue
			}

			dx := q.x - p.x
			dy := q.y - p.y
			distSq := dx*dx + dy*dy
			dist := math.Sqrt(distSq)

			if dist < 0.001 {
				continue
			}

			// normalize
			nx := dx / dist
			ny := dy / dist

			// same-type neighborhood
			if p.typ == q.typ && dist < neighborRadiusSame {
				// spring-like attraction/repulsion
				force := baseTension * (dist - preferredSpacing)

				// polygon bias → honeycomb-ish
				angleBias := math.Cos(3*math.Atan2(dy, dx)) * polygonBias
				force += angleBias

				fx += force * nx
				fy += force * ny
			}

			// cross-type interaction
			if p.typ != q.typ && dist < neighborRadiusCross {
				// gentle repulsion → keeps tissues separate but still mixing
				force := crossRepel * (neighborRadiusCross - dist)
				fx -= force * nx
				fy -= force * ny
			}
		}

		// Brownian / thermal jitter
		fx += (rand.Float64()*2 - 1) * noiseStrength
		fy += (rand.Float64()*2 - 1) * noiseStrength

		// integrate velocity
		p.vx = (p.vx + fx) * damping
		p.vy = (p.vy + fy) * damping

		// apply movement
		p.x += p.vx
		p.y += p.vy

		// wrap around boundaries
		if p.x < 0 {
			p.x += screenW
		}
		if p.x >= screenW {
			p.x -= screenW
		}
		if p.y < 0 {
			p.y += screenH
		}
		if p.y >= screenH {
			p.y -= screenH
		}
	}
	return nil
}

func (g *Game) Draw(screen *ebiten.Image) {
	// Background
	screen.Fill(color.RGBA{R: 0x00, G: 0x00, B: 0x00, A: 0xFF})

	// Draw particles
	for _, p := range particles {
		// vector.FillCircle is efficient for drawing many small circles
		vector.FillCircle(screen, float32(p.x), float32(p.y), float32(particleRadius), p.clr, true)
	}

	// Debug info
	ebitenutil.DebugPrint(screen, "Life-like Electro-Quantum Simulation (Ebiten)")

}

func (g *Game) Layout(outsideWidth, outsideHeight int) (int, int) {
	return screenW, screenH
}

// ===============================
// MAIN
// ===============================
func main() {

	particles = newParticles()

	ebiten.SetWindowSize(screenW, screenH)
	ebiten.SetWindowTitle("Life-like Electro-Quantum Simulation (Ebiten)")
	if err := ebiten.RunGame(&Game{}); err != nil {
		log.Fatal(err)
	}
}
