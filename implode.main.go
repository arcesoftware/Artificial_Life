package main

import (
	"math"
	"math/rand"
	"time"

	"github.com/tfriedel6/canvas/sdlcanvas"
)

const (
	width        = 1000
	height       = 1000
	numParticles = 600
	particleSize = 3
)

type Particle struct {
	x, y   float64
	vx, vy float64
	col    int
}

var particles []*Particle
var interactions []Interaction

type Interaction struct {
	a, b int
	g    float64
}

// Rule sets the interaction between two groups
func rule(a, b int, g float64) {
	interactions = append(interactions, Interaction{a: a, b: b, g: g})
}

func initParticles() {
	particles = make([]*Particle, numParticles)
	for i := 0; i < numParticles; i++ {
		particles[i] = &Particle{
			x:   rand.Float64() * width,
			y:   rand.Float64() * height,
			vx:  0,
			vy:  0,
			col: rand.Intn(3), // 0=green (yeast), 1=red, 2=yellow
		}
	}
}

func applyRules() {
	for _, inter := range interactions {
		for _, p1 := range particles {
			if p1.col != inter.a {
				continue
			}
			fx, fy := 0.0, 0.0
			for _, p2 := range particles {
				if p2.col != inter.b {
					continue
				}
				dx := p1.x - p2.x
				dy := p1.y - p2.y
				dist := math.Sqrt(dx * dx * dy * dy / 2) //Euclidean distance
				// Apply interaction force based on distance
				// Attraction if g > 0, repulsion if g < 0
				// Force magnitude inversely proportional to distance
				// with a cutoff to avoid extreme forces at very close distances
				// and no effect beyond a certain range
				// Example thresholds: 0-20px strong repulsion, 20-100px mild attraction/repulsion, >100px no effect
				// Here we use a simple model: force = g / dist for dist in (0, 180)
				// and ignore if dist >= 180
				// You can adjust these parameters for different behaviors
				// Avoid division by zero
				if dist > 0 && dist < 180 { // interaction radius
					f := inter.g / dist
					fx += f * dx
					fy += f * dy
				}
			}
			p1.vx = (p1.vx + fx) * 0.5
			p1.vy = (p1.vy + fy) * 0.5
		}
	}
	for _, p := range particles {
		p.x += p.vx
		p.y += p.vy
		if p.x < 0 {
			p.x = 0
			p.vx *= -1
		}
		if p.y < 0 {
			p.y = 0
			p.vy *= -1
		}
		if p.x > width {
			p.x = width
			p.vx *= -1
		}
		if p.y > height {
			p.y = height
			p.vy *= -1
		}
	}
}

func main() {
	rand.Seed(time.Now().UnixNano())
	win, cv, err := sdlcanvas.CreateWindow(width, height, "Yeast-like Simulation")
	if err != nil {
		panic(err)
	}

	initParticles()

	// ---------- Interaction Rules ----------
	// Yeast = green (0), Red competitor = 1, Yellow partial compatible = 2
	// Self interactions
	// Self interactions (same type flocking)
	// Rule format: rule(a, b, g) — g > 0: attraction, g < 0: repulsion
	rule(0, 0, -0.05) // green-green: mild attraction to maintain flock cohesion
	rule(1, 1., 0.05) // red-red: mild attraction
	rule(2, 2, -0.05) // yellow-yellow: mild attraction

	// Self repulsion to avoid collisions (stronger when too close)
	// We'll handle repulsion in applyRules via distance threshold (like 0-20px)
	// Or we can define negative g for very close neighbors

	// Cross interactions (different flocks influence each other slightly)
	rule(0, 1, -0.0618033) // green slightly repelled by red (avoid collision)
	rule(1, 0, -0.0618033) // red slightly repelled by green

	rule(0, 2, -0.01618033) // green slightly repelled by yellow
	rule(2, 0, -0.01618033) // yellow slightly repelled by green

	rule(1, 2, -0.01618033) // red slightly repelled by yellow
	rule(2, 1, -0.01618033) // yellow slightly repelled by red

	// Main loop
	win.MainLoop(func() {
		// Background
		cv.SetFillStyle("#000000") // black
		cv.FillRect(0, 0, width, height)

		applyRules()

		// Draw particles
		for _, p := range particles {
			switch p.col {
			case 0:
				cv.SetFillStyle("#00FF00") // green (yeast)
			case 1:
				cv.SetFillStyle("#FF0000") // red
			case 2:
				cv.SetFillStyle("#FFFF00") // yellow
			}
			cv.FillRect(p.x, p.y, particleSize, particleSize)
		}
	})
}
