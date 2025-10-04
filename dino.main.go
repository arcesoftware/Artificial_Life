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
	numParticles = 400
	particleSize = 3

	maxSpeed         = 4.0
	perception       = 50.0 // radius to consider neighbors
	avoidRadius      = 15.0 // minimal distance to avoid collisions
	alignFactor      = 0.05 // velocity alignment strength
	cohesionFactor   = 0.01 // flock cohesion strength
	separationFactor = 0.15 // repulsion from too close neighbors
)

type Particle struct {
	x, y   float64
	vx, vy float64
	col    int
}

var particles []*Particle

func initParticles() {
	particles = make([]*Particle, numParticles)
	for i := 0; i < numParticles; i++ {
		angle := rand.Float64() * 2 * math.Pi
		speed := rand.Float64() * maxSpeed
		particles[i] = &Particle{
			x:   rand.Float64() * width,
			y:   rand.Float64() * height,
			vx:  math.Cos(angle) * speed,
			vy:  math.Sin(angle) * speed,
			col: rand.Intn(3), // optional color groups
		}
	}
}

// Apply flocking rules for murmuration
func applyFlocking() {
	for _, p := range particles {
		var (
			avgVx, avgVy     float64
			centerX, centerY float64
			count            int
			sepX, sepY       float64
		)

		for _, other := range particles {
			if p == other {
				continue
			}
			dx := other.x - p.x
			dy := other.y - p.y
			dist := math.Sqrt(dx*dx + dy*dy)

			if dist < perception {
				// Alignment
				avgVx += other.vx
				avgVy += other.vy

				// Cohesion (move toward center)
				centerX += other.x
				centerY += other.y

				count++

				// Separation (avoid collisions)
				if dist < avoidRadius && dist > 0 {
					sepX -= (other.x - p.x) / dist
					sepY -= (other.y - p.y) / dist
				}
			}
		}

		if count > 0 {
			// Alignment
			avgVx /= float64(count)
			avgVy /= float64(count)
			p.vx += (avgVx - p.vx) * alignFactor
			p.vy += (avgVy - p.vy) * alignFactor

			// Cohesion
			centerX /= float64(count)
			centerY /= float64(count)
			p.vx += (centerX - p.x) * cohesionFactor
			p.vy += (centerY - p.y) * cohesionFactor

			// Separation
			p.vx += sepX * separationFactor
			p.vy += sepY * separationFactor
		}

		// Limit speed
		speed := math.Sqrt(p.vx*p.vx + p.vy*p.vy)
		if speed > maxSpeed {
			p.vx = (p.vx / speed) * maxSpeed
			p.vy = (p.vy / speed) * maxSpeed
		}
	}

	// Update positions
	for _, p := range particles {
		p.x += p.vx
		p.y += p.vy

		// Wrap around edges (torus behavior)
		if p.x < 0 {
			p.x += width
		}
		if p.x > width {
			p.x -= width
		}
		if p.y < 0 {
			p.y += height
		}
		if p.y > height {
			p.y -= height
		}
	}
}

func main() {
	rand.Seed(time.Now().UnixNano())
	win, cv, err := sdlcanvas.CreateWindow(width, height, "Murmuration Simulation")
	if err != nil {
		panic(err)
	}

	initParticles()

	win.MainLoop(func() {
		// Background
		cv.SetFillStyle("#000000")
		cv.FillRect(0, 0, width, height)

		applyFlocking()

		// Draw particles
		for _, p := range particles {
			switch p.col {
			case 0:
				cv.SetFillStyle("#00FF00") // green
			case 1:
				cv.SetFillStyle("#FF0000") // red
			case 2:
				cv.SetFillStyle("#FFFF00") // yellow
			}
			cv.FillRect(p.x, p.y, particleSize, particleSize)
		}
	})
}
