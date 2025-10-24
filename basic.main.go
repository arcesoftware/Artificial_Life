package main

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"

	"github.com/tfriedel6/canvas"
	"github.com/tfriedel6/canvas/sdlcanvas"
)

const (
	width          = 2700
	height         = 1000
	padding        = 67
	particleSize   = 5
	localRadius    = 30.0
	massBase       = 1.0
	energyStart    = 100.0
	energyCostMove = 0.001
	predationRange = 10.0
	predationRate  = 1.5
	maxForceDist   = 80.0
	damping        = 0.5
)

// Particle structure with essential properties and pre-allocated force fields
type particle struct {
	x, y   float64
	vx, vy float64
	fx, fy float64
	color  string
	mass   float64
	energy float64
}

var (
	particles []*particle
	cv        *canvas.Canvas
	wg        sync.WaitGroup
	yellow    []*particle
	red       []*particle
	green     []*particle
)

func init() {
	rand.Seed(time.Now().UnixNano())
}

func randomX() float64 { return (rand.Float64() * (width - padding*2)) + padding }
func randomY() float64 { return (rand.Float64() * (height - padding*2)) + padding }

// Convert alpha to a two-digit hex string
func toHexAlpha(alpha float64) string {
	a := math.Max(0.0, math.Min(1.0, alpha))
	alphaByte := byte(a * 255)
	return fmt.Sprintf("%02X", alphaByte)
}

// Draw particle with energy-based transparency
func draw(p *particle) {
	alpha := math.Max(0.1, p.energy/energyStart)
	colorWithAlpha := p.color + toHexAlpha(alpha)
	cv.SetFillStyle(colorWithAlpha)
	cv.FillRect(p.x, p.y, particleSize, particleSize)
}

// Create a batch of particles efficiently
func create(number int, color string, m float64) {
	ps := make([]*particle, number)
	for i := range ps {
		ps[i] = &particle{
			x:      randomX(),
			y:      randomY(),
			color:  color,
			mass:   m,
			energy: energyStart,
		}
	}
	particles = append(particles, ps...)
	switch color {
	case "#FFFF00":
		yellow = append(yellow, ps...)
	case "#FF0000":
		red = append(red, ps...)
	case "#00FF00":
		green = append(green, ps...)
	}
}

// Apply interaction rules concurrently without data races
func rule(particles1, particles2 []*particle, g float64) {
	n1 := len(particles1)
	if n1 == 0 || len(particles2) == 0 {
		return
	}
	wg.Add(n1)

	isSelf := &particles1[0] == &particles2[0] && len(particles1) == len(particles2)

	for i := 0; i < n1; i++ {
		go func(i int) {
			defer wg.Done()
			a := particles1[i]
			startJ := 0
			if isSelf {
				startJ = i + 1
			}
			for j := startJ; j < len(particles2); j++ {
				b := particles2[j]
				if !isSelf && a == b {
					continue
				}
				dx, dy := a.x-b.x, a.y-b.y
				distSq := dx*dx + dy*dy
				if distSq == 0 || distSq > maxForceDist*maxForceDist {
					continue
				}
				d := math.Sqrt(distSq)
				F := g * b.mass / d
				fx := F * dx
				fy := F * dy
				a.fx += fx
				a.fy += fy
				if isSelf {
					b.fx -= fx
					b.fy -= fy
				}

				// Simple predation
				if a.color == "#FF0000" && b.color == "#00FF00" && d < predationRange {
					transfer := predationRate / b.mass
					a.energy = math.Min(energyStart, a.energy+transfer)
					b.energy -= transfer
				}
			}
		}(i)
	}
	wg.Wait()
}

// Update particle physics, positions, and lifecycle
func updateWorld() {
	// Apply accumulated forces and damping
	for _, p := range particles {
		p.vx = (p.vx + p.fx) * damping
		p.vy = (p.vy + p.fy) * damping
		p.fx, p.fy = 0, 0
	}

	// Local velocity redistribution (mass-conserving)
	for i, a := range particles {
		for j := i + 1; j < len(particles); j++ {
			b := particles[j]
			dx, dy := b.x-a.x, b.y-a.y
			distSq := dx*dx + dy*dy
			if distSq < localRadius*localRadius && distSq > 0 {
				fx := (a.vx - b.vx) * 0.5
				fy := (a.vy - b.vy) * 0.5
				a.vx -= fx
				a.vy -= fy
				b.vx += fx
				b.vy += fy
			}
		}
	}

	// Update positions and handle energy and boundaries
	living := make([]*particle, 0, len(particles))
	var newYellow, newRed, newGreen []*particle

	for _, p := range particles {
		p.x += p.vx
		p.y += p.vy

		speed := math.Sqrt(p.vx*p.vx + p.vy*p.vy)
		p.energy -= energyCostMove * speed

		if p.x < 0 || p.x > float64(width-particleSize) {
			p.vx *= -1
			p.x = math.Max(0, math.Min(p.x, float64(width-particleSize)))
		}
		if p.y < 0 || p.y > float64(height-particleSize) {
			p.vy *= -1
			p.y = math.Max(0, math.Min(p.y, float64(height-particleSize)))
		}

		if p.energy > 0 {
			living = append(living, p)
			switch p.color {
			case "#FFFF00":
				newYellow = append(newYellow, p)
			case "#FF0000":
				newRed = append(newRed, p)
			case "#00FF00":
				newGreen = append(newGreen, p)
			}
		}
	}
	particles, yellow, red, green = living, newYellow, newRed, newGreen
}

func main() {
	wnd, canvasObj, err := sdlcanvas.CreateWindow(width, height, "Artificial Life - Optimized (No Fonts)")
	if err != nil {
		panic(err)
	}
	cv = canvasObj

	create(2000, "#FFFF00", massBase*0.8)
	create(1000, "#FF0000", massBase*1.2)
	create(2000, "#00FF00", massBase*1.0)

	wnd.MainLoop(func() {
		cv.SetFillStyle("#000")
		cv.FillRect(0, 0, float64(width), float64(height))

		// Display simple counts (no fonts)
		fmt.Printf("\rYellow: %d | Red: %d | Green: %d", len(yellow), len(red), len(green))

		// Interaction rules
		rule(green, green, -0.32)
		rule(red, red, -0.1)
		rule(yellow, yellow, 0.15)
		rule(green, red, -0.17)
		rule(green, yellow, 0.34)
		rule(red, green, -0.34)
		rule(yellow, green, -0.20)

		updateWorld()

		for _, p := range particles {
			draw(p)
		}
	})
}
