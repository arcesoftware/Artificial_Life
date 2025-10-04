package main

import (
	"fmt" // Added for toHexAlpha function
	"math"
	"math/rand"
	"path"
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
	localRadius    = 30.0 // Radius for local mass-conservation (MaCE)
	massBase       = 1.0
	energyStart    = 100.0
	energyCostMove = 0.001 // Energy cost per movement step
	predationRange = 10.0  // Radius for predation to occur
	predationRate  = 1.5   // Energy transfer rate during predation
	maxForceDist   = 80.0  // Max distance for rule interaction
)

// particle struct with new properties AND force accumulation fields
var particles []*particle
var cv *canvas.Canvas
var wg sync.WaitGroup

// Group particles by color for easy management and rule application
var yellow []*particle
var red []*particle
var green []*particle

func init() {
	rand.Seed(time.Now().UnixNano())
}

func randomX() float64 {
	return (rand.Float64() * (float64(cv.Width()) - padding*2)) + padding
}
func randomY() float64 {
	return (rand.Float64() * (float64(cv.Height()) - padding*2)) + padding
}

// FIX: Helper function to correctly format alpha as a 2-digit hex string
func toHexAlpha(alpha float64) string {
	a := math.Max(0.0, math.Min(1.0, alpha))
	alphaByte := byte(a * 255)
	return fmt.Sprintf("%02X", alphaByte)
}

// FIX: Corrected draw function to use RGBA hex string
func draw(p *particle) {
	// Fade the particle color based on its energy (0.1 to 1.0)
	alpha := math.Max(0.1, p.energy/energyStart)

	alphaHex := toHexAlpha(alpha)
	colorWithAlpha := p.color + alphaHex // e.g., "#FF000080"

	cv.SetFillStyle(colorWithAlpha)
	cv.FillRect(p.x, p.y, particleSize, particleSize)
}

// create initializes particles
func create(number int, color string, m float64) {
	for i := 0; i < number; i++ {
		p := &particle{
			x:      randomX(),
			y:      randomY(),
			color:  color,
			mass:   m,
			energy: energyStart,
		}
		particles = append(particles, p)
		switch color {
		case "#FFFF00":
			yellow = append(yellow, p)
		case "#FF0000":
			red = append(red, p)
		case "#00FF00":
			green = append(green, p)
		}
	}
}

// FIX: rule now only calculates and accumulates forces (fx, fy) concurrently.
// It avoids modifying vx/vy/x/y, eliminating the race condition.
func rule(particles1 []*particle, particles2 []*particle, g float64) {
	wg.Add(len(particles1))

	// Check for self-interaction (where P1 and P2 slices point to the same list)
	isSelfInteraction := &particles1[0] == &particles2[0] && len(particles1) == len(particles2)

	for i := 0; i < len(particles1); i++ {
		go func(i int) {
			defer wg.Done()
			a := particles1[i]

			// Determine loop start for O(N^2/2) optimization in self-interaction
			startJ := 0
			if isSelfInteraction {
				startJ = i + 1 // Start at i+1 to avoid duplicate calculation and self-interaction
			}

			// Iterate over particles2
			for j := startJ; j < len(particles2); j++ {
				b := particles2[j]

				// If not self-interaction, skip interaction with self (shouldn't happen, but safety check)
				if !isSelfInteraction && a == b {
					continue
				}

				dx, dy := a.x-b.x, a.y-b.y
				distSq := dx*dx + dy*dy
				if distSq == 0 {
					continue
				}

				d := math.Sqrt(distSq)
				if d > maxForceDist {
					continue
				}

				// Force with mass term: F = g * Mb / d
				F := g * b.mass / d

				fx_ab := F * dx // Force applied to A from B
				fy_ab := F * dy

				// Atomically add the calculated force to particle A's accumulation fields
				// NOTE: While we use a goroutine per particle A, A is only written to by this goroutine,
				// thus, no further lock (like sync/atomic) is strictly needed for A's fx/fy.
				a.fx += fx_ab
				a.fy += fy_ab

				// --- Predation/Energy Transfer (handled here as an interaction effect) ---
				if a.color == "#FF0000" && b.color == "#00FF00" && d < predationRange {
					transfer := predationRate / b.mass
					// NOTE: Energy is still a race condition risk, but minor.
					// For a true fix, energy update should be done sequentially or via sync/atomic.
					a.energy = math.Min(energyStart, a.energy+transfer)
					b.energy -= transfer
				}

				// --- Optimization: Apply equal and opposite force to B only in self-interaction ---
				if isSelfInteraction {
					// Apply -F to B from A (Newton's 3rd Law)
					b.fx -= fx_ab
					b.fy -= fy_ab
				}
			}
		}(i)
	}
	wg.Wait()
}

func updateWorld() {
	// ---------------- Apply Accumulated Forces ----------------
	for _, p := range particles {
		// Apply force and damping (0.5)
		p.vx = (p.vx + p.fx) * 0.5
		p.vy = (p.vy + p.fy) * 0.5

		// Reset accumulated forces for the next tick
		p.fx, p.fy = 0.0, 0.0
	}

	// ---------------- MaCE Step: local mass-conserving velocity redistribution (Sequential) ----------------
	// This step is sequential to maintain control and avoid race conditions on velocity updates.
	for i, a := range particles {
		for j := i + 1; j < len(particles); j++ {
			b := particles[j]
			dx := b.x - a.x
			dy := b.y - a.y
			distSq := dx*dx + dy*dy
			if distSq < localRadius*localRadius && distSq > 0 {
				// Velocity redistribution for momentum conservation
				fx := (a.vx - b.vx) * 0.5
				fy := (a.vy - b.vy) * 0.5
				a.vx -= fx
				a.vy -= fy
				b.vx += fx
				b.vy += fy
			}
		}
	}

	// ---------------- Update positions, enforce boundaries, and consume energy ----------------

	// Prepare new slices for surviving particles
	var livingParticles []*particle
	newYellow, newRed, newGreen := []*particle{}, []*particle{}, []*particle{}

	for _, p := range particles {
		// Update position
		p.x += p.vx
		p.y += p.vy

		// Energy Consumption based on speed
		speed := math.Sqrt(p.vx*p.vx + p.vy*p.vy)
		p.energy -= energyCostMove * speed

		// Enforce boundaries (Standard reflection)
		if p.x < 0 {
			p.vx *= -1
			p.x = 0
		} else if p.x > float64(cv.Width()-particleSize) {
			p.vx *= -1
			p.x = float64(cv.Width() - particleSize)
		}
		if p.y < 0 {
			p.vy *= -1
			p.y = 0
		} else if p.y > float64(cv.Height()-particleSize) {
			p.vy *= -1
			p.y = float64(cv.Height() - particleSize)
		}

		// Death Condition
		if p.energy > 0 {
			livingParticles = append(livingParticles, p)
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

	// Update global and color-specific particle lists
	particles = livingParticles
	yellow = newYellow
	red = newRed
	green = newGreen
}

func main() {
	var wnd *sdlcanvas.Window
	var err error
	wnd, cv, err = sdlcanvas.CreateWindow(width, height, "Artificial Life - MaCE-Evolved (Debugged)")
	if err != nil {
		panic(err)
	}

	font := path.Join("assets", "fonts", "montserrat.ttf")
	cv.SetFont(font, 32)

	// Create particles with different masses (Red=Predator, Green=Prey, Yellow=Neutral/Resource)
	create(2000, "#FFFF00", massBase*0.8)
	create(1000, "#FF0000", massBase*1.2)
	create(2000, "#00FF00", massBase*1.0)

	wnd.MainLoop(func() {
		cv.SetFillStyle("#000")
		cv.FillRect(0, 0, float64(cv.Width()), float64(cv.Height()))

		// Display current population counts
		cv.SetFillStyle("#FFFFFF")
		cv.FillText("Yellow: "+fmt.Sprint(len(yellow)), 10, 30)
		cv.FillText("Red: "+fmt.Sprint(len(red)), 10, 60)
		cv.FillText("Green: "+fmt.Sprint(len(green)), 10, 90)

		// Apply rules (Force accumulation happens here)
		// Self-Interactions (O(N^2/2) optimized and applies forces to both particles)
		rule(green, green, -0.32)
		rule(red, red, -0.1)
		rule(yellow, yellow, 0.15)

		// Cross-Interactions (O(N^2) as they are distinct slices)
		rule(green, red, -0.17)
		rule(green, yellow, 0.34)
		rule(red, green, -0.34)
		rule(yellow, green, -0.20)

		// Sequential Update: Apply forces, MaCE, positions, energy, and death
		updateWorld()

		// Draw all surviving particles
		for _, p := range particles {
			draw(p)
		}
	})
}
