package main

import (
	"math"
	"math/rand"
	"path"
	"sync"

	"github.com/tfriedel6/canvas"
	"github.com/tfriedel6/canvas/sdlcanvas"
)

const (
	width        = 1700
	height       = 1000
	padding      = 67
	particleSize = 5
	localRadius  = 30.0 // Radius for local mass-conservation
)

var particles []*particle
var cv *canvas.Canvas
var wg sync.WaitGroup

func randomX() float64 {
	return (rand.Float64() * (float64(cv.Width()) - padding*2)) + padding
}
func randomY() float64 {
	return (rand.Float64() * (float64(cv.Height()) - padding*2)) + padding
}

func draw(p *particle) {
	cv.SetFillStyle(p.color)
	cv.FillRect(p.x, p.y, particleSize, particleSize)
}

func create(number int, color string) []particle {
	group := make([]particle, number)
	for i := 0; i < number; i++ {
		group[i] = particle{x: randomX(), y: randomY(), color: color}
		particles = append(particles, &group[i])
	}
	return group
}

// ---------------- Interaction with MaCE-inspired mass conservation ----------------
func rule(particles1 []particle, particles2 []particle, g float64) {
	wg.Add(len(particles1))
	for i := 0; i < len(particles1); i++ {
		go func(i int) {
			defer wg.Done()
			a := &particles1[i]
			fx, fy := 0.0, 0.0
			for j := 0; j < len(particles2); j++ {
				b := &particles2[j]
				dx, dy := a.x-b.x, a.y-b.y
				if dx == 0 && dy == 0 {
					continue
				}
				d := math.Sqrt(dx*dx + dy*dy)
				if d > 80 {
					continue
				}
				F := g / d
				fx += F * dx
				fy += F * dy
			}
			a.vx = (a.vx + fx) * 0.5
			a.vy = (a.vy + fy) * 0.5
		}(i)
	}
	wg.Wait()

	// ---------------- MaCE Step: local mass-conserving velocity redistribution ----------------
	for i, a := range particles {
		for j := i + 1; j < len(particles); j++ {
			b := particles[j]
			dx := b.x - a.x
			dy := b.y - a.y
			distSq := dx*dx + dy*dy
			if distSq < localRadius*localRadius && distSq > 0 {
				// Redistribute velocities to conserve local momentum
				fx := (a.vx - b.vx) * 0.5
				fy := (a.vy - b.vy) * 0.5
				a.vx -= fx
				a.vy -= fy
				b.vx += fx
				b.vy += fy
			}
		}
	}

	// ---------------- Update positions and enforce boundaries ----------------
	for _, p := range particles {
		p.x += p.vx
		p.y += p.vy

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
	}
}

func main() {
	var wnd *sdlcanvas.Window
	var err error
	wnd, cv, err = sdlcanvas.CreateWindow(width, height, "Artificial Life - MaCE")
	if err != nil {
		panic(err)
	}

	font := path.Join("assets", "fonts", "montserrat.ttf")
	cv.SetFont(font, 32)

	yellow := create(2000, "#FFFF00")
	red := create(1000, "#FF0000")
	green := create(2000, "#00FF00")

	wnd.MainLoop(func() {
		cv.SetFillStyle("#000")
		cv.FillRect(0, 0, float64(cv.Width()), float64(cv.Height()))

		rule(green, green, -0.32)
		rule(green, red, -0.17)
		rule(green, yellow, 0.34)
		rule(red, red, -0.1)
		rule(red, green, -0.34)
		rule(yellow, yellow, 0.15)
		rule(yellow, green, -0.20)

		for _, p := range particles {
			draw(p)
		}
	})
}
