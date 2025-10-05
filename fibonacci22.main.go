package main

import (
	"fmt"
	"math"
	"math/rand"
	"path"
	"sync"
	"time"

	"github.com/tfriedel6/canvas"
	"github.com/tfriedel6/canvas/sdlcanvas"
)

const width = 1080
const height = 1440
const padding = 50
const particleSize = 3.14159

type particle struct {
	x     float64
	y     float64
	vx    float64
	vy    float64
	color string
}

var particles []*particle = make([]*particle, 0)
var cv *canvas.Canvas
var wg sync.WaitGroup

// Fibonacci returns the nth Fibonacci number
func Fibonacci(n int) int {
	if n <= 1 {
		return n
	}
	a, b := 0, 1
	for i := 2; i <= n; i++ {
		a, b = b, a+b
	}
	return b
}

// --------- Random Position Helpers ---------
func randomX() float64 {
	return (rand.Float64() * (float64(cv.Width()) - padding*2)) + padding
}
func randomY() float64 {
	return (rand.Float64() * (float64(cv.Height()) - padding*2)) + padding
}

// --------- HSV to HEX Conversion ---------
func hsvToHex(h, s, v float64) string {
	h = math.Mod(h, 360)
	c := v * s
	x := c * (1 - math.Abs(math.Mod(h/60.0, 2)-1))
	m := v - c
	var r, g, b float64

	switch {
	case 0 <= h && h < 60:
		r, g, b = c, x, 0
	case 60 <= h && h < 120:
		r, g, b = x, c, 0
	case 120 <= h && h < 180:
		r, g, b = 0, c, x
	case 180 <= h && h < 240:
		r, g, b = 0, x, c
	case 240 <= h && h < 300:
		r, g, b = x, 0, c
	default:
		r, g, b = c, 0, x
	}
	R := int((r + m) * 255)
	G := int((g + m) * 255)
	B := int((b + m) * 255)
	return fmt.Sprintf("#%02X%02X%02X", R, G, B)
}

// --------- Drawing ---------
func draw(p *particle) {
	color := computeColor(p)
	cv.SetFillStyle(color)
	cv.FillRect(p.x, p.y, particleSize, particleSize)
}

// --------- Dynamic Color (velocity + density) ---------
type cluster struct {
	particles []*particle
	hue       float64
}

// Returns cluster ID for each particle
func findClusters(particles []*particle, radius float64) []*cluster {
	visited := make([]bool, len(particles))
	var clusters []*cluster

	for i := 0; i < len(particles); i++ {
		if visited[i] {
			continue
		}
		queue := []int{i}
		visited[i] = true
		c := &cluster{particles: []*particle{}, hue: rand.Float64() * 360}
		for len(queue) > 0 {
			idx := queue[0]
			queue = queue[1:]
			c.particles = append(c.particles, particles[idx])
			for j := 0; j < len(particles); j++ {
				if visited[j] {
					continue
				}
				dx := particles[idx].x - particles[j].x
				dy := particles[idx].y - particles[j].y
				if dx*dx+dy*dy <= radius*radius {
					queue = append(queue, j)
					visited[j] = true
				}
			}
		}
		clusters = append(clusters, c)
	}
	return clusters
}

var clusters []*cluster

func computeColor(p *particle) string {
	// Find which cluster this particle belongs to
	var baseHue float64
	for _, c := range clusters {
		for _, q := range c.particles {
			if q == p {
				baseHue = c.hue
				break
			}
		}
	}

	// Modulate hue by velocity
	speed := math.Sqrt(p.vx*p.vx + p.vy*p.vy)
	maxSpeed := 5.0
	normSpeed := math.Min(speed/maxSpeed, 1.0)

	hue := math.Mod(baseHue+180*normSpeed, 360)
	return hsvToHex(hue, 1.0, 0.6+0.4*normSpeed)
}

// --------- Particle Creation ---------
func create(Fibonacci int) []particle {
	group := make([]particle, Fibonacci)
	for i := 0; i < Fibonacci; i++ {
		group[i] = particle{x: float64(Fibonacci) + randomX(), y: float64(Fibonacci) + randomY()}
		particles = append(particles, &group[i])
	}
	return group
}

// --------- Rule Application ---------
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
				F := g / (d)
				fx += (F * dx)
				fy += (F * dy)
			}
			a.vx = (a.vx + fx) * 0.5
			a.vy = (a.vy + fy) * 0.5
			a.x += a.vx
			a.y += a.vy
			if a.x <= 0 {
				a.vx *= -1
				a.x = 0
			} else if a.x >= float64(cv.Width())-particleSize {
				a.vx *= -1
				a.x = float64(cv.Width()) - particleSize
			}
			if a.y <= 0 {
				a.vy *= -1
				a.y = 0
			} else if a.y >= float64(cv.Height())-particleSize {
				a.vy *= -1
				a.y = float64(cv.Height()) - particleSize
			}
		}(i)
	}
	wg.Wait()
}

// --------- FPS Debug ---------
func printFps(elapsedTime time.Duration) {
	cv.SetFillStyle("#FFFFFF")
	cv.SetStrokeStyle("#000000")
	cv.SetLineWidth(2)
	fpsValue := int(1 / elapsedTime.Seconds())
	fpsText := fmt.Sprintf("FPS: %d", fpsValue)
	cv.FillText(fpsText, 5, 35)
	cv.StrokeText(fpsText, 5, 35)
}

// --------- MAIN ---------
func main() {
	var wnd *sdlcanvas.Window
	var err error
	wnd, cv, err = sdlcanvas.CreateWindow(width, height, "Artificial Life - Chromatic")
	if err != nil {
		panic(err)
	}

	font := path.Join("assets", "fonts", "montserrat.ttf")
	cv.SetFont(font, 32)

	// create particle groups (but no fixed color now)
	green := create(2000)
	red := create(1000)
	yellow := create(2000)
	clusters = findClusters(particles, 80)
	for _, p := range particles {
		draw(p)
	}

	wnd.MainLoop(func() {
		startTime := time.Now()
		w, h := float64(cv.Width()), float64(cv.Height())
		cv.SetFillStyle("#000")
		cv.FillRect(0, 0, w, h)

		// rules (still based on groups)
		// apply Fibonacci values, scaled down to stay stable
		fib5 := float64(Fibonacci(5)) // 5
		fib3 := float64(Fibonacci(3)) // 2
		fib1 := float64(Fibonacci(1)) // 1

		scale := 0.05 // reduces intensity so the forces remain in [-1,1] range

		rule(red, red, fib5*scale*-1)  // slight repulsion, prevents solid red cores
		rule(red, green, fib3*scale)   // mild attraction for interplay with green
		rule(red, yellow, -fib3*scale) // balanced repulsion/attraction for vortex flow

		rule(yellow, yellow, fib1*scale) // mild self-attraction forms yellow filaments
		rule(yellow, green, -0.20)       // gentle repulsion, stabilizes green-yellow interface
		rule(yellow, red, -0.15)         // light repulsion, avoids yellow-red clumping
		rule(green, green, -fib5*scale)  // strong self-repulsion creates green voids
		rule(green, red, -fib3*scale)
		rule(green, yellow, 0.20) // gentle attraction, stabilizes green-yellow interface

		clusters = findClusters(particles, 67)

		frameCount := 0
		idx := (frameCount / 60) % 10 // change every ~second
		f := float64(Fibonacci(idx))

		rule(red, red, -f*scale)
		rule(red, green, f*0.5*scale)
		rule(red, yellow, -f*0.4*scale)

		// draw all
		for _, p := range particles {
			draw(p)
		}

		elapsedTime := time.Since(startTime)
		printFps(elapsedTime)
	})
}
