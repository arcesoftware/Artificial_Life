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
const padding = 44
const particleSize = 3.14159

var particles []*particle = make([]*particle, 0)
var cv *canvas.Canvas
var wg sync.WaitGroup

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

// --------- Drawing ---------
func draw(p *particle) {
	color := computeColor(p)
	cv.SetFillStyle(color)
	cv.FillRect(p.x, p.y, particleSize, particleSize)
}

// --------- Particle Creation ---------
func create(number int) []particle {
	group := make([]particle, number)
	for i := 0; i < number; i++ {
		group[i] = particle{x: randomX(), y: randomY()}
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
		rule(green, green, -1.618033)
		rule(green, red, 1.17)
		rule(green, yellow, 0.34)
		rule(red, red, 1.1)
		rule(red, green, 1.34)
		rule(yellow, yellow, 0.15)
		rule(yellow, green, 1.20)

		// draw all
		for _, p := range particles {
			draw(p)
		}

		elapsedTime := time.Since(startTime)
		printFps(elapsedTime)
	})
}
