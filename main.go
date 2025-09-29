// lenia_evolve.go
package main

import (
	"fmt"
	"image/color"
	"log"
	"math"
	"math/rand"
	"sort"
	"time"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/text"
	"golang.org/x/image/font/basicfont"
)

// ---------- Simulation parameters (tweak these) ----------
const (
	gridW        = 200  // lattice width
	gridH        = 120  // lattice height
	cellSize     = 4    // display pixel size for each lattice cell
	evalSteps    = 120  // simulation steps per genome evaluation (short)
	populationSz = 12   // evolutionary population size
	elitism      = 3    // keep top N as-is
	mutationRate = 0.15 // per-parameter mutation probability
)

// ---------- Types ----------
type KernelEntry struct {
	dx, dy int
	w      float64
}

type Genome struct {
	Mu         float64 // μ
	Sigma      float64 // σ
	Radius     float64 // R
	ShellSigma float64 // shell shape
	Dt         float64 // Δt
	ColorBias  float64 // shift color mapping influence [-0.5,0.5]
	Fitness    float64 // cached after evaluation
}

type Game struct {
	A       [][]float64
	Anext   [][]float64
	kernel  []KernelEntry
	Knorm   float64
	texture *ebiten.Image

	// runtime
	generation      int
	population      []Genome
	currentIndex    int
	stepCount       int
	autoEvolve      bool
	autoEvolveDelay time.Duration
	lastEvolveTime  time.Time

	// visualization
	frame   int
	start   time.Time
	lastFPS int
}

// ---------- Utility ----------
func clamp(v, a, b float64) float64 {
	if v < a {
		return a
	}
	if v > b {
		return b
	}
	return v
}
func wrap(x, m int) int {
	if x >= 0 {
		return x % m
	}
	return (x%m + m) % m
}

// ---------- Kernel generation ----------
func buildKernel(R float64, shellSigma float64) ([]KernelEntry, float64) {
	var entries []KernelEntry
	if R <= 0 {
		R = 1
	}
	if shellSigma <= 0 {
		shellSigma = 0.15
	}
	Kc := func(rNorm float64) float64 {
		x := (rNorm - 0.5) / shellSigma
		return math.Exp(-0.5 * x * x)
	}
	Ri := int(math.Ceil(R))
	var sum float64
	for dy := -Ri; dy <= Ri; dy++ {
		for dx := -Ri; dx <= Ri; dx++ {
			dfx := float64(dx)
			dfy := float64(dy)
			dist := math.Hypot(dfx, dfy)
			if dist <= R {
				rnorm := dist / R
				weight := Kc(rnorm)
				entries = append(entries, KernelEntry{dx: dx, dy: dy, w: weight})
				sum += weight
			}
		}
	}
	if sum == 0 {
		sum = 1
	}
	for i := range entries {
		entries[i].w /= sum
	}
	return entries, 1.0
}

// ---------- Growth mapping ----------
func growth(u, mu, sigma float64) float64 {
	if sigma <= 0 {
		return 0
	}
	val := 2*math.Exp(-((u-mu)*(u-mu))/(2*sigma*sigma)) - 1
	if val > 1 {
		val = 1
	} else if val < -1 {
		val = -1
	}
	return val
}

// ---------- Initialize ----------
func NewGame() *Game {
	rand.Seed(time.Now().UnixNano())

	A := make([][]float64, gridH)
	Anext := make([][]float64, gridH)
	for y := 0; y < gridH; y++ {
		A[y] = make([]float64, gridW)
		Anext[y] = make([]float64, gridW)
	}

	g := &Game{
		A:               A,
		Anext:           Anext,
		texture:         ebiten.NewImage(gridW, gridH),
		generation:      0,
		currentIndex:    0,
		stepCount:       0,
		autoEvolve:      false,
		autoEvolveDelay: 3 * time.Second,
		lastEvolveTime:  time.Now(),
		start:           time.Now(),
	}

	// initialize random population
	g.population = make([]Genome, populationSz)
	for i := 0; i < populationSz; i++ {
		g.population[i] = randomGenome()
	}
	// prepare kernel for first genome
	g.applyGenomeKernel(&g.population[0])
	// seed grid for first genome
	g.seedFromGenome(&g.population[0])
	return g
}

func randomGenome() Genome {
	return Genome{
		Mu:         0.18 + rand.Float64()*0.5,  // 0.18..0.68
		Sigma:      0.02 + rand.Float64()*0.18, // 0.02..0.2
		Radius:     3.0 + rand.Float64()*8.0,   // 3..11
		ShellSigma: 0.08 + rand.Float64()*0.3,  // 0.08..0.38
		Dt:         0.03 + rand.Float64()*0.12, // 0.03..0.15
		ColorBias:  rand.Float64()*1.0 - 0.5,   // -0.5..0.5
	}
}

func (g *Game) applyGenomeKernel(gen *Genome) {
	k, kn := buildKernel(gen.Radius, gen.ShellSigma)
	g.kernel = k
	g.Knorm = kn
}

// seed grid with a blob pattern influenced by genome (variation between genomes)
func (g *Game) seedFromGenome(gen *Genome) {
	cx, cy := gridW/2, gridH/2
	// clear grid
	for y := 0; y < gridH; y++ {
		for x := 0; x < gridW; x++ {
			g.A[y][x] = 0
			g.Anext[y][x] = 0
		}
	}
	// make center blob size proportional to radius
	base := int(math.Max(6, gen.Radius*1.5))
	for y := 0; y < gridH; y++ {
		for x := 0; x < gridW; x++ {
			d := math.Hypot(float64(x-cx), float64(y-cy))
			if d < float64(base) {
				g.A[y][x] = 0.6 * math.Exp(-d*d/(2*float64(base)*float64(base)))
			}
			// sprinkle genome-specific noise
			if rand.Float64() < 0.002+0.001*rand.Float64() {
				g.A[y][x] = rand.Float64()*0.8 + 0.05
			}
		}
	}
}

// ---------- Single step ----------
func (g *Game) step(gen *Genome) {
	for y := 0; y < gridH; y++ {
		for x := 0; x < gridW; x++ {
			var u float64
			for _, k := range g.kernel {
				nx := wrap(x+k.dx, gridW)
				ny := wrap(y+k.dy, gridH)
				u += k.w * g.A[ny][nx]
			}
			grow := growth(u, gen.Mu, gen.Sigma)
			val := g.A[y][x] + gen.Dt*grow
			g.Anext[y][x] = clamp(val, 0.0, 1.0)
		}
	}
	g.A, g.Anext = g.Anext, g.A
}

// ---------- Fitness evaluation ----------
func (g *Game) evaluateGenome(gen *Genome) float64 {
	// seed and apply kernel
	g.applyGenomeKernel(gen)
	g.seedFromGenome(gen)

	// simulate for a short period and collect stats
	var activitySum float64
	var varianceSum float64
	var edgeSum float64

	for step := 0; step < evalSteps; step++ {
		g.step(gen)
		// compute stats each few steps
		if step%4 == 0 {
			mean := 0.0
			for y := 0; y < gridH; y++ {
				for x := 0; x < gridW; x++ {
					mean += g.A[y][x]
				}
			}
			mean /= float64(gridW * gridH)
			variance := 0.0
			edge := 0.0
			for y := 0; y < gridH; y++ {
				for x := 0; x < gridW; x++ {
					v := g.A[y][x]
					variance += (v - mean) * (v - mean)
					// simple edge metric: gradient magnitude
					r := g.A[y][wrap(x+1, gridW)] - v
					b := g.A[wrap(y+1, gridH)][x] - v
					edge += math.Abs(r) + math.Abs(b)
				}
			}
			variance /= float64(gridW * gridH)
			edge /= float64(gridW * gridH)
			activity := mean
			activitySum += activity
			varianceSum += variance
			edgeSum += edge
		}
	}

	// combine metrics into a fitness score
	// prefer moderate mean activity (not all-zero, not full), high variance (texture), and decent edges (structure)
	meanActivity := activitySum / float64(evalSteps/4)
	meanVar := varianceSum / float64(evalSteps/4)
	meanEdge := edgeSum / float64(evalSteps/4)

	// reward mid activity (bell around 0.25)
	actScore := math.Exp(-math.Pow((meanActivity-0.25)/0.12, 2))
	// scale variance and edge with diminishing returns
	varScore := math.Log(1 + meanVar*100)
	edgeScore := math.Log(1 + meanEdge*50)

	score := 1.2*actScore + 0.9*varScore + 0.8*edgeScore
	// small penalty for extreme radius or tiny sigma (to avoid degenerate)
	score *= 1.0 - 0.05*math.Abs(gen.Radius-6.0)/6.0
	if score < 0 {
		score = 0
	}
	return score
}

// ---------- Evolutionary operators ----------
func crossover(a, b Genome) Genome {
	child := Genome{
		Mu:         a.Mu,
		Sigma:      b.Sigma,
		Radius:     (a.Radius + b.Radius) * 0.5,
		ShellSigma: (a.ShellSigma + b.ShellSigma) * 0.5,
		Dt:         (a.Dt + b.Dt) * 0.5,
		ColorBias:  (a.ColorBias + b.ColorBias) * 0.5,
	}
	// mix some params randomly
	if rand.Float64() < 0.5 {
		child.Mu = b.Mu
	}
	if rand.Float64() < 0.5 {
		child.Sigma = a.Sigma
	}
	return child
}
func mutate(g *Genome) {
	if rand.Float64() < mutationRate {
		g.Mu += rand.NormFloat64() * 0.03
		g.Mu = clamp(g.Mu, 0.01, 1.0)
	}
	if rand.Float64() < mutationRate {
		g.Sigma += rand.NormFloat64() * 0.01
		g.Sigma = clamp(g.Sigma, 0.005, 0.5)
	}
	if rand.Float64() < mutationRate {
		g.Radius += rand.NormFloat64() * 1.2
		g.Radius = clamp(g.Radius, 1.5, 18.0)
	}
	if rand.Float64() < mutationRate {
		g.ShellSigma += rand.NormFloat64() * 0.05
		g.ShellSigma = clamp(g.ShellSigma, 0.02, 0.6)
	}
	if rand.Float64() < mutationRate {
		g.Dt += rand.NormFloat64() * 0.02
		g.Dt = clamp(g.Dt, 0.005, 0.5)
	}
	if rand.Float64() < mutationRate {
		g.ColorBias += rand.NormFloat64() * 0.12
		g.ColorBias = clamp(g.ColorBias, -1.0, 1.0)
	}
}

// ---------- Keyboard and update ----------
func (g *Game) Update() error {
	// toggle auto-evolve
	if ebiten.IsKeyPressed(ebiten.KeySpace) {
		// debounce by time
		if time.Since(g.lastEvolveTime) > 300*time.Millisecond {
			g.autoEvolve = !g.autoEvolve
			g.lastEvolveTime = time.Now()
		}
	}
	// manual evolve (generate next pop)
	if ebiten.IsKeyPressed(ebiten.KeyG) {
		if time.Since(g.lastEvolveTime) > 300*time.Millisecond {
			g.evolveOnce()
			g.lastEvolveTime = time.Now()
		}
	}
	// switch genome being displayed
	if ebiten.IsKeyPressed(ebiten.KeyRight) {
		if time.Since(g.lastEvolveTime) > 200*time.Millisecond {
			g.currentIndex = (g.currentIndex + 1) % len(g.population)
			g.applyGenomeKernel(&g.population[g.currentIndex])
			g.seedFromGenome(&g.population[g.currentIndex])
			g.stepCount = 0
			g.lastEvolveTime = time.Now()
		}
	}
	if ebiten.IsKeyPressed(ebiten.KeyLeft) {
		if time.Since(g.lastEvolveTime) > 200*time.Millisecond {
			g.currentIndex = (g.currentIndex - 1 + len(g.population)) % len(g.population)
			g.applyGenomeKernel(&g.population[g.currentIndex])
			g.seedFromGenome(&g.population[g.currentIndex])
			g.stepCount = 0
			g.lastEvolveTime = time.Now()
		}
	}

	// auto-evolve
	if g.autoEvolve && time.Since(g.lastEvolveTime) > g.autoEvolveDelay {
		g.evolveOnce()
		g.lastEvolveTime = time.Now()
	}

	// run one simulation step for the displayed genome
	cur := &g.population[g.currentIndex]
	g.step(cur)
	g.stepCount++
	g.frame++
	if g.frame%30 == 0 {
		elapsed := time.Since(g.start).Seconds()
		if elapsed > 0 {
			g.lastFPS = int(float64(g.frame) / elapsed)
		}
	}
	return nil
}

// ---------- Evolution procedure ----------
func (g *Game) evolveOnce() {
	// evaluate all genomes
	for i := range g.population {
		score := g.evaluateGenome(&g.population[i])
		g.population[i].Fitness = score
	}
	// sort by fitness desc
	sort.Slice(g.population, func(i, j int) bool {
		return g.population[i].Fitness > g.population[j].Fitness
	})

	// keep some elites
	newPop := make([]Genome, 0, populationSz)
	for i := 0; i < elitism && i < len(g.population); i++ {
		newPop = append(newPop, g.population[i])
	}

	// fill rest with crossover+mutate
	for len(newPop) < populationSz {
		// tournament selection
		a := tournamentSelect(g.population)
		b := tournamentSelect(g.population)
		child := crossover(a, b)
		mutate(&child)
		newPop = append(newPop, child)
	}

	g.population = newPop
	g.generation++
	// reset viewer to best genome
	g.currentIndex = 0
	g.applyGenomeKernel(&g.population[0])
	g.seedFromGenome(&g.population[0])
	g.stepCount = 0
}

// tournament selection (size 3)
func tournamentSelect(pop []Genome) Genome {
	best := pop[rand.Intn(len(pop))]
	for i := 0; i < 2; i++ {
		cand := pop[rand.Intn(len(pop))]
		if cand.Fitness > best.Fitness {
			best = cand
		}
	}
	return best
}

// ---------- Draw / display ----------
func (g *Game) Draw(screen *ebiten.Image) {
	// map A -> texture using genome color bias
	bias := g.population[g.currentIndex].ColorBias
	for y := 0; y < gridH; y++ {
		for x := 0; x < gridW; x++ {
			v := clamp(g.A[y][x]+bias*0.08, 0, 1)
			r, gg, b := colorRamp(v)
			g.texture.Set(x, y, color.NRGBA{R: r, G: gg, B: b, A: 0xFF})
		}
	}

	op := &ebiten.DrawImageOptions{}
	op.GeoM.Scale(float64(cellSize), float64(cellSize))
	op.Filter = ebiten.FilterNearest
	screen.DrawImage(g.texture, op)

	// overlay info
	cur := &g.population[g.currentIndex]
	txt := fmt.Sprintf("Gen: %d  Index: %d/%d  Fitness(best): %.3f  μ:%.3f σ:%.3f R:%.2f shell:%.2f Δt:%.3f",
		g.generation, g.currentIndex, len(g.population), g.population[0].Fitness, cur.Mu, cur.Sigma, cur.Radius, cur.ShellSigma, cur.Dt)
	text.Draw(screen, txt, basicfont.Face7x13, 6, 16, color.White)

	help := "Keys: ←/→ switch genome   G evolve once   SPACE toggle auto-evolve   (auto delay 3s)    FPS:"
	text.Draw(screen, help, basicfont.Face7x13, 6, 32, color.White)
	fps := fmt.Sprintf("%d", g.lastFPS)
	text.Draw(screen, fps, basicfont.Face7x13, 6, 48, color.White)
}

func (g *Game) Layout(outW, outH int) (int, int) {
	return gridW * cellSize, gridH * cellSize
}

// ---------- color ramp ----------
func colorRamp(v float64) (r, g, b uint8) {
	v = clamp(v, 0, 1)
	if v < 0.5 {
		t := v / 0.5
		return uint8(20 + 50*t), uint8(50 + 150*t), uint8(200 - 100*t)
	}
	t := (v - 0.5) / 0.5
	return uint8(70 + 180*t), uint8(200 - 80*t), uint8(100 + 150*t)
}

// ---------- main ----------
func main() {
	ebiten.SetWindowSize(gridW*cellSize, gridH*cellSize)
	ebiten.SetWindowTitle("Evolving Lenia-like Artificial Life (Ebiten)")

	game := NewGame()
	if err := ebiten.RunGame(game); err != nil {
		log.Fatal(err)
	}
}
