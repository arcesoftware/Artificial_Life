// lenia_evolve_extended.go
package main

import (
	"fmt"
	"image/color"
	"log"
	"math"
	"math/cmplx"
	"math/rand"
	"sort"
	"time"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/text"
	"golang.org/x/image/font/basicfont"
)

// ---------- Simulation parameters (tweak these) ----------
const (
	gridW        = 400      // lattice width
	gridH        = 400      // lattice height
	cellSize     = 3        // display pixel size for each lattice cell
	evalSteps    = 120      // simulation steps per genome evaluation (short)
	populationSz = 6180     // evolutionary population size
	elitism      = 9        // keep top N as-is
	mutationRate = 0.618033 // per-parameter mutation probability

	// ANOMALY PARAMETERS - The seed of evil
	anomalyRadius    = 9.0      // Radius of the devouring influence (float for math)
	anomalyStrength  = 1.618033 // Amount of activity 'devoured' per step in the radius
	anomalyColorBias = -0.5     // Shift color mapping for anomaly zone visualization

	// ANOMALY MOVEMENT PARAMETERS (NEW)
	anomalySearchRadius = 37  // Radius to search for the nearest population peak
	anomalyMoveSpeed    = 1.0 // Distance the anomaly moves per step (in grid units)
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

type Lorenz struct {
	x, y, z float64
	sigma   float64
	rho     float64
	beta    float64
	dt      float64
}

func (l *Lorenz) Step() {
	// classic Lorenz attractor integration (RK4 could be used but simple Euler suffices here)
	dx := l.sigma * (l.y - l.x)
	dy := l.x*(l.rho-l.z) - l.y
	dz := l.x*l.y - l.beta*l.z
	l.x += dx * l.dt
	l.y += dy * l.dt
	l.z += dz * l.dt
}

type Game struct {
	A       [][]float64
	Anext   [][]float64
	kernel  []KernelEntry
	Knorm   float64
	texture *ebiten.Image

	// Anomaly State (NEW)
	anomalyX float64
	anomalyY float64
	lorenz   Lorenz

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

func wrapDiff(c1, c2, m int) int {
	diff := c2 - c1
	if math.Abs(float64(diff)) <= float64(m)/2 {
		return diff
	}
	if diff > 0 {
		return diff - m
	}
	return diff + m
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

// ---------- Fibonacci sequence generator ----------
// simple iterative fibonacci that returns nth Fibonacci number (uint64, safe up to ~93)
func fib(n int) uint64 {
	if n <= 0 {
		return 0
	}
	if n == 1 {
		return 1
	}
	var a, b uint64 = 0, 1
	for i := 2; i <= n; i++ {
		a, b = b, a+b
	}
	return b
}

// ---------- FFT (Cooley-Tukey, power-of-two) ----------
func nextPow2(n int) int {
	p := 1
	for p < n {
		p <<= 1
	}
	return p
}

func fft(a []complex128) []complex128 {
	n := len(a)
	if n == 1 {
		return []complex128{a[0]}
	}
	if n%2 != 0 {
		// pad to next pow2 (shouldn't reach here in recursion if we start with pow2)
		np := nextPow2(n)
		b := make([]complex128, np)
		copy(b, a)
		return fft(b)
	}
	// split
	even := make([]complex128, n/2)
	odd := make([]complex128, n/2)
	for i := 0; i < n/2; i++ {
		even[i] = a[2*i]
		odd[i] = a[2*i+1]
	}
	even = fft(even)
	odd = fft(odd)
	res := make([]complex128, n)
	for k := 0; k < n/2; k++ {
		t := cmplx.Exp(complex(0, -2*math.Pi*float64(k)/float64(n))) * odd[k]
		res[k] = even[k] + t
		res[k+n/2] = even[k] - t
	}
	return res
}

func ifft(A []complex128) []complex128 {
	n := len(A)
	conj := make([]complex128, n)
	for i := 0; i < n; i++ {
		conj[i] = cmplx.Conj(A[i])
	}
	y := fft(conj)
	res := make([]complex128, n)
	for i := 0; i < n; i++ {
		res[i] = cmplx.Conj(y[i]) / complex(float64(n), 0)
	}
	return res
}

// Apply a simple FFT-based filter across each row: low-pass with cutoff ratio
func (g *Game) applyFFTFilter(cutoffRatio float64) {
	// cutoffRatio between 0..1
	L := gridW
	np := nextPow2(L)
	buf := make([]complex128, np)
	for y := 0; y < gridH; y++ {
		// fill
		for x := 0; x < np; x++ {
			if x < L {
				buf[x] = complex(g.A[y][x], 0)
			} else {
				buf[x] = 0
			}
		}
		// fft
		F := fft(buf)
		// zero high freq
		cut := int(cutoffRatio * float64(len(F)) / 2.0)
		for k := cut; k < len(F)-cut; k++ {
			F[k] = 0
		}
		// inverse
		inv := ifft(F)
		// write back real parts (normalize/clamp)
		for x := 0; x < L; x++ {
			v := real(inv[x])
			g.A[y][x] = clamp(v, 0, 1)
		}
	}
}

// ---------- Five-dimensional mapping ----------
// Map genome parameters into a 5D normalized vector for downstream modulation
func map5D(gen *Genome) [5]float64 {
	// normalize each parameter into 0..1 ranges based on expected bounds
	mu := (gen.Mu - 0.01) / (1.0 - 0.01)
	sigma := (gen.Sigma - 0.005) / (0.5 - 0.005)
	radius := (gen.Radius - 1.5) / (18.0 - 1.5)
	shell := (gen.ShellSigma - 0.02) / (0.6 - 0.02)
	dt := (gen.Dt - 0.005) / (0.5 - 0.005)
	return [5]float64{clamp(mu, 0, 1), clamp(sigma, 0, 1), clamp(radius, 0, 1), clamp(shell, 0, 1), clamp(dt, 0, 1)}
}

// ---------- Modified: Implements the devouring effect using dynamic anomalyX/Y ----------
func (g *Game) applyAnomalyEffect() {
	ax, ay := g.anomalyX, g.anomalyY
	Ri := int(math.Ceil(anomalyRadius))
	for dy := -Ri; dy <= Ri; dy++ {
		for dx := -Ri; dx <= Ri; dx++ {
			x := wrap(int(ax)+dx, gridW)
			y := wrap(int(ay)+dy, gridH)
			// shortest toroidal distance approx
			dxF := float64(x) - ax
			dyF := float64(y) - ay
			if dxF > float64(gridW/2) {
				dxF -= float64(gridW)
			} else if dxF < float64(-gridW/2) {
				dxF += float64(gridW)
			}
			if dyF > float64(gridH/2) {
				dyF -= float64(gridH)
			} else if dyF < float64(-gridH/2) {
				dyF += float64(gridH)
			}
			dist := math.Hypot(dxF, dyF)
			if dist < anomalyRadius {
				factor := 1.0 - (dist / anomalyRadius)
				devour := anomalyStrength * factor
				g.A[y][x] = clamp(g.A[y][x]-devour, 0.0, 1.0)
			}
		}
	}
}

// NEW FUNCTION: Implements the shortest path movement towards highest local activity
// and also influenced by Lorenz attractor and Fibonacci-modulated speed
func (g *Game) findTargetAndMoveAnomaly() {
	ax, ay := int(g.anomalyX), int(g.anomalyY)
	searchRi := int(anomalySearchRadius)

	var maxActivity float64 = -1.0
	var targetX, targetY int = ax, ay

	for dy := -searchRi; dy <= searchRi; dy++ {
		for dx := -searchRi; dx <= searchRi; dx++ {
			x := wrap(ax+dx, gridW)
			y := wrap(ay+dy, gridH)
			activity := g.A[y][x]
			if activity > maxActivity {
				maxActivity = activity
				targetX = ax + dx
				targetY = ay + dy
			}
		}
	}

	// Lorenz influence: step and map to a small offset
	g.lorenz.Step()
	lorX := g.lorenz.x
	lorY := g.lorenz.y
	lorZ := g.lorenz.z

	// Fibonacci-modulated speed: pick an index from lorenz z
	fibIdx := 10 + int(math.Abs(lorZ))%20 // safe small index
	fibVal := float64(fib(fibIdx % 90))   // avoid overflow
	// normalize fibVal to a reasonable multiplier
	fibMul := 0.0005 * (1.0 + math.Mod(fibVal, 1000.0)/1000.0)

	// If a significant target is found
	if maxActivity > 0.01 {
		dx := float64(targetX) - g.anomalyX
		dy := float64(targetY) - g.anomalyY
		// adjust for toroidal shortest path
		if dx > float64(gridW/2) {
			dx -= float64(gridW)
		} else if dx < float64(-gridW/2) {
			dx += float64(gridW)
		}
		if dy > float64(gridH/2) {
			dy -= float64(gridH)
		} else if dy < float64(-gridH/2) {
			dy += float64(gridH)
		}
		dist := math.Hypot(dx, dy)
		if dist > 1e-6 {
			dx /= dist
			dy /= dist
		} else {
			return
		}

		// base speed influenced by anomalyMoveSpeed, Lorenz x/y, and fibMul
		speed := anomalyMoveSpeed*(1.0+0.2*lorX+0.2*lorY) + fibMul*float64(fibIdx)
		// small randomness
		speed += (rand.Float64() - 0.5) * 0.3

		// limit
		if speed < 0.1 {
			speed = 0.1
		}
		if speed > 8.0 {
			speed = 8.0
		}

		// move anomaly
		g.anomalyX += dx * speed
		g.anomalyY += dy * speed

		// also add a small Lorenz swirl
		g.anomalyX += lorX * 0.05
		g.anomalyY += lorY * 0.05

		// wrap
		g.anomalyX = math.Mod(g.anomalyX+gridW, gridW)
		g.anomalyY = math.Mod(g.anomalyY+gridH, gridH)
	}
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
		// Initialize Anomaly position in the center
		anomalyX: float64(gridW / 2),
		anomalyY: float64(gridH / 2),
		// initialize Lorenz attractor with standard params but small dt
		lorenz: Lorenz{x: 0.1, y: 0.0, z: 0.0, sigma: 10.0, rho: 28.0, beta: 8.0 / 3.0, dt: 0.005},
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

func (g *Game) seedFromGenome(gen *Genome) {
	cx, cy := gridW/2, gridH/2
	for y := 0; y < gridH; y++ {
		for x := 0; x < gridW; x++ {
			g.A[y][x] = 0
			g.Anext[y][x] = 0
		}
	}
	base := int(math.Max(6, gen.Radius*1.5))
	for y := 0; y < gridH; y++ {
		for x := 0; x < gridW; x++ {
			d := math.Hypot(float64(x-cx), float64(y-cy))
			if d < float64(base) {
				g.A[y][x] = 1.618033 * math.Exp(-d*d/(2*float64(base)*float64(base)))
			}
			if rand.Float64() < 1.618033+0.001*rand.Float64() {
				g.A[y][x] = rand.Float64()*0.8 + 0.05
			}
		}
	}
	// Reset anomaly position to center on new genome start
	g.anomalyX = float64(gridW / 2)
	g.anomalyY = float64(gridH / 2)
}

// ---------- Single step (MODIFIED) ----------
func (g *Game) step(gen *Genome) {
	// 1. Move the anomaly (finding the shortest path to peak activity)
	g.findTargetAndMoveAnomaly()

	// 2. Perform Lenia-step
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

	// 3. Apply the devouring effect
	g.applyAnomalyEffect()

	// 4. Occasionally apply FFT-based filtering to the field to create wave-like structures
	if g.stepCount%8 == 0 {
		// cutoff influenced by lorenz z and 5D mapping
		mm := map5D(gen)
		cutoff := 0.08 + 0.4*mm[2] // radius component influences cutoff
		// further modulate by lorenz z
		cutoff *= 0.5 + 0.5*math.Tanh(g.lorenz.z*0.02)
		if cutoff < 0.02 {
			cutoff = 0.02
		}
		if cutoff > 0.95 {
			cutoff = 0.95
		}
		g.applyFFTFilter(cutoff)
	}
}

// ---------- Fitness evaluation ----------
func (g *Game) evaluateGenome(gen *Genome) float64 {
	g.applyGenomeKernel(gen)
	g.seedFromGenome(gen)

	var activitySum float64
	var varianceSum float64
	var edgeSum float64

	for step := 0; step < evalSteps; step++ {
		g.step(gen)
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

	meanActivity := activitySum / float64(evalSteps/4)
	meanVar := varianceSum / float64(evalSteps/4)
	meanEdge := edgeSum / float64(evalSteps/4)

	actScore := meanActivity * 3.0
	varScore := math.Log(1 + meanVar*200)
	edgeScore := math.Log(1 + meanEdge*100)

	score := 2.0*actScore + 1.0*varScore + 1.0*edgeScore
	if meanActivity < 0.01 {
		score = score * 0.1
	}
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
	if ebiten.IsKeyPressed(ebiten.KeySpace) {
		if time.Since(g.lastEvolveTime) > 300*time.Millisecond {
			g.autoEvolve = !g.autoEvolve
			g.lastEvolveTime = time.Now()
		}
	}
	if ebiten.IsKeyPressed(ebiten.KeyG) {
		if time.Since(g.lastEvolveTime) > 300*time.Millisecond {
			g.evolveOnce()
			g.lastEvolveTime = time.Now()
		}
	}
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

	if g.autoEvolve {
		if time.Since(g.lastEvolveTime) > g.autoEvolveDelay {
			g.currentIndex++
			g.lastEvolveTime = time.Now()
			if g.currentIndex >= len(g.population) {
				g.evolveOnce()
			} else {
				g.applyGenomeKernel(&g.population[g.currentIndex])
				g.seedFromGenome(&g.population[g.currentIndex])
				g.stepCount = 0
			}
		}
	}

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
	// Evaluate all genomes
	for i := range g.population {
		score := g.evaluateGenome(&g.population[i])
		g.population[i].Fitness = score
	}
	sort.Slice(g.population, func(i, j int) bool {
		return g.population[i].Fitness > g.population[j].Fitness
	})

	newPop := make([]Genome, 0, populationSz)
	for i := 0; i < elitism && i < len(g.population); i++ {
		newPop = append(newPop, g.population[i])
	}
	for len(newPop) < populationSz {
		a := tournamentSelect(g.population)
		b := tournamentSelect(g.population)
		child := crossover(a, b)
		mutate(&child)
		newPop = append(newPop, child)
	}
	g.population = newPop
	g.generation++
	g.currentIndex = 0
	g.applyGenomeKernel(&g.population[0])
	g.seedFromGenome(&g.population[0])
	g.stepCount = 0
}

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

// ---------- Draw / display (MODIFIED for Anomaly visualization) ----------
func (g *Game) Draw(screen *ebiten.Image) {
	bias := g.population[g.currentIndex].ColorBias
	ax, ay := g.anomalyX, g.anomalyY

	for y := 0; y < gridH; y++ {
		for x := 0; x < gridW; x++ {
			v := clamp(g.A[y][x]+bias*0.08, 0, 1)
			dx := float64(x) - ax
			dy := float64(y) - ay
			if dx > float64(gridW/2) {
				dx -= float64(gridW)
			} else if dx < float64(-gridW/2) {
				dx += float64(gridW)
			}
			if dy > float64(gridH/2) {
				dy -= float64(gridH)
			} else if dy < float64(-gridH/2) {
				dy += float64(gridH)
			}
			dist := math.Hypot(dx, dy)

			localBias := bias
			if dist < anomalyRadius {
				localBias += anomalyColorBias
				v = clamp(g.A[y][x]+localBias*0.08, 0, 1)
			}

			r, gg, b := colorRamp(v)
			g.texture.Set(x, y, color.NRGBA{R: r, G: gg, B: b, A: 0xFF})
		}
	}

	op := &ebiten.DrawImageOptions{}
	op.GeoM.Scale(float64(cellSize), float64(cellSize))
	op.Filter = ebiten.FilterNearest
	screen.DrawImage(g.texture, op)

	cur := &g.population[g.currentIndex]
	txt := fmt.Sprintf("Gen: %d  Index: %d/%d  Fitness(best): %.3f  μ:%.3f σ:%.3f R:%.2f shell:%.2f Δt:%.3f",
		g.generation, g.currentIndex, len(g.population), g.population[0].Fitness, cur.Mu, cur.Sigma, cur.Radius, cur.ShellSigma, cur.Dt)
	text.Draw(screen, txt, basicfont.Face7x13, 6, 16, color.White)

	help := fmt.Sprintf("Keys: ←/→ switch genome   G evolve once   SPACE toggle auto-evolve   (auto delay %.1fs)    FPS:", g.autoEvolveDelay.Seconds())
	text.Draw(screen, help, basicfont.Face7x13, 6, 32, color.White)
	fps := fmt.Sprintf("%d", g.lastFPS)
	text.Draw(screen, fps, basicfont.Face7x13, 6, 48, color.White)

	anomalyPos := fmt.Sprintf("Anomaly Pos: (%.1f, %.1f)  Lorenz Z: %.3f", g.anomalyX, g.anomalyY, g.lorenz.z)
	text.Draw(screen, anomalyPos, basicfont.Face7x13, 6, 64, color.White)
}

func (g *Game) Layout(outW, outH int) (int, int) {
	return gridW * cellSize, gridH * cellSize
}

// ---------- color ramp (modified to use 5D mapping subtly) ----------
func colorRamp(v float64) (r, gCol, b uint8) {
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
	ebiten.SetWindowTitle("Evolving Lenia-like Artificial Life (Ebiten) - Extended")

	game := NewGame()
	if err := ebiten.RunGame(game); err != nil {
		log.Fatal(err)
	}
}
