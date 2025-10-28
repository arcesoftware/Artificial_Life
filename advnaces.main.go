package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"runtime"
	"sync"
	"time"

	"github.com/go-gl/gl/v4.1-core/gl"
	"github.com/go-gl/glfw/v3.3/glfw"
	"github.com/go-gl/mathgl/mgl32"
)

// The Universe is 3-fold.
const numSpheres = 3

// ----------------------------------------------------
// ## Performance Optimization Structures (Spatial Hashing)
// ----------------------------------------------------

// HashBucket holds indices of particles residing in a grid cell.
type HashBucket []int

// SpatialHash is a uniform grid structure for O(N) average time neighbor finding.
type SpatialHash struct {
	Grid        map[int]HashBucket
	CellSize    float32
	InvCellSize float32
}

func NewSpatialHash(particles []Particle, maxRadius float32) *SpatialHash {
	sh := &SpatialHash{
		CellSize:    maxRadius,
		InvCellSize: 1.0 / maxRadius,
		Grid:        make(map[int]HashBucket, len(particles)),
	}
	sh.Update(particles)
	return sh
}

// HashIndex converts a 3D position to a unique integer hash for the cell.
func (sh *SpatialHash) HashIndex(pos mgl32.Vec3) int {
	// A simple multiplicative hashing function for 3D coordinates (grid cell indices)
	ix := int(pos.X() * sh.InvCellSize)
	iy := int(pos.Y() * sh.InvCellSize)
	iz := int(pos.Z() * sh.InvCellSize)
	// These large primes help distribute the hashes uniformly
	return ix*73856093 + iy*1934983 + iz*83492791
}

// Update rebuilds the hash map.
func (sh *SpatialHash) Update(particles []Particle) {
	sh.Grid = make(map[int]HashBucket, len(particles))
	for i, p := range particles {
		key := sh.HashIndex(p.Pos)
		sh.Grid[key] = append(sh.Grid[key], i)
	}
}

// FindNeighborsSH uses the Spatial Hash to only check the 27 adjacent cells.
func (sh *SpatialHash) FindNeighborsSH(index int, pos mgl32.Vec3, data []Particle, radius float32) []int {
	neighbors := []int{}
	radiusSq := radius * radius

	// Determine the grid coordinates of the particle's cell
	ix0 := int(pos.X() * sh.InvCellSize)
	iy0 := int(pos.Y() * sh.InvCellSize)
	iz0 := int(pos.Z() * sh.InvCellSize)

	// Iterate through the 3x3x3 cell neighborhood (27 cells)
	for dx := -1; dx <= 1; dx++ {
		for dy := -1; dy <= 1; dy++ {
			for dz := -1; dz <= 1; dz++ {

				// Recalculate the key for the adjacent cell (must match HashIndex logic)
				key := (ix0+dx)*73856093 + (iy0+dy)*1934983 + (iz0+dz)*83492791

				if bucket, ok := sh.Grid[key]; ok {
					for _, otherIndex := range bucket {
						if otherIndex == index {
							continue // Don't check against self
						}
						// Final distance check for precision, as cell check is only an approximation
						distSq := pos.Sub(data[otherIndex].Pos).LenSqr()
						if distSq < radiusSq {
							neighbors = append(neighbors, otherIndex)
						}
					}
				}
			}
		}
	}
	return neighbors
}

// ----------------------------------------------------
// ## Topological Analysis Structure (Union-Find for Betti-0)
// ----------------------------------------------------

// UnionFind implements the Disjoint Set Union (DSU) structure to efficiently
// track connected components (Betti-0).
type UnionFind struct {
	Parent []int // Parent[i] is the parent of node i
	Size   []int // Size[i] stores the size of the tree rooted at i
}

func NewUnionFind(n int) *UnionFind {
	parent := make([]int, n)
	size := make([]int, n)
	for i := range parent {
		parent[i] = i // Each node is initially its own root
		size[i] = 1
	}
	return &UnionFind{Parent: parent, Size: size}
}

// Find with Path Compression: Finds the root of the set containing element i.
func (uf *UnionFind) Find(i int) int {
	if uf.Parent[i] == i {
		return i
	}
	// Path compression: set parent to the root for faster lookups
	uf.Parent[i] = uf.Find(uf.Parent[i])
	return uf.Parent[i]
}

// Union by Size: Merges the sets containing elements i and j.
func (uf *UnionFind) Union(i, j int) {
	rootI := uf.Find(i)
	rootJ := uf.Find(j)
	if rootI != rootJ {
		// Union by size heuristic: attach the smaller tree to the root of the larger tree
		if uf.Size[rootI] < uf.Size[rootJ] {
			rootI, rootJ = rootJ, rootI
		}
		uf.Parent[rootJ] = rootI
		uf.Size[rootI] += uf.Size[rootJ]
	}
}

// ----------------------------------------------------
// ## Simulation Data and Constants
// ----------------------------------------------------

type SimData struct {
	Particles []Particle
	// Coms[i] stores the KINEMATIC TARGET CENTER for sphere i
	Coms [numSpheres]mgl32.Vec3

	// Topological State for the Poincaré Polynomial P(x) = B0 + B1*x
	Beta0 [numSpheres]int // Number of connected components (b_0)
	Beta1 [numSpheres]int // Hole approximation (b_1)

	// Internal flag: true if Beta0 == 1
	Connected [numSpheres]bool
	// ParticleID[i] stores 0, 1, or 2
	ParticleID []int
}

const (
	winWidth   = 1280
	winHeight  = 720
	nParticles = 2400 // Particle count optimized for 3 spheres
	pointSize  = 1.5

	// Performance / Parallelism
	particleVboSize = 6 * 4
	numWorkers      = 8

	// DDG Constants
	kNeighbors = 4     // Target number of neighbors for local structure
	ddgRadius  = 30.0  // Interaction radius for DDG
	ddgSpringK = 150.0 // Stiffness of the DDG spring

	// TDA Constants (Betti-0: Component Repair)
	tdaClusterRadius = 150.0       // Distance threshold for component linking
	tdaRestoreForce  = 700.0       // Strong force for Betti-0 repair
	tdaDamping       = 0.618033989 // Golden Ratio Damping (quick convergence)

	// TDA Constants (Betti-1: Hole Repair / Thin-Spot)
	tdaBetaOneThreshold  = 2     // If neighbors < (kNeighbors - 2), trigger Beta-1 repair
	tdaBetaOneForceK     = 300.0 // Inward force strength for hole-filling
	betaOneParticleRatio = 0.01  // If more than 1% of particles are B1-active, set B1=1

	// Confinement
	sphereRadius   = 120.0
	springStrength = 22.0

	// --- Spirograph Kinematic Parameters (Hypotrochoid) ---
	spiroR     float32 = 180.0
	spiroK     float32 = 0.531 // k = r/R
	spiroL     float32 = 0.854 // l = rho/r
	spiroSpeed float32 = 0.8

	// Inter-Sphere Repulsion
	repulsionK         float32 = 120.0
	repulsionDistCheck         = 300.0

	// Logging
	logIntervalSeconds = 5.0
)

type Particle struct {
	Pos mgl32.Vec3
	Vel mgl32.Vec3
	Col mgl32.Vec3
}

var (
	simData         SimData
	timeAccumulator float32 // Tracks total elapsed simulation time
	logTimer        float32 // Timer for periodic logging

	prog uint32
	vao  uint32
	vbo  uint32

	// Camera Controls
	azimuth, elevation float64 = 0.6, 0.2
	distance           float64 = 700
	lastX, lastY       float64
	dragging           bool
)

func init() {
	runtime.LockOSThread()
}

// ----------------------------------------------------
// ## Main and Setup
// ----------------------------------------------------

func main() {
	rand.Seed(time.Now().UnixNano())
	if err := glfw.Init(); err != nil {
		log.Fatalln("failed to init glfw:", err)
	}
	defer glfw.Terminate()

	glfw.WindowHint(glfw.ContextVersionMajor, 4)
	glfw.WindowHint(glfw.ContextVersionMinor, 1)
	glfw.WindowHint(glfw.OpenGLProfile, glfw.OpenGLCoreProfile)
	glfw.WindowHint(glfw.OpenGLForwardCompatible, glfw.True)

	window, err := glfw.CreateWindow(winWidth, winHeight, "Tri-Sphere TDA Dynamics", nil, nil)
	if err != nil {
		panic(err)
	}
	window.MakeContextCurrent()
	if err := gl.Init(); err != nil {
		panic(err)
	}

	gl.Enable(gl.DEPTH_TEST)
	gl.Enable(gl.BLEND)
	gl.BlendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA)
	gl.PointSize(pointSize)

	prog, err = newProgram(vertexShader, fragmentShader)
	if err != nil {
		panic(err)
	}

	// --- Particle Initialization (Triple Spheres, started near origin) ---
	particles := make([]Particle, nParticles)
	simData.ParticleID = make([]int, nParticles)
	initialCenter := mgl32.Vec3{0, 0, 0}

	for i := 0; i < nParticles; i++ {
		sphereID := i % numSpheres // Assign ID 0, 1, or 2
		simData.ParticleID[i] = sphereID

		// Color Coding: Red, Blue, Green
		color := mgl32.Vec3{1.0, 0.2, 0.2} // Red (0)
		if sphereID == 1 {
			color = mgl32.Vec3{0.2, 0.2, 1.0} // Blue (1)
		} else if sphereID == 2 {
			color = mgl32.Vec3{0.2, 1.0, 0.2} // Green (2)
		}

		// Scatter particles tightly around the target sphere surface
		const jitter float32 = 5.0
		dist := sphereRadius + float32(rand.Float64()*2-1)*jitter

		angle1 := float32(rand.Float64() * 2 * math.Pi)
		angle2 := float32(rand.Float64() * math.Pi)

		posOffset := mgl32.Vec3{
			dist * float32(math.Cos(float64(angle1))) * float32(math.Sin(float64(angle2))),
			dist * float32(math.Sin(float64(angle1))) * float32(math.Sin(float64(angle2))),
			dist * float32(math.Cos(float64(angle2))),
		}

		particles[i].Pos = initialCenter.Add(posOffset)
		// Small initial random velocity for quick settling
		particles[i].Vel = mgl32.Vec3{
			float32(rand.Float64()*2-1) * 2,
			float32(rand.Float64()*2-1) * 2,
			float32(rand.Float64()*2-1) * 2,
		}
		particles[i].Col = color
	}

	simData.Particles = particles

	setupBuffers()
	window.SetCursorPosCallback(mouseMove)
	window.SetMouseButtonCallback(mouseButton)
	window.SetScrollCallback(scrollCallback)

	prev := time.Now()
	for !window.ShouldClose() {
		now := time.Now()
		dt := now.Sub(prev).Seconds()
		prev = now

		updateParticlesPerformant(dt)
		render()
		window.SwapBuffers()
		glfw.PollEvents()
	}
}

// ----------------------------------------------------
// ## Spirograph Kinematics (Three-Body Entanglement)
// ----------------------------------------------------

// updateSphereCenters calculates the desired (kinematic) center position for each sphere
// based on the 2D Spirograph equations lifted to 3D.
func updateSphereCenters(dt float64) {
	timeAccumulator += float32(dt)
	t := timeAccumulator * spiroSpeed

	R := spiroR
	k := spiroK
	l := spiroL
	ratioTerm := (1.0 - k) / k

	// Spirograph Path (Base X/Y)
	term1_x := R * (1.0 - k) * float32(math.Cos(float64(t)))
	term1_y := R * (1.0 - k) * float32(math.Sin(float64(t)))
	term2_x := R * l * k * float32(math.Cos(float64(ratioTerm*t)))
	term2_y := R * l * k * float32(math.Sin(float64(ratioTerm*t)))

	x_base := term1_x + term2_x
	y_base := term1_y - term2_y

	// Oscillation terms
	z_osc1 := R * 0.2 * float32(math.Sin(float64(t*0.5)))
	z_osc2 := R * 0.2 * float32(math.Cos(float64(t*0.5)))
	xy_osc := R * 0.1 * float32(math.Sin(float64(t)))

	// --- Assign Center Targets (3 Bodies) ---

	// Sphere 0 (Red): Base path + Z oscillation 1
	xA := x_base + xy_osc
	yA := y_base
	zA := z_osc1
	simData.Coms[0] = mgl32.Vec3{xA, yA, zA}

	// Sphere 1 (Blue): Mirrored path + Z oscillation 2
	xB := -x_base
	yB := -y_base + xy_osc
	zB := -z_osc2
	simData.Coms[1] = mgl32.Vec3{xB, yB, zB}

	// Sphere 2 (Green): Perpendicular/Out-of-Plane Motion
	// Traces a smaller circle on the Z=0 plane and oscillates perpendicular to the main motion.
	xC := R * 0.6 * float32(math.Cos(float64(t*1.5)))
	yC := R * 0.6 * float32(math.Sin(float64(t*1.5)))
	zC := R * 0.4 * float32(math.Sin(float64(t*0.7)))
	simData.Coms[2] = mgl32.Vec3{xC, yC, zC}
}

// ----------------------------------------------------
// ## High-Performance Dynamics Loop (The Calculation)
// ----------------------------------------------------

func updateParticlesPerformant(dt float64) {
	dt32 := float32(dt)
	dampingMultiplier := float32(1.0 - tdaDamping)

	// STEP 0: UPDATE DYNAMIC CENTER TARGETS (The Spirograph motion)
	updateSphereCenters(dt)

	// Step 1: TDA/Topological Analysis (β0 check)
	actualCOMs := [numSpheres]mgl32.Vec3{}
	// This function updates simData.Beta0 and simData.Connected, and fills actualCOMs
	TopologicalAnalysisModule(simData.Particles, simData.ParticleID, actualCOMs[:], simData.Connected[:], simData.Beta0[:])

	// Step 2: Rebuild Spatial Hash (O(N) operation)
	sh := NewSpatialHash(simData.Particles, ddgRadius)

	// Track results from workers (B1 activations per sphere)
	workerResults := make([][numSpheres]int, numWorkers)
	var wg sync.WaitGroup

	// -----------------------------------------------------------------
	// Step 3: DDG/TDA Physics Integration (Parallelized)
	// -----------------------------------------------------------------

	chunkSize := nParticles / numWorkers
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)

		start := w * chunkSize
		end := start + chunkSize
		if w == numWorkers-1 {
			end = nParticles
		}

		// Pass a pointer to the worker's result slot for non-concurrent write
		go func(start, end int, resultPtr *[numSpheres]int) {
			defer wg.Done()

			// Local array to accumulate B1 activations in this worker thread
			localBetaOneActiveCounts := [numSpheres]int{0, 0, 0}

			for i := start; i < end; i++ {
				p := &simData.Particles[i]
				pos := p.Pos
				sphereID := simData.ParticleID[i]

				targetCenter := simData.Coms[sphereID]
				isConnected := simData.Connected[sphereID]

				// 1. DDG Force and Betti-1 Check (Local Surface Maintenance and Hole Repair)
				ddgForce, betaOneForce, betaOneActive := getDDGAndBetaOneForce(i, pos, sh, simData.Particles)
				if betaOneActive {
					localBetaOneActiveCounts[sphereID]++
				}

				// 2. Spring Force (Kinematic Confinement)
				centerForce := getSphereSpringForce(pos, targetCenter, sphereRadius)

				// 3. TDA Betti-0 Restoring Force (Topology Repair)
				tdaForce := mgl32.Vec3{0, 0, 0}
				if !isConnected {
					// Apply force towards the actual Center of Mass to reform the cluster
					actualCOM := actualCOMs[sphereID]
					dir := actualCOM.Sub(pos).Normalize()

					// Force scales with distance from actual COM to restore the cluster faster
					distToCOM := actualCOM.Sub(pos).Len()
					tdaForce = dir.Mul(tdaRestoreForce * (1.0 + distToCOM/sphereRadius))
				}

				// 4. Inter-Sphere Repulsion
				repulsionForce := mgl32.Vec3{0, 0, 0}
				for otherID := 0; otherID < numSpheres; otherID++ {
					if otherID != sphereID {
						otherCenter := simData.Coms[otherID]
						repulsionForce = repulsionForce.Add(getInterSphereRepulsion(pos, otherCenter))
					}
				}

				// Combined Acceleration: Summation of all geometric and topological forces
				totalAcc := ddgForce.Add(centerForce).Add(tdaForce).Add(repulsionForce).Add(betaOneForce)

				// Velocity and Position Update (Verlet Integration proxy)
				p.Vel = p.Vel.Add(totalAcc.Mul(dt32))
				p.Vel = p.Vel.Mul(dampingMultiplier) // High damping ensures stability
				p.Pos = p.Pos.Add(p.Vel.Mul(dt32))
			}
			// Write local result to the dedicated slot
			*resultPtr = localBetaOneActiveCounts
		}(start, end, &workerResults[w])
	}

	wg.Wait()

	// Step 4: Aggregate Results and Calculate Global Betti-1 Proxy
	betaOneActiveCounts := [numSpheres]int{0, 0, 0}
	for _, res := range workerResults {
		for id := 0; id < numSpheres; id++ {
			betaOneActiveCounts[id] += res[id]
		}
	}

	for id := 0; id < numSpheres; id++ {
		// Calculate the Betti-1 approximation
		totalParticlesInSphere := nParticles / numSpheres
		// FIX: Use float32() for type conversion
		if float32(betaOneActiveCounts[id]) > float32(totalParticlesInSphere)*betaOneParticleRatio {
			simData.Beta1[id] = 1 // Thin-spot/hole proxy is active
		} else {
			simData.Beta1[id] = 0 // Structure is locally stable
		}
	}

	// Log the current topological state periodically
	logTimer += dt32
	if logTimer >= logIntervalSeconds {
		logTimer = 0
		logPoincarePolynomials()
	}
}

// Logs the current Poincaré polynomial approximation for all three spheres.
func logPoincarePolynomials() {
	colors := []string{"Red", "Blue", "Green"}
	log.Printf("--- Topological Signature (Poincaré Polynomial P(x) = b₀ + b₁x) ---")

	for i := 0; i < numSpheres; i++ {
		b0 := simData.Beta0[i]
		b1 := simData.Beta1[i]

		// Ideal S^2: P(x) = 1 + x^2. Our approximation: P(x) ≈ b_0 + b_1*x
		poly := fmt.Sprintf("%d", b0)
		if b1 > 0 {
			poly += fmt.Sprintf(" + %d x", b1)
		}

		status := "Ideal"
		if b0 > 1 {
			status = "CRITICAL (Fragmented)"
		} else if b1 > 0 {
			status = "WARNING (Thin-Spot)"
		}

		log.Printf("[%s Sphere] P(x) = %s  | Status: %s", colors[i], poly, status)
	}
}

// ----------------------------------------------------
// ## Force Modules (DDG & Betti-1)
// ----------------------------------------------------

// getDDGAndBetaOneForce calculates the local DDG force and the Betti-1 hole-filling force.
// Returns an additional boolean indicating if the Betti-1 force was applied.
func getDDGAndBetaOneForce(index int, pos mgl32.Vec3, sh *SpatialHash, data []Particle) (ddgForce mgl32.Vec3, betaOneForce mgl32.Vec3, betaOneActive bool) {
	// 1. Find Neighbors
	neighbors := sh.FindNeighborsSH(index, pos, data, ddgRadius)
	numNeighbors := len(neighbors)

	if numNeighbors == 0 {
		return mgl32.Vec3{0, 0, 0}, mgl32.Vec3{0, 0, 0}, false
	}

	// 2. Calculate DDG Force (Local Surface Maintenance)
	totalForce := mgl32.Vec3{0, 0, 0}
	// Target distance is scaled to favor a denser local structure than a linear arrangement
	targetDist := ddgRadius / float32(math.Sqrt(float64(kNeighbors))) * 0.7

	for _, neighborIndex := range neighbors {
		neighborPos := data[neighborIndex].Pos
		vec := neighborPos.Sub(pos)
		dist := vec.Len()
		forceMag := ddgSpringK * (dist - targetDist)

		if dist > 0.001 {
			dir := vec.Normalize()
			// DDG force is distributed among neighbors (div by 2.0)
			totalForce = totalForce.Add(dir.Mul(forceMag * 0.5))
		}
	}
	// Average force contribution from all neighbors
	ddgForce = totalForce.Mul(1.0 / float32(numNeighbors))

	// 3. Betti-1 Check (Hole/Thin-Spot Detection and Repair)
	betaOneForce = mgl32.Vec3{0, 0, 0}
	betaOneActive = false

	// If local neighbor count is critically low (less than kNeighbors - threshold)
	if numNeighbors < kNeighbors-tdaBetaOneThreshold {

		// Calculate the local Center of Neighbors (CoN)
		sumPos := mgl32.Vec3{0, 0, 0}
		for _, nIndex := range neighbors {
			sumPos = sumPos.Add(data[nIndex].Pos)
		}
		CoN := sumPos.Mul(1.0 / float32(numNeighbors))

		// Force direction: from the particle towards the CoN.
		// This attempts to 'tuck' the particle inwards to thicken the surface and close the gap.
		dir := CoN.Sub(pos).Normalize()
		betaOneForce = dir.Mul(tdaBetaOneForceK)
		betaOneActive = true
	}

	return ddgForce, betaOneForce, betaOneActive
}

// Confinement force: pulls particle toward its moving sphere's target center.
func getSphereSpringForce(pos mgl32.Vec3, center mgl32.Vec3, targetRadius float32) mgl32.Vec3 {
	vec := pos.Sub(center)
	dist := float64(vec.Len())

	if dist < 0.001 {
		return mgl32.Vec3{0, 0, 0}
	}

	dir := vec.Normalize()
	// Force is proportional to the deviation from the target radius
	force := springStrength * (dist - float64(targetRadius))
	return dir.Mul(-float32(force)) // Negative force pulls it back
}

// Repulsion force between a particle and the OTHER sphere's center.
func getInterSphereRepulsion(pos mgl32.Vec3, otherCenter mgl32.Vec3) mgl32.Vec3 {
	vec := pos.Sub(otherCenter)
	dist := vec.Len()

	if dist > repulsionDistCheck {
		return mgl32.Vec3{0, 0, 0}
	}

	// Inverse distance squared falloff for repulsion: F = k / d^2
	forceMag := repulsionK / (dist * dist)
	dir := vec.Normalize()

	return dir.Mul(forceMag)
}

// ----------------------------------------------------
// ## TDA Module (Betti-0: Connected Components)
// ----------------------------------------------------

// TopologicalAnalysisModule calculates actual COM and Betti-0 connectivity for all spheres.
// It now updates the Betti-0 count in the simData struct.
func TopologicalAnalysisModule(data []Particle, ids []int, coms []mgl32.Vec3, connected []bool, beta0 []int) {
	// Group particles by sphere ID (0, 1, or 2)
	groups := make([][]int, numSpheres)
	for i, id := range ids {
		groups[id] = append(groups[id], i)
	}

	// 1. Calculate Center of Mass (COM) for each sphere (Actual COM)
	sumCom := [numSpheres]mgl32.Vec3{}
	counts := [numSpheres]int{}

	for i, p := range data {
		id := ids[i]
		sumCom[id] = sumCom[id].Add(p.Pos)
		counts[id]++
	}

	for i := 0; i < numSpheres; i++ {
		if counts[i] > 0 {
			coms[i] = sumCom[i].Mul(1.0 / float32(counts[i]))
		} else {
			coms[i] = mgl32.Vec3{0, 0, 0}
		}
	}

	// 2. Betti-0 Connectivity Check (Union-Find)
	radiusSq := float32(0.25) * tdaClusterRadius * tdaClusterRadius

	for sphereID := 0; sphereID < numSpheres; sphereID++ {
		group := groups[sphereID]
		if len(group) <= 1 {
			connected[sphereID] = true
			beta0[sphereID] = 1
			continue
		}

		// Map global particle index to local Union-Find index
		indexMap := make(map[int]int, len(group))
		for localIdx, globalIdx := range group {
			indexMap[globalIdx] = localIdx
		}

		uf := NewUnionFind(len(group))

		// O(N^2) check within the group, which remains the single most expensive part
		for i := 0; i < len(group); i++ {
			p1GlobalIdx := group[i]
			p1Pos := data[p1GlobalIdx].Pos

			for j := i + 1; j < len(group); j++ {
				p2GlobalIdx := group[j]
				p2Pos := data[p2GlobalIdx].Pos

				// FIX: Removed extra opening brace {
				if p1Pos.Sub(p2Pos).LenSqr() < radiusSq {
					uf.Union(indexMap[p1GlobalIdx], indexMap[p2GlobalIdx])
				}
			}
		}

		// Count components (Betti-0)
		numComponents := 0
		for i := 0; i < len(group); i++ {
			if uf.Parent[i] == i {
				numComponents++
			}
		}
		// Update the Betti-0 count
		beta0[sphereID] = numComponents
		// The sphere is broken if Betti-0 > 1
		connected[sphereID] = (numComponents <= 1)
	}

}

// ----------------------------------------------------
// ## Rendering, Camera, and Shaders
// ----------------------------------------------------

func setupBuffers() {
	gl.GenVertexArrays(1, &vao)
	gl.BindVertexArray(vao)
	gl.GenBuffers(1, &vbo)
	gl.BindBuffer(gl.ARRAY_BUFFER, vbo)
	// Buffer size for Pos(3) + Col(3) = 6 floats per particle
	bufferSize := nParticles * particleVboSize
	gl.BufferData(gl.ARRAY_BUFFER, bufferSize, nil, gl.DYNAMIC_DRAW)
	stride := int32(particleVboSize)

	// Position attribute (layout = 0)
	gl.EnableVertexAttribArray(0)
	gl.VertexAttribPointer(0, 3, gl.FLOAT, false, stride, gl.PtrOffset(0))
	// Color attribute (layout = 1)
	gl.EnableVertexAttribArray(1)
	gl.VertexAttribPointer(1, 3, gl.FLOAT, false, stride, gl.PtrOffset(3*4))

	gl.BindVertexArray(0)
}

func render() {
	gl.ClearColor(0, 0, 0, 1)
	gl.Clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)
	gl.UseProgram(prog)

	// Set up View-Projection Matrix
	proj := mgl32.Perspective(mgl32.DegToRad(45), float32(winWidth)/winHeight, 0.1, 5000)
	view := cameraMatrix()
	vp := proj.Mul4(view)
	loc := gl.GetUniformLocation(prog, gl.Str("uVP\x00"))
	gl.UniformMatrix4fv(loc, 1, false, &vp[0])

	// Prepare VBO Data
	vboData := make([]float32, 0, nParticles*6)
	for _, p := range simData.Particles {
		vboData = append(vboData, p.Pos.X(), p.Pos.Y(), p.Pos.Z())
		vboData = append(vboData, p.Col.X(), p.Col.Y(), p.Col.Z())
	}

	// Stream data to GPU
	gl.BindBuffer(gl.ARRAY_BUFFER, vbo)
	dataSize := len(vboData) * 4
	gl.BufferSubData(gl.ARRAY_BUFFER, 0, dataSize, gl.Ptr(vboData))

	// Draw particles
	gl.BindVertexArray(vao)
	gl.DrawArrays(gl.POINTS, 0, int32(nParticles))
	gl.BindVertexArray(0)
}

func cameraMatrix() mgl32.Mat4 {
	x := float32(distance * math.Cos(azimuth) * math.Cos(elevation))
	y := float32(distance * math.Sin(elevation))
	z := float32(distance * math.Sin(azimuth) * math.Cos(elevation))
	return mgl32.LookAtV(mgl32.Vec3{x, y, z}, mgl32.Vec3{0, 0, 0}, mgl32.Vec3{0, 1, 0})
}

// Standard boilerplate for user input and GLSL utilities
func mouseMove(w *glfw.Window, xpos, ypos float64) {
	if dragging {
		azimuth -= (xpos - lastX) * 0.005
		elevation -= (ypos - lastY) * 0.005
		if elevation > 1.4 {
			elevation = 1.4
		} else if elevation < -1.4 {
			elevation = -1.4
		}
	}
	lastX = xpos
	lastY = ypos
}

func mouseButton(w *glfw.Window, button glfw.MouseButton, action glfw.Action, mods glfw.ModifierKey) {
	if button == glfw.MouseButton1 {
		dragging = action == glfw.Press
	}
}

func scrollCallback(w *glfw.Window, xoff, yoff float64) {
	distance -= yoff * 30
	if distance < 200 {
		distance = 200
	}
	if distance > 3000 {
		distance = 3000
	}
}

func compileShader(src string, shaderType uint32) (uint32, error) {
	shader := gl.CreateShader(shaderType)
	csources, free := gl.Strs(src)
	gl.ShaderSource(shader, 1, csources, nil)
	free()
	gl.CompileShader(shader)
	var status int32
	gl.GetShaderiv(shader, gl.COMPILE_STATUS, &status)
	if status == gl.FALSE {
		var logLength int32
		gl.GetShaderiv(shader, gl.INFO_LOG_LENGTH, &logLength)
		logBuf := make([]byte, logLength+1)
		gl.GetShaderInfoLog(shader, logLength, nil, &logBuf[0])
		return 0, fmt.Errorf("failed to compile shader: %s", string(logBuf))
	}
	return shader, nil
}

func newProgram(vertSrc, fragSrc string) (uint32, error) {
	vert, err := compileShader(vertSrc, gl.VERTEX_SHADER)
	if err != nil {
		return 0, err
	}
	frag, err := compileShader(fragSrc, gl.FRAGMENT_SHADER)
	if err != nil {
		return 0, err
	}
	prog := gl.CreateProgram()
	gl.AttachShader(prog, vert)
	gl.AttachShader(prog, frag)
	gl.LinkProgram(prog)
	var status int32
	gl.GetProgramiv(prog, gl.LINK_STATUS, &status)
	if status == gl.FALSE {
		var logLength int32
		gl.GetProgramiv(prog, gl.INFO_LOG_LENGTH, &logLength)
		logBuf := make([]byte, logLength+1)
		gl.GetProgramInfoLog(prog, logLength, nil, &logBuf[0])
		return 0, fmt.Errorf("failed to link program: %s", string(logBuf))
	}
	gl.DeleteShader(vert)
	gl.DeleteShader(frag)
	return prog, nil
}

// --------- Shaders ---------
var vertexShader = `
#version 410 core
layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inCol;
uniform mat4 uVP;
out vec3 vColor;
void main() {
	gl_Position = uVP * vec4(inPos, 1.0);
	vColor = inCol;
}
` + "\x00"

var fragmentShader = `
#version 410 core
in vec3 vColor;
out vec4 fragColor;
void main() {
	fragColor = vec4(vColor, 1.0);
}
` + "\x00"
