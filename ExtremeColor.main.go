// main.go (optimized)
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

// -----------------------------
// Configuration / Constants
// -----------------------------

const numSpheres = 3

const (
	winWidth   = 1280
	winHeight  = 720
	nParticles = 2400
	// pos(3) + col(3) + vel(3) = 9 floats per particle
	particleVboSize = 9 * 4 // bytes
	pointSize       = 1.5

	numWorkers = 8

	kNeighbors = 4
	ddgRadius  = 30.0
	ddgSpringK = 150.0

	tdaClusterRadius = 150.0
	tdaRestoreForce  = 700.0
	tdaDamping       = 0.618033989

	tdaBetaOneThreshold  = 2
	tdaBetaOneForceK     = 300.0
	betaOneParticleRatio = 0.01

	sphereRadius   = 120.0
	springStrength = 22.0

	spiroR     float32 = 180.0
	spiroK     float32 = 0.531
	spiroL     float32 = 0.854
	spiroSpeed float32 = 0.8

	repulsionK         float32 = 120.0
	repulsionDistCheck         = 300.0

	logIntervalSeconds = 5.0
)

// -----------------------------
// Types: Particle, SpatialHash, UnionFind, SimData
// -----------------------------

type Particle struct {
	Pos mgl32.Vec3
	Vel mgl32.Vec3
	Col mgl32.Vec3
}

type HashBucket []int

type SpatialHash struct {
	Grid        map[int]HashBucket
	CellSize    float32
	InvCellSize float32
}

// Reuse the same map instead of reallocating each frame.
// Update will clear existing buckets (delete keys) and repopulate.
func NewSpatialHash(particles []Particle, maxRadius float32) *SpatialHash {
	sh := &SpatialHash{
		CellSize:    maxRadius,
		InvCellSize: 1.0 / maxRadius,
		Grid:        make(map[int]HashBucket, len(particles)),
	}
	sh.Update(particles)
	return sh
}

func (sh *SpatialHash) HashIndexPosComponents(ix, iy, iz int) int {
	// same hash weights you used
	return ix*73856093 + iy*1934983 + iz*83492791
}

func (sh *SpatialHash) HashIndex(pos mgl32.Vec3) int {
	ix := int(pos.X() * sh.InvCellSize)
	iy := int(pos.Y() * sh.InvCellSize)
	iz := int(pos.Z() * sh.InvCellSize)
	return sh.HashIndexPosComponents(ix, iy, iz)
}

func (sh *SpatialHash) Update(particles []Particle) {
	// clear map but keep allocation to avoid reallocation cost
	for k := range sh.Grid {
		delete(sh.Grid, k)
	}
	// repopulate
	for i, p := range particles {
		key := sh.HashIndex(p.Pos)
		sh.Grid[key] = append(sh.Grid[key], i)
	}
}

// FindNeighborsSH: same logic but avoid some allocations inside loops
func (sh *SpatialHash) FindNeighborsSH(index int, pos mgl32.Vec3, data []Particle, radius float32) []int {
	radiusSq := radius * radius
	ix0 := int(pos.X() * sh.InvCellSize)
	iy0 := int(pos.Y() * sh.InvCellSize)
	iz0 := int(pos.Z() * sh.InvCellSize)

	// Reserve a small slice; will grow if needed
	neighbors := make([]int, 0, 8)

	for dx := -1; dx <= 1; dx++ {
		baseX := (ix0 + dx) * 73856093
		for dy := -1; dy <= 1; dy++ {
			baseY := (iy0 + dy) * 1934983
			for dz := -1; dz <= 1; dz++ {
				key := baseX + baseY + (iz0+dz)*83492791
				if bucket, ok := sh.Grid[key]; ok {
					for _, otherIndex := range bucket {
						if otherIndex == index {
							continue
						}
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

type UnionFind struct {
	Parent []int
	Size   []int
}

func NewUnionFind(n int) *UnionFind {
	parent := make([]int, n)
	size := make([]int, n)
	for i := range parent {
		parent[i] = i
		size[i] = 1
	}
	return &UnionFind{Parent: parent, Size: size}
}

func (uf *UnionFind) Find(i int) int {
	// path compression
	if uf.Parent[i] != i {
		uf.Parent[i] = uf.Find(uf.Parent[i])
	}
	return uf.Parent[i]
}

func (uf *UnionFind) Union(i, j int) {
	rootI := uf.Find(i)
	rootJ := uf.Find(j)
	if rootI != rootJ {
		if uf.Size[rootI] < uf.Size[rootJ] {
			rootI, rootJ = rootJ, rootI
		}
		uf.Parent[rootJ] = rootI
		uf.Size[rootI] += uf.Size[rootJ]
	}
}

type SimData struct {
	Particles  []Particle
	Coms       [numSpheres]mgl32.Vec3
	Beta0      [numSpheres]int
	Beta1      [numSpheres]int
	Connected  [numSpheres]bool
	ParticleID []int

	// pre-allocated VBO float buffer reused every frame (optimization)
	VBOFloatBuf []float32
}

// -----------------------------
// Globals
// -----------------------------

var (
	simData         SimData
	timeAccumulator float32
	logTimer        float32

	prog uint32
	vao  uint32
	vbo  uint32

	camera = Camera{
		Position:    mgl32.Vec3{0, 0, 700},
		Front:       mgl32.Vec3{0, 0, -1},
		Up:          mgl32.Vec3{0, 1, 0},
		WorldUp:     mgl32.Vec3{0, 1, 0},
		Yaw:         -90.0,
		Pitch:       0.0,
		Speed:       500.0,
		Sensitivity: 0.2,
	}

	lastX, lastY float64
	dragging     bool

	currentWidth  = winWidth
	currentHeight = winHeight
)

func init() {
	runtime.LockOSThread()
}

// -----------------------------
// Camera
// -----------------------------

type Camera struct {
	Position    mgl32.Vec3
	Front       mgl32.Vec3
	Up          mgl32.Vec3
	Right       mgl32.Vec3
	WorldUp     mgl32.Vec3
	Yaw, Pitch  float64
	Speed       float32
	Sensitivity float32
}

func (cam *Camera) UpdateVectors() {
	// convert degrees to radians using mgl32 helper + math trig
	yawRad := float32(cam.Yaw) * (math.Pi / 180.0)
	pitchRad := float32(cam.Pitch) * (math.Pi / 180.0)

	// maintain float32 math where possible
	cosYaw := float32(math.Cos(float64(yawRad)))
	sinYaw := float32(math.Sin(float64(yawRad)))
	cosPitch := float32(math.Cos(float64(pitchRad)))
	sinPitch := float32(math.Sin(float64(pitchRad)))

	front := mgl32.Vec3{
		cosYaw * cosPitch,
		sinPitch,
		sinYaw * cosPitch,
	}
	cam.Front = front.Normalize()
	cam.Right = cam.Front.Cross(cam.WorldUp).Normalize()
	cam.Up = cam.Right.Cross(cam.Front).Normalize()
}

func processInput(window *glfw.Window, dt float32) {
	velocity := camera.Speed * dt
	if window.GetKey(glfw.KeyW) == glfw.Press {
		camera.Position = camera.Position.Add(camera.Front.Mul(velocity))
	}
	if window.GetKey(glfw.KeyS) == glfw.Press {
		camera.Position = camera.Position.Sub(camera.Front.Mul(velocity))
	}
	if window.GetKey(glfw.KeyA) == glfw.Press {
		camera.Position = camera.Position.Sub(camera.Right.Mul(velocity))
	}
	if window.GetKey(glfw.KeyD) == glfw.Press {
		camera.Position = camera.Position.Add(camera.Right.Mul(velocity))
	}
	if window.GetKey(glfw.KeySpace) == glfw.Press {
		camera.Position = camera.Position.Add(camera.WorldUp.Mul(velocity))
	}
	if window.GetKey(glfw.KeyLeftControl) == glfw.Press || window.GetKey(glfw.KeyLeftShift) == glfw.Press {
		camera.Position = camera.Position.Sub(camera.WorldUp.Mul(velocity))
	}
	// Ctrl+G to recenter on average COM
	if window.GetKey(glfw.KeyG) == glfw.Press && window.GetKey(glfw.KeyLeftControl) == glfw.Press {
		total := mgl32.Vec3{0, 0, 0}
		for _, p := range simData.Particles {
			total = total.Add(p.Pos)
		}
		if len(simData.Particles) > 0 {
			total = total.Mul(1.0 / float32(len(simData.Particles)))
		}
		camera.Position = total.Add(mgl32.Vec3{0, 0, 700})
		camera.UpdateVectors()
	}
}

func mouseCallback(w *glfw.Window, xpos, ypos float64) {
	if !dragging {
		lastX = xpos
		lastY = ypos
		return
	}

	xoffset := xpos - lastX
	yoffset := lastY - ypos
	lastX = xpos
	lastY = ypos

	xoffset *= float64(camera.Sensitivity)
	yoffset *= float64(camera.Sensitivity)

	camera.Yaw += xoffset
	camera.Pitch += yoffset

	if camera.Pitch > 89.0 {
		camera.Pitch = 89.0
	}
	if camera.Pitch < -89.0 {
		camera.Pitch = -89.0
	}

	camera.UpdateVectors()
}

func mouseButtonCallback(w *glfw.Window, button glfw.MouseButton, action glfw.Action, mods glfw.ModifierKey) {
	if button == glfw.MouseButtonLeft {
		dragging = (action == glfw.Press)
	}
}

func scrollCallback(w *glfw.Window, xoff, yoff float64) {
	camera.Position = camera.Position.Add(camera.Front.Mul(float32(yoff) * 30.0))
}

// -----------------------------
// Main
// -----------------------------

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

	window, err := glfw.CreateWindow(winWidth, winHeight, "Tri-Sphere TDA Dynamics - Glow", nil, nil)
	if err != nil {
		panic(err)
	}
	window.MakeContextCurrent()

	if err := gl.Init(); err != nil {
		panic(err)
	}

	// Enable features
	gl.Enable(gl.DEPTH_TEST)
	gl.Enable(gl.BLEND)
	// We'll switch blending funcs during render passes
	gl.PointSize(pointSize)

	// Create program
	prog, err = newProgram(vertexShader, fragmentShader)
	if err != nil {
		panic(err)
	}

	// --- Particle Initialization ---
	particles := make([]Particle, nParticles)
	simData.ParticleID = make([]int, nParticles)
	initialCenter := mgl32.Vec3{0, 0, 0}

	for i := 0; i < nParticles; i++ {
		sphereID := i % numSpheres
		simData.ParticleID[i] = sphereID

		color := mgl32.Vec3{1.0, 0.2, 0.2}
		if sphereID == 1 {
			color = mgl32.Vec3{0.2, 0.2, 1.0}
		} else if sphereID == 2 {
			color = mgl32.Vec3{0.2, 1.0, 0.2}
		}

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
		particles[i].Vel = mgl32.Vec3{
			float32(rand.Float64()*2-1) * 2,
			float32(rand.Float64()*2-1) * 2,
			float32(rand.Float64()*2-1) * 2,
		}
		particles[i].Col = color
	}

	simData.Particles = particles

	// allocate VBO float buffer once and reuse (optimization)
	simData.VBOFloatBuf = make([]float32, nParticles*9)

	setupBuffers()

	// Input & callbacks
	window.SetInputMode(glfw.CursorMode, glfw.CursorDisabled)
	window.SetCursorPosCallback(mouseCallback)
	window.SetMouseButtonCallback(mouseButtonCallback)
	window.SetScrollCallback(scrollCallback)

	window.SetFramebufferSizeCallback(func(w *glfw.Window, width, height int) {
		if width <= 0 || height <= 0 {
			return
		}
		currentWidth = width
		currentHeight = height
		gl.Viewport(0, 0, int32(width), int32(height))
	})

	camera.UpdateVectors()

	prev := time.Now()
	for !window.ShouldClose() {
		now := time.Now()
		dt := now.Sub(prev).Seconds()
		prev = now

		processInput(window, float32(dt))
		updateParticlesPerformant(dt)
		render()
		window.SwapBuffers()
		glfw.PollEvents()
	}
}

// -----------------------------
// Spirograph / Sphere Centers
// -----------------------------

func updateSphereCenters(dt float64) {
	timeAccumulator += float32(dt)
	t := timeAccumulator * spiroSpeed

	R := spiroR
	k := spiroK
	l := spiroL
	ratioTerm := (1.0 - k) / k

	term1_x := R * (1.0 - k) * float32(math.Cos(float64(t)))
	term1_y := R * (1.0 - k) * float32(math.Sin(float64(t)))
	term2_x := R * l * k * float32(math.Cos(float64(ratioTerm*t)))
	term2_y := R * l * k * float32(math.Sin(float64(ratioTerm*t)))

	x_base := term1_x + term2_x
	y_base := term1_y - term2_y

	z_osc1 := R * 0.2 * float32(math.Sin(float64(t*0.5)))
	z_osc2 := R * 0.2 * float32(math.Cos(float64(t*0.5)))
	xy_osc := R * 0.1 * float32(math.Sin(float64(t)))

	xA := x_base + xy_osc
	yA := y_base
	zA := z_osc1
	simData.Coms[0] = mgl32.Vec3{xA, yA, zA}

	xB := -x_base
	yB := -y_base + xy_osc
	zB := -z_osc2
	simData.Coms[1] = mgl32.Vec3{xB, yB, zB}

	xC := R * 0.6 * float32(math.Cos(float64(t*1.5)))
	yC := R * 0.6 * float32(math.Sin(float64(t*1.5)))
	zC := R * 0.4 * float32(math.Sin(float64(t*0.7)))
	simData.Coms[2] = mgl32.Vec3{xC, yC, zC}
}

// -----------------------------
// Update Loop / Multi-threaded
// -----------------------------

func updateParticlesPerformant(dt float64) {
	dt32 := float32(dt)
	dampingMultiplier := float32(1.0 - tdaDamping)

	updateSphereCenters(dt)

	actualCOMs := [numSpheres]mgl32.Vec3{}
	TopologicalAnalysisModule(simData.Particles, simData.ParticleID, actualCOMs[:], simData.Connected[:], simData.Beta0[:])

	sh := NewSpatialHash(simData.Particles, ddgRadius)

	workerResults := make([][numSpheres]int, numWorkers)
	var wg sync.WaitGroup

	chunkSize := nParticles / numWorkers
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)

		start := w * chunkSize
		end := start + chunkSize
		if w == numWorkers-1 {
			end = nParticles
		}

		go func(start, end int, resultPtr *[numSpheres]int) {
			defer wg.Done()

			localBetaOneActiveCounts := [numSpheres]int{0, 0, 0}

			for i := start; i < end; i++ {
				p := &simData.Particles[i]
				pos := p.Pos
				sphereID := simData.ParticleID[i]

				targetCenter := simData.Coms[sphereID]
				isConnected := simData.Connected[sphereID]

				ddgForce, betaOneForce, betaOneActive := getDDGAndBetaOneForce(i, pos, sh, simData.Particles)
				if betaOneActive {
					localBetaOneActiveCounts[sphereID]++
				}

				centerForce := getSphereSpringForce(pos, targetCenter, sphereRadius)

				tdaForce := mgl32.Vec3{0, 0, 0}
				if !isConnected {
					actualCOM := actualCOMs[sphereID]
					dir := actualCOM.Sub(pos)
					if dir.Len() > 0.001 {
						// reuse normalized dir
						distToCOM := dir.Len()
						dir = dir.Normalize()
						tdaForce = dir.Mul(tdaRestoreForce * (1.0 + distToCOM/sphereRadius))
					}
				}

				repulsionForce := mgl32.Vec3{0, 0, 0}
				for otherID := 0; otherID < numSpheres; otherID++ {
					if otherID != sphereID {
						otherCenter := simData.Coms[otherID]
						repulsionForce = repulsionForce.Add(getInterSphereRepulsion(pos, otherCenter))
					}
				}

				totalAcc := ddgForce.Add(centerForce).Add(tdaForce).Add(repulsionForce).Add(betaOneForce)

				p.Vel = p.Vel.Add(totalAcc.Mul(dt32))
				p.Vel = p.Vel.Mul(dampingMultiplier)
				p.Pos = p.Pos.Add(p.Vel.Mul(dt32))
			}
			*resultPtr = localBetaOneActiveCounts
		}(start, end, &workerResults[w])
	}

	wg.Wait()

	betaOneActiveCounts := [numSpheres]int{0, 0, 0}
	for _, res := range workerResults {
		for id := 0; id < numSpheres; id++ {
			betaOneActiveCounts[id] += res[id]
		}
	}

	for id := 0; id < numSpheres; id++ {
		totalParticlesInSphere := nParticles / numSpheres
		if float32(betaOneActiveCounts[id]) > float32(totalParticlesInSphere)*betaOneParticleRatio {
			simData.Beta1[id] = 1
		} else {
			simData.Beta1[id] = 0
		}
	}

	logTimer += dt32
	if logTimer >= logIntervalSeconds {
		logTimer = 0
		logPoincarePolynomials()
	}
}

func logPoincarePolynomials() {
	colors := []string{"Red", "Blue", "Green"}
	log.Printf("--- Topological Signature (Poincaré Polynomial P(x) = b₀ + b₁x) ---")

	for i := 0; i < numSpheres; i++ {
		b0 := simData.Beta0[i]
		b1 := simData.Beta1[i]

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

		log.Printf("[%s Sphere] P(x) = %s  | Status: %s", colors[i], poly, status)
	}
}

// -----------------------------
// Forces
// -----------------------------

// Precompute ddgTargetDist once per call; avoid repeated sqrt in loops
func getDDGAndBetaOneForce(index int, pos mgl32.Vec3, sh *SpatialHash, data []Particle) (mgl32.Vec3, mgl32.Vec3, bool) {
	neighbors := sh.FindNeighborsSH(index, pos, data, ddgRadius)
	numNeighbors := len(neighbors)

	if numNeighbors == 0 {
		return mgl32.Vec3{0, 0, 0}, mgl32.Vec3{0, 0, 0}, false
	}

	totalForce := mgl32.Vec3{0, 0, 0}
	targetDist := float32(ddgRadius) / float32(math.Sqrt(float64(kNeighbors))) * 0.7

	for _, neighborIndex := range neighbors {
		neighborPos := data[neighborIndex].Pos
		vec := neighborPos.Sub(pos)
		dist := vec.Len()
		forceMag := float32(ddgSpringK) * (dist - targetDist)

		if dist > 0.001 {
			dir := vec.Normalize()
			totalForce = totalForce.Add(dir.Mul(forceMag * 0.5))
		}
	}
	ddgForce := totalForce.Mul(1.0 / float32(numNeighbors))

	betaOneForce := mgl32.Vec3{0, 0, 0}
	betaOneActive := false

	if numNeighbors < kNeighbors-tdaBetaOneThreshold {
		sumPos := mgl32.Vec3{0, 0, 0}
		for _, nIndex := range neighbors {
			sumPos = sumPos.Add(data[nIndex].Pos)
		}
		CoN := sumPos.Mul(1.0 / float32(numNeighbors))
		dir := CoN.Sub(pos)
		if dir.Len() > 0.001 {
			dir = dir.Normalize()
			betaOneForce = dir.Mul(float32(tdaBetaOneForceK))
			betaOneActive = true
		}
	}

	return ddgForce, betaOneForce, betaOneActive
}

func getSphereSpringForce(pos mgl32.Vec3, center mgl32.Vec3, targetRadius float32) mgl32.Vec3 {
	vec := pos.Sub(center)
	dist := vec.Len()

	if dist < 0.001 {
		return mgl32.Vec3{0, 0, 0}
	}

	dir := vec.Normalize()
	force := float32(springStrength) * (dist - targetRadius)
	return dir.Mul(-force)
}

func getInterSphereRepulsion(pos mgl32.Vec3, otherCenter mgl32.Vec3) mgl32.Vec3 {
	vec := pos.Sub(otherCenter)
	distSq := vec.LenSqr()

	if distSq > repulsionDistCheck*repulsionDistCheck {
		return mgl32.Vec3{0, 0, 0}
	}
	if distSq < 1e-6 {
		// small jitter to avoid divide by zero
		return mgl32.Vec3{0, 0, 0}
	}
	forceMag := repulsionK / (distSq)
	dir := vec.Normalize()
	return dir.Mul(forceMag)
}

// -----------------------------
// Topological Analysis (Betti-0)
// -----------------------------

func TopologicalAnalysisModule(data []Particle, ids []int, coms []mgl32.Vec3, connected []bool, beta0 []int) {
	groups := make([][]int, numSpheres)
	for i, id := range ids {
		if id >= 0 && id < numSpheres {
			groups[id] = append(groups[id], i)
		}
	}

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

	radiusSq := 0.25 * tdaClusterRadius * tdaClusterRadius

	for sphereID := 0; sphereID < numSpheres; sphereID++ {
		group := groups[sphereID]
		if len(group) <= 1 {
			connected[sphereID] = true
			beta0[sphereID] = 1
			continue
		}

		indexMap := make(map[int]int, len(group))
		for localIdx, globalIdx := range group {
			indexMap[globalIdx] = localIdx
		}

		uf := NewUnionFind(len(group))

		for i := 0; i < len(group); i++ {
			p1GlobalIdx := group[i]
			p1Pos := data[p1GlobalIdx].Pos

			for j := i + 1; j < len(group); j++ {
				p2GlobalIdx := group[j]
				p2Pos := data[p2GlobalIdx].Pos

				if p1Pos.Sub(p2Pos).LenSqr() < float32(radiusSq) {
					uf.Union(indexMap[p1GlobalIdx], indexMap[p2GlobalIdx])
				}
			}
		}

		// Efficiently count unique roots using Find
		rootSeen := make(map[int]struct{}, len(group))
		for i := 0; i < len(group); i++ {
			root := uf.Find(i)
			rootSeen[root] = struct{}{}
		}
		numComponents := len(rootSeen)
		beta0[sphereID] = numComponents
		connected[sphereID] = (numComponents <= 1)
	}
}

// -----------------------------
// Rendering
// -----------------------------

func setupBuffers() {
	gl.GenVertexArrays(1, &vao)
	gl.BindVertexArray(vao)
	gl.GenBuffers(1, &vbo)
	gl.BindBuffer(gl.ARRAY_BUFFER, vbo)
	bufferSize := nParticles * particleVboSize
	gl.BufferData(gl.ARRAY_BUFFER, bufferSize, nil, gl.DYNAMIC_DRAW)
	stride := int32(particleVboSize)

	// location 0: vec3 position (offset 0)
	gl.EnableVertexAttribArray(0)
	gl.VertexAttribPointer(0, 3, gl.FLOAT, false, stride, gl.PtrOffset(0))

	// location 1: vec3 color (offset 3*4)
	gl.EnableVertexAttribArray(1)
	gl.VertexAttribPointer(1, 3, gl.FLOAT, false, stride, gl.PtrOffset(3*4))

	// location 2: vec3 velocity (offset 6*4)
	gl.EnableVertexAttribArray(2)
	gl.VertexAttribPointer(2, 3, gl.FLOAT, false, stride, gl.PtrOffset(6*4))

	gl.BindVertexArray(0)
}

func render() {
	// --- 1. Fade previous frame (for smooth trails) ---
	gl.Disable(gl.DEPTH_TEST)
	gl.Enable(gl.BLEND)
	gl.BlendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA)

	// Draw a translucent black overlay instead of clearing
	// NOTE: deprecated immediate mode used in original; kept for parity

	gl.Enable(gl.DEPTH_TEST)

	// --- 2. Prepare rendering state ---
	gl.Clear(gl.DEPTH_BUFFER_BIT) // only clear depth (not color!)
	gl.UseProgram(prog)

	aspect := float32(currentWidth) / float32(currentHeight)
	proj := mgl32.Perspective(mgl32.DegToRad(45.0), aspect, 0.1, 5000.0)
	view := mgl32.LookAtV(camera.Position, camera.Position.Add(camera.Front), camera.Up)
	vp := proj.Mul4(view)

	loc := gl.GetUniformLocation(prog, gl.Str("uVP\x00"))
	gl.UniformMatrix4fv(loc, 1, false, &vp[0])

	locCam := gl.GetUniformLocation(prog, gl.Str("uCameraPos\x00"))
	gl.Uniform3f(locCam, camera.Position.X(), camera.Position.Y(), camera.Position.Z())

	locPointScale := gl.GetUniformLocation(prog, gl.Str("uPointScale\x00"))
	locFogColor := gl.GetUniformLocation(prog, gl.Str("uFogColor\x00"))
	gl.Uniform3f(locFogColor, 0.03, 0.03, 0.05)

	locFogDensity := gl.GetUniformLocation(prog, gl.Str("uFogDensity\x00"))
	gl.Uniform1f(locFogDensity, 0.0015)

	locTime := gl.GetUniformLocation(prog, gl.Str("uTime\x00"))
	gl.Uniform1f(locTime, float32(timeAccumulator))

	// --- 3. Build particle data (pos, color, vel) into preallocated buffer ---
	// This avoids repeated appends and allocations.
	buf := simData.VBOFloatBuf
	// buf length expected = nParticles * 9
	// Use index writes for speed
	idx := 0
	for i := 0; i < nParticles; i++ {
		p := &simData.Particles[i]
		// position
		buf[idx] = p.Pos.X()
		buf[idx+1] = p.Pos.Y()
		buf[idx+2] = p.Pos.Z()
		// color (brighter if Beta1 active)
		col := p.Col
		sphereID := simData.ParticleID[i]
		if simData.Beta1[sphereID] > 0 {
			// clamp multipliers inline
			r := float32(math.Min(float64(col.X()*1.6), 1.0))
			g := float32(math.Min(float64(col.Y()*1.6), 1.0))
			b := float32(math.Min(float64(col.Z()*1.6), 1.0))
			buf[idx+3] = r
			buf[idx+4] = g
			buf[idx+5] = b
		} else {
			buf[idx+3] = col.X()
			buf[idx+4] = col.Y()
			buf[idx+5] = col.Z()
		}
		// velocity
		buf[idx+6] = p.Vel.X()
		buf[idx+7] = p.Vel.Y()
		buf[idx+8] = p.Vel.Z()
		idx += 9
	}

	gl.BindBuffer(gl.ARRAY_BUFFER, vbo)
	dataSize := len(buf) * 4
	gl.BufferSubData(gl.ARRAY_BUFFER, 0, dataSize, gl.Ptr(buf))
	gl.BindVertexArray(vao)

	// --- 4. Two-pass glow rendering ---
	gl.Enable(gl.BLEND)

	// Pass 1: halo (additive blending, larger size)
	gl.BlendFunc(gl.SRC_ALPHA, gl.ONE)
	gl.Uniform1f(locPointScale, 24000.0)
	gl.DrawArrays(gl.POINTS, 0, int32(nParticles))

	// Pass 2: core (standard alpha blending)
	gl.BlendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA)
	gl.Uniform1f(locPointScale, 16000.0)
	gl.DrawArrays(gl.POINTS, 0, int32(nParticles))

	gl.BindVertexArray(0)
}

// -----------------------------
// Shader compilation / sources
// -----------------------------

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

var vertexShader = `
#version 410 core
layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inCol;
layout(location = 2) in vec3 inVel;

uniform mat4 uVP;
uniform vec3 uCameraPos;
uniform float uPointScale;
uniform float uTime;

out vec3 vColor;
out float vIntensity;
out float vDistance;

void main() {
    gl_Position = uVP * vec4(inPos, 1.0);
    vColor = inCol;

    vIntensity = clamp(length(inVel), 0.0, 20.0);
    vDistance = length(inPos - uCameraPos);

    float size = uPointScale / (0.001 + vDistance);
    size = clamp(size, 2.0, 64.0);
    gl_PointSize = size;
}
` + "\x00"

var fragmentShader = `
#version 410 core
in vec3 vColor;
in float vIntensity;
in float vDistance;

out vec4 fragColor;

uniform vec3 uFogColor;
uniform float uFogDensity;
uniform float uTime;

void main() {
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);
    if (dist > 0.5) discard;

    float alpha = smoothstep(0.5, 0.0, dist);

    float bright = 0.5 + smoothstep(0.0, 10.0, vIntensity) * 1.2;

    float pulse = 0.9 + 0.12 * sin(uTime*3.0 + vDistance*0.01);

    vec3 base = vColor * bright * pulse;

    float fogFactor = exp(-uFogDensity * uFogDensity * vDistance * vDistance);
    fogFactor = clamp(fogFactor, 0.0, 1.0);

    vec3 col = mix(uFogColor, base, fogFactor);

    fragColor = vec4(col, alpha * 0.9);
}
` + "\x00"
