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
	ix := int(pos.X() * sh.InvCellSize)
	iy := int(pos.Y() * sh.InvCellSize)
	iz := int(pos.Z() * sh.InvCellSize)
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

	ix0 := int(pos.X() * sh.InvCellSize)
	iy0 := int(pos.Y() * sh.InvCellSize)
	iz0 := int(pos.Z() * sh.InvCellSize)

	for dx := -1; dx <= 1; dx++ {
		for dy := -1; dy <= 1; dy++ {
			for dz := -1; dz <= 1; dz++ {
				key := (ix0+dx)*73856093 + (iy0+dy)*1934983 + (iz0+dz)*83492791

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

// ----------------------------------------------------
// ## Topological Analysis Structure (Union-Find for Betti-0)
// ----------------------------------------------------

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
	if uf.Parent[i] == i {
		return i
	}
	uf.Parent[i] = uf.Find(uf.Parent[i])
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

// ----------------------------------------------------
// ## Simulation Data and Constants
// ----------------------------------------------------

type SimData struct {
	Particles []Particle
	Coms      [numSpheres]mgl32.Vec3
	Beta0     [numSpheres]int
	Beta1     [numSpheres]int
	Connected [numSpheres]bool
	ParticleID []int
}

const (
	winWidth   = 1280
	winHeight  = 720
	nParticles = 2400
	pointSize  = 1.5

	particleVboSize = 6 * 4 // 6 floats * 4 bytes
	numWorkers      = 8

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

type Particle struct {
	Pos mgl32.Vec3
	Vel mgl32.Vec3
	Col mgl32.Vec3
}

var (
	simData         SimData
	timeAccumulator float32
	logTimer        float32

	prog uint32
	vao  uint32
	vbo  uint32

	// Camera (free-fly, Blender-like)
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

	// mouse/cursor state
	lastX, lastY float64
	dragging     bool

	// framebuffer size
	currentWidth  = winWidth
	currentHeight = winHeight
)

func init() {
	runtime.LockOSThread()
}

// ----------------------------------------------------
// Camera implementation
// ----------------------------------------------------

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
	yawRad := mgl32.DegToRad(float32(cam.Yaw))
	pitchRad := mgl32.DegToRad(float32(cam.Pitch))
	front := mgl32.Vec3{
		float32(math.Cos(float64(yawRad)) * math.Cos(float64(pitchRad))),
		float32(math.Sin(float64(pitchRad))),
		float32(math.Sin(float64(yawRad)) * math.Cos(float64(pitchRad))),
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
}

func mouseCallback(w *glfw.Window, xpos, ypos float64) {
	if !dragging {
		lastX = xpos
		lastY = ypos
		return
	}

	xoffset := xpos - lastX
	yoffset := lastY - ypos // reversed (screen coords)
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
		// When starting dragging, hide and capture cursor (already set to disabled input mode)
	}
}

func scrollCallback(w *glfw.Window, xoff, yoff float64) {
	// Move camera forward/back along its front vector (acts as zoom)
	camera.Position = camera.Position.Add(camera.Front.Mul(float32(yoff) * 30.0))
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

	window, err := glfw.CreateWindow(winWidth, winHeight, "Tri-Sphere TDA Dynamics - FreeFly Camera", nil, nil)
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
	gl.BlendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA)
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

	setupBuffers()

	// Input & callbacks
	// Hide cursor and capture for free-fly camera (like Blender viewport when in fly mode)
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

	// initialize camera vectors
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

// ----------------------------------------------------
// ## Spirograph Kinematics (Three-Body Entanglement)
// ----------------------------------------------------

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

// ----------------------------------------------------
// ## High-Performance Dynamics Loop (The Calculation)
// ----------------------------------------------------

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
						dir = dir.Normalize()
						distToCOM := actualCOM.Sub(pos).Len()
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

// ----------------------------------------------------
// ## Force Modules (DDG & Betti-1)
// ----------------------------------------------------

func getDDGAndBetaOneForce(index int, pos mgl32.Vec3, sh *SpatialHash, data []Particle) (ddgForce mgl32.Vec3, betaOneForce mgl32.Vec3, betaOneActive bool) {
	neighbors := sh.FindNeighborsSH(index, pos, data, ddgRadius)
	numNeighbors := len(neighbors)

	if numNeighbors == 0 {
		return mgl32.Vec3{0, 0, 0}, mgl32.Vec3{0, 0, 0}, false
	}

	totalForce := mgl32.Vec3{0, 0, 0}
	targetDist := ddgRadius / float32(math.Sqrt(float64(kNeighbors))) * 0.7

	for _, neighborIndex := range neighbors {
		neighborPos := data[neighborIndex].Pos
		vec := neighborPos.Sub(pos)
		dist := vec.Len()
		forceMag := ddgSpringK * (dist - targetDist)

		if dist > 0.001 {
			dir := vec.Normalize()
			totalForce = totalForce.Add(dir.Mul(forceMag * 0.5))
		}
	}
	ddgForce = totalForce.Mul(1.0 / float32(numNeighbors))

	betaOneForce = mgl32.Vec3{0, 0, 0}
	betaOneActive = false

	if numNeighbors < kNeighbors-tdaBetaOneThreshold {
		sumPos := mgl32.Vec3{0, 0, 0}
		for _, nIndex := range neighbors {
			sumPos = sumPos.Add(data[nIndex].Pos)
		}
		CoN := sumPos.Mul(1.0 / float32(numNeighbors))
		dir := CoN.Sub(pos)
		if dir.Len() > 0.001 {
			dir = dir.Normalize()
			betaOneForce = dir.Mul(tdaBetaOneForceK)
			betaOneActive = true
		}
	}

	return ddgForce, betaOneForce, betaOneActive
}

func getSphereSpringForce(pos mgl32.Vec3, center mgl32.Vec3, targetRadius float32) mgl32.Vec3 {
	vec := pos.Sub(center)
	dist := float64(vec.Len())

	if dist < 0.001 {
		return mgl32.Vec3{0, 0, 0}
	}

	dir := vec.Normalize()
	force := springStrength * (dist - float64(targetRadius))
	return dir.Mul(-float32(force))
}

func getInterSphereRepulsion(pos mgl32.Vec3, otherCenter mgl32.Vec3) mgl32.Vec3 {
	vec := pos.Sub(otherCenter)
	dist := vec.Len()

	if dist > repulsionDistCheck {
		return mgl32.Vec3{0, 0, 0}
	}

	forceMag := repulsionK / (dist * dist)
	dir := vec.Normalize()

	return dir.Mul(forceMag)
}

// ----------------------------------------------------
// ## TDA Module (Betti-0: Connected Components)
// ----------------------------------------------------

func TopologicalAnalysisModule(data []Particle, ids []int, coms []mgl32.Vec3, connected []bool, beta0 []int) {
	groups := make([][]int, numSpheres)
	for i, id := range ids {
		groups[id] = append(groups[id], i)
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

	radiusSq := float32(0.25) * tdaClusterRadius * tdaClusterRadius

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

				if p1Pos.Sub(p2Pos).LenSqr() < radiusSq {
					uf.Union(indexMap[p1GlobalIdx], indexMap[p2GlobalIdx])
				}
			}
		}

		numComponents := 0
		for i := 0; i < len(group); i++ {
			if uf.Parent[i] == i {
				numComponents++
			}
		}
		beta0[sphereID] = numComponents
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
	bufferSize := nParticles * particleVboSize
	gl.BufferData(gl.ARRAY_BUFFER, bufferSize, nil, gl.DYNAMIC_DRAW)
	stride := int32(particleVboSize)

	gl.EnableVertexAttribArray(0)
	gl.VertexAttribPointer(0, 3, gl.FLOAT, false, stride, gl.PtrOffset(0))
	gl.EnableVertexAttribArray(1)
	gl.VertexAttribPointer(1, 3, gl.FLOAT, false, stride, gl.PtrOffset(3*4))

	gl.BindVertexArray(0)
}

func render() {
	gl.ClearColor(0, 0, 0, 1)
	gl.Clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)
	gl.UseProgram(prog)

	aspect := float32(currentWidth) / float32(currentHeight)
	proj := mgl32.Perspective(mgl32.DegToRad(45.0), aspect, 0.1, 5000.0)
	view := mgl32.LookAtV(camera.Position, camera.Position.Add(camera.Front), camera.Up)
	vp := proj.Mul4(view)
	loc := gl.GetUniformLocation(prog, gl.Str("uVP\x00"))
	gl.UniformMatrix4fv(loc, 1, false, &vp[0])

	vboData := make([]float32, 0, nParticles*6)
	for _, p := range simData.Particles {
		vboData = append(vboData, p.Pos.X(), p.Pos.Y(), p.Pos.Z())
		vboData = append(vboData, p.Col.X(), p.Col.Y(), p.Col.Z())
	}

	gl.BindBuffer(gl.ARRAY_BUFFER, vbo)
	dataSize := len(vboData) * 4
	gl.BufferSubData(gl.ARRAY_BUFFER, 0, dataSize, gl.Ptr(vboData))

	gl.BindVertexArray(vao)
	gl.DrawArrays(gl.POINTS, 0, int32(nParticles))
	gl.BindVertexArray(0)
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
