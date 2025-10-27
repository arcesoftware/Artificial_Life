package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"runtime"
	"sync" // Required for parallel processing
	"time"
	"unsafe" // Required for efficient GPU buffer updates

	"github.com/go-gl/gl/v4.1-core/gl"
	"github.com/go-gl/glfw/v3.3/glfw"
	"github.com/go-gl/mathgl/mgl32"
)

// ----------------------------------------------------
// ## Performance Optimization Structures (Spatial Hashing)
// ----------------------------------------------------

// HashBucket holds indices of particles residing in a grid cell.
type HashBucket []int

// SpatialHash is a uniform grid structure for O(N) average time neighbor finding.
type SpatialHash struct {
	Grid        map[int]HashBucket
	CellSize    float32 // Must be >= max interaction radius (ddgRadius)
	InvCellSize float32
}

// NewSpatialHash initializes and populates the hash map.
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
	// Quantize position to grid coordinates (ix, iy, iz)
	ix := int(pos.X() * sh.InvCellSize)
	iy := int(pos.Y() * sh.InvCellSize)
	iz := int(pos.Z() * sh.InvCellSize)

	// Combine 3D coordinates into a single hash key using large prime multipliers
	return ix*73856093 + iy*1934983 + iz*83492791
}

// Update rebuilds the hash map.
func (sh *SpatialHash) Update(particles []Particle) {
	// Clear and reallocate the map
	sh.Grid = make(map[int]HashBucket, len(particles))
	for i, p := range particles {
		key := sh.HashIndex(p.Pos)
		sh.Grid[key] = append(sh.Grid[key], i)
	}
}

// FindNeighborsSH uses the Spatial Hash to only check the 27 adjacent cells. (O(1) average time per query)
func (sh *SpatialHash) FindNeighborsSH(index int, pos mgl32.Vec3, data []Particle, radius float32) []int {
	neighbors := []int{}
	radiusSq := radius * radius

	// Find the cell containing the current particle
	ix0 := int(pos.X() * sh.InvCellSize)
	iy0 := int(pos.Y() * sh.InvCellSize)
	iz0 := int(pos.Z() * sh.InvCellSize)

	// Iterate through the 3x3x3 neighboring cells (27 cells total)
	for dx := -1; dx <= 1; dx++ {
		for dy := -1; dy <= 1; dy++ {
			for dz := -1; dz <= 1; dz++ {

				// Hash the neighboring cell's coordinates
				key := (ix0+dx)*73856093 + (iy0+dy)*1934983 + (iz0+dz)*83492791

				if bucket, ok := sh.Grid[key]; ok {
					for _, otherIndex := range bucket {
						if otherIndex == index {
							continue
						}
						// Final distance check is still required
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
// ## Simulation Data and Constants
// ----------------------------------------------------

type SimData struct {
	Particles []Particle
	// NeighborTree is now handled transiently by SpatialHash in the update loop
}

const (
	winWidth  = 1280
	winHeight = 720
	nParticles = 4500 // 10x scale for performance testing
	pointSize = 1.0

	// Performance / Parallelism
	particleVboSize = 6 * 4
	numWorkers      = 8 // Use 8 goroutines (adjust based on CPU cores)

	// DDG Constants
	kNeighbors = 8
	ddgRadius  = 30.0  // Must be the same as SpatialHash.CellSize
	ddgSpringK = 150.0

	// TDA Constants
	tdaClusterRadius = 150.0
	tdaRestoreForce  = 500.0
	tdaDamping       = 0.05
)

type Particle struct {
	Pos mgl32.Vec3
	Vel mgl32.Vec3
	Col mgl32.Vec3
}

var (
	simData SimData

	prog uint32
	vao  uint32
	vbo  uint32

	azimuth, elevation float64 = 0.6, 0.2
	distance           float64 = 900
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

	window, err := glfw.CreateWindow(winWidth, winHeight, "DDG/TDA Particles - Large N", nil, nil)
	if err != nil {
		panic(err)
	}
	window.MakeContextCurrent()
	if err := gl.Init(); err != nil {
		panic(err)
	}

	fmt.Println("OpenGL version", gl.GoStr(gl.GetString(gl.VERSION)))
	gl.Enable(gl.DEPTH_TEST)
	gl.Enable(gl.BLEND)
	gl.BlendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA)
	gl.PointSize(pointSize)

	prog, err = newProgram(vertexShader, fragmentShader)
	if err != nil {
		panic(err)
	}

	// --- Particle Initialization ---
	particles := make([]Particle, nParticles)
	for i := 0; i < nParticles; i++ {
		// Initialize particles within a small volume
		dist := float32(rand.Float64() * 100)
		angle1 := float32(rand.Float64() * 2 * math.Pi)
		angle2 := float32(rand.Float64() * math.Pi)
		particles[i].Pos = mgl32.Vec3{
			dist * float32(math.Cos(float64(angle1))) * float32(math.Sin(float64(angle2))),
			dist * float32(math.Sin(float64(angle1))) * float32(math.Sin(float64(angle2))),
			dist * float32(math.Cos(float64(angle2))),
		}
		particles[i].Vel = mgl32.Vec3{0, 0, 0}
		r := float32(rand.Float64()*0.6 + 0.4)
		g := float32(rand.Float64()*0.6 + 0.1)
		b := float32(rand.Float64() * 0.3)
		particles[i].Col = mgl32.Vec3{r, g, b}
	}

	simData.Particles = particles // Assign to global simData

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
// ## High-Performance Dynamics Loop
// ----------------------------------------------------

func updateParticlesPerformant(dt float64) {
	dt32 := float32(dt)
	dampingMultiplier := float32(1.0 - tdaDamping)

	// Step 1: TDA/Topological Analysis (O(N) for COM, O(N*k) for connectivity proxy)
	// Must be done before forces are calculated.
	centerOfMass, isConnected := TopologicalAnalysisModule(simData.Particles)

	// Step 2: Rebuild Spatial Hash (O(N) time)
	// The DDG radius (30.0) is the largest interaction radius.
	sh := NewSpatialHash(simData.Particles, ddgRadius)

	// -----------------------------------------------------------------
	// Step 3: DDG/TDA Physics Integration (Parallelized)
	// -----------------------------------------------------------------

	var wg sync.WaitGroup
	// Determine the range of particles each worker will handle
	chunkSize := nParticles / numWorkers

	for w := 0; w < numWorkers; w++ {
		wg.Add(1)

		start := w * chunkSize
		end := start + chunkSize
		if w == numWorkers-1 {
			end = nParticles // Ensure the last worker gets the remainder
		}

		go func(start, end int) {
			defer wg.Done()

			// Process particles in this worker's assigned range
			for i := start; i < end; i++ {
				p := &simData.Particles[i]
				pos := p.Pos // Use particle's current position for hash lookup

				// 1. DDG Force (O(1) average lookup using Spatial Hash)
				ddgForce := DDGModuleSH(i, pos, sh, simData.Particles)

				// 2. Spring Force (Global containment - O(1))
				centerForce := getCenterSpringForce(pos)

				// 3. TDA Restoring Force (Topological Constraint - O(1))
				tdaForce := mgl32.Vec3{0, 0, 0}
				if !isConnected {
					dir := centerOfMass.Sub(pos).Normalize()
					tdaForce = dir.Mul(tdaRestoreForce)
				}

				// Combined Acceleration (m=1)
				totalAcc := ddgForce.Add(centerForce).Add(tdaForce)

				// Velocity and Position Update
				p.Vel = p.Vel.Add(totalAcc.Mul(dt32))
				p.Vel = p.Vel.Mul(dampingMultiplier)
				p.Pos = p.Pos.Add(p.Vel.Mul(dt32))
			}
		}(start, end)
	}

	wg.Wait() // Block until all workers are done
}

// ----------------------------------------------------
// ## DDG Module (Discrete Curvature Flow Proxy)
// ----------------------------------------------------

// DDGModuleSH uses the efficient Spatial Hash for neighbor finding.
func DDGModuleSH(index int, pos mgl32.Vec3, sh *SpatialHash, data []Particle) mgl32.Vec3 {
	neighbors := sh.FindNeighborsSH(index, pos, data, ddgRadius)
	if len(neighbors) == 0 {
		return mgl32.Vec3{0, 0, 0}
	}

	totalForce := mgl32.Vec3{0, 0, 0}
	// Target distance approximation based on neighbor count and radius
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

	// Normalize by neighbor count to average the flow direction
	return totalForce.Mul(1.0 / float32(len(neighbors)))
}

// getCenterSpringForce provides the global, extrinsic confinement force.
func getCenterSpringForce(pos mgl32.Vec3) mgl32.Vec3 {
	const sphereRadius = 150.0
	const springStrength = 100.0

	dist := float64(pos.Len())
	if dist < 0.001 {
		return mgl32.Vec3{0, 0, 0}
	}

	dir := pos.Normalize()
	force := springStrength * (dist - sphereRadius)
	return dir.Mul(-float32(force)) // force is always directed toward the sphere center
}

// ----------------------------------------------------
// ## TDA Module (Persistent Homology Proxy for Betti-0)
// ----------------------------------------------------

func TopologicalAnalysisModule(data []Particle) (centerOfMass mgl32.Vec3, isConnected bool) {
	if len(data) == 0 {
		return mgl32.Vec3{0, 0, 0}, true
	}

	// Calculate Center of Mass (O(N))
	com := mgl32.Vec3{0, 0, 0}
	for _, p := range data {
		com = com.Add(p.Pos)
	}
	centerOfMass = com.Mul(1.0 / float32(len(data)))

	// Connected Components Check (Proxy for Betti-0) - O(N^2) worst case, but O(N*k) average time where k is number of neighbors
	visited := make([]bool, len(data))
	numComponents := 0

	for i := range data {
		if !visited[i] {
			numComponents++
			// Breadth-First Search (BFS) for connectivity
			queue := []int{i}
			visited[i] = true

			for len(queue) > 0 {
				curr := queue[0]
				queue = queue[1:]

				for j := range data {
					// Check neighbor connectivity only if not visited
					if !visited[j] && data[curr].Pos.Sub(data[j].Pos).LenSqr() < tdaClusterRadius*tdaClusterRadius {
						visited[j] = true
						queue = append(queue, j)
					}
				}
			}
		}
	}

	// If numComponents > 1, the single-component topology is broken.
	return centerOfMass, numComponents <= 1
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
	proj := mgl32.Perspective(mgl32.DegToRad(45), float32(winWidth)/winHeight, 0.1, 5000)
	view := cameraMatrix()
	vp := proj.Mul4(view)
	loc := gl.GetUniformLocation(prog, gl.Str("uVP\x00"))
	gl.UniformMatrix4fv(loc, 1, false, &vp[0])

	// Prepare interleaved data for VBO update (O(N) time)
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

func cameraMatrix() mgl32.Mat4 {
	x := float32(distance * math.Cos(azimuth) * math.Cos(elevation))
	y := float32(distance * math.Sin(elevation))
	z := float32(distance * math.Sin(azimuth) * math.Cos(elevation))
	return mgl32.LookAtV(mgl32.Vec3{x, y, z}, mgl32.Vec3{0, 0, 0}, mgl32.Vec3{0, 1, 0})
}

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

// compileShader and newProgram are essential for OpenGL setup.
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
