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
// ## Simulation Data and Constants
// ----------------------------------------------------

type SimData struct {
	Particles []Particle
	// Coms[i] stores the KINEMATIC TARGET CENTER for sphere i
	Coms [2]mgl32.Vec3
	// Connected[i] stores the TDA connectivity status for sphere i
	Connected [2]bool

	// ParticleID[i] stores 0 for sphere A, 1 for sphere B
	ParticleID []int
}

const (
	winWidth   = 1280
	winHeight  = 720
	nParticles = 1618
	pointSize  = 1.5

	// Performance / Parallelism
	particleVboSize = 6 * 4
	numWorkers      = 8

	// DDG Constants
	kNeighbors = 4
	ddgRadius  = 30.0
	ddgSpringK = 150.0

	// TDA Constants
	tdaClusterRadius = 150.0
	tdaRestoreForce  = 500.0
	tdaDamping       = 0.618033989 // High damping for quick convergence to the shape

	// Confinement
	sphereRadius   = 150.0 // The radius the spring force targets
	springStrength = 22.0  // Softer spring to reduce initial bounce

	// --- Spirograph Kinematic Parameters ---
	spiroR     float32 = 180.0 // Outer circle radius (Scaling factor)
	spiroK     float32 = 0.531 // k = r/R (Ratio of inner circle radius to outer)
	spiroL     float32 = 0.854 // l = rho/r (Ratio of pen distance to inner radius)
	spiroSpeed float32 = 0.8   // Speed multiplier for the parameter 't'

	// Inter-Sphere Repulsion
	repulsionK         float32 = 80.0
	repulsionDistCheck         = 250.0
)

type Particle struct {
	Pos mgl32.Vec3
	Vel mgl32.Vec3
	Col mgl32.Vec3
}

var (
	simData         SimData
	timeAccumulator float32 // Tracks total elapsed simulation time

	prog uint32
	vao  uint32
	vbo  uint32

	azimuth, elevation float64 = 0.6, 0.2
	distance           float64 = 700 // Zoomed out for the figure
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

	window, err := glfw.CreateWindow(winWidth, winHeight, "Dynamic 3D Spirograph Particles", nil, nil)
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

	// --- Particle Initialization (Dual Spheres, started near origin) ---
	particles := make([]Particle, nParticles)
	simData.ParticleID = make([]int, nParticles)
	initialCenter := mgl32.Vec3{0, 0, 0}

	for i := 0; i < nParticles; i++ {
		sphereID := i % 2
		simData.ParticleID[i] = sphereID

		color := mgl32.Vec3{1.0, 0.2, 0.2} // Red
		if sphereID == 1 {
			color = mgl32.Vec3{0.2, 0.2, 1.0} // Blue
		}

		// Scatter particles tightly around the target sphere surface (R=150)
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
// ## Spirograph Kinematics
// ----------------------------------------------------

// updateSphereCenters calculates the desired (kinematic) center position for each sphere
// based on the 2D Spirograph equations lifted to 3D.
func updateSphereCenters(dt float64) {
	timeAccumulator += float32(dt)
	// The Spirograph parameter t is the time accumulator scaled by spiroSpeed
	t := timeAccumulator * spiroSpeed

	R := spiroR
	k := spiroK
	l := spiroL

	// Constant ratio term from the equations: (1-k)/k
	ratioTerm := (1.0 - k) / k

	// --- Calculate the base Spirograph coordinates (X_base, Y_base) ---

	// Term 1: R*(1-k)*cos(t) or sin(t)
	term1_x := R * (1.0 - k) * float32(math.Cos(float64(t)))
	term1_y := R * (1.0 - k) * float32(math.Sin(float64(t)))

	// Term 2: R*l*k*cos(ratioTerm*t) or sin(ratioTerm*t)
	term2_x := R * l * k * float32(math.Cos(float64(ratioTerm*t)))
	term2_y := R * l * k * float32(math.Sin(float64(ratioTerm*t))) // Note: positive sin here for the general form

	// Base Spirograph Path: x(t)=... + ... , y(t)=... - ...
	x_base := term1_x + term2_x
	y_base := term1_y - term2_y

	// Z-axis motion (Simple opposite oscillation for 3D separation)
	z_osc := R * 0.15 * float32(math.Sin(float64(t*0.5)))

	// --- Assign Center Targets ---

	// Sphere A: Follows the base path
	xA := x_base
	yA := y_base
	zA := z_osc

	// Sphere B: Follows the mirrored path with opposite Z-offset
	xB := -x_base
	yB := -y_base
	zB := -z_osc

	// Store the desired target center
	simData.Coms[0] = mgl32.Vec3{xA, yA, zA} // Target Center A
	simData.Coms[1] = mgl32.Vec3{xB, yB, zB} // Target Center B
}

// ----------------------------------------------------
// ## High-Performance Dynamics Loop
// ----------------------------------------------------

func updateParticlesPerformant(dt float64) {
	dt32 := float32(dt)
	dampingMultiplier := float32(1.0 - tdaDamping)

	// STEP 0: UPDATE DYNAMIC CENTER TARGETS (The Spirograph motion)
	updateSphereCenters(dt)

	// Step 1: TDA/Topological Analysis
	// Calculates the actual COM and connectivity status
	actualCOMs := [2]mgl32.Vec3{}
	TopologicalAnalysisModule(simData.Particles, simData.ParticleID, actualCOMs[:], simData.Connected[:])

	// Step 2: Rebuild Spatial Hash
	sh := NewSpatialHash(simData.Particles, ddgRadius)

	// -----------------------------------------------------------------
	// Step 3: DDG/TDA Physics Integration (Parallelized)
	// -----------------------------------------------------------------

	var wg sync.WaitGroup
	chunkSize := nParticles / numWorkers

	for w := 0; w < numWorkers; w++ {
		wg.Add(1)

		start := w * chunkSize
		end := start + chunkSize
		if w == numWorkers-1 {
			end = nParticles
		}

		go func(start, end int) {
			defer wg.Done()

			for i := start; i < end; i++ {
				p := &simData.Particles[i]
				pos := p.Pos
				sphereID := simData.ParticleID[i]

				// TargetCenter is the MOVING Spirograph point
				targetCenter := simData.Coms[sphereID]
				isConnected := simData.Connected[sphereID]

				// 1. DDG Force (local surface maintenance)
				ddgForce := DDGModuleSH(i, pos, sh, simData.Particles)

				// 2. Spring Force (Confinement: pulls toward the MOVING target center)
				centerForce := getSphereSpringForce(pos, targetCenter)

				// 3. TDA Restoring Force (Topology repair: pulls toward the actual COM if broken)
				tdaForce := mgl32.Vec3{0, 0, 0}
				if !isConnected {
					actualCOM := actualCOMs[sphereID]
					dir := actualCOM.Sub(pos).Normalize()
					tdaForce = dir.Mul(tdaRestoreForce)
				}

				// 4. Inter-Sphere Repulsion (keeps the two spheres separate)
				otherID := 1 - sphereID
				otherCenter := simData.Coms[otherID]
				repulsionForce := getInterSphereRepulsion(pos, otherCenter)

				// Combined Acceleration
				totalAcc := ddgForce.Add(centerForce).Add(tdaForce).Add(repulsionForce)

				// Velocity and Position Update (Verlet Integration proxy)
				p.Vel = p.Vel.Add(totalAcc.Mul(dt32))
				p.Vel = p.Vel.Mul(dampingMultiplier)
				p.Pos = p.Pos.Add(p.Vel.Mul(dt32))
			}
		}(start, end)
	}

	wg.Wait()
}

// ----------------------------------------------------
// ## Force Modules
// ----------------------------------------------------

func DDGModuleSH(index int, pos mgl32.Vec3, sh *SpatialHash, data []Particle) mgl32.Vec3 {
	neighbors := sh.FindNeighborsSH(index, pos, data, ddgRadius)
	if len(neighbors) == 0 {
		return mgl32.Vec3{0, 0, 0}
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
	return totalForce.Mul(1.0 / float32(len(neighbors)))
}

// Confinement force: pulls particle toward its moving sphere's target center.
func getSphereSpringForce(pos mgl32.Vec3, center mgl32.Vec3) mgl32.Vec3 {
	vec := pos.Sub(center)
	dist := float64(vec.Len())

	if dist < 0.001 {
		return mgl32.Vec3{0, 0, 0}
	}

	dir := vec.Normalize()
	force := springStrength * (dist - sphereRadius)
	return dir.Mul(-float32(force))
}

// Repulsion force between a particle and the OTHER sphere's center.
func getInterSphereRepulsion(pos mgl32.Vec3, otherCenter mgl32.Vec3) mgl32.Vec3 {
	vec := pos.Sub(otherCenter)
	dist := vec.Len()

	if dist > repulsionDistCheck {
		return mgl32.Vec3{0, 0, 0}
	}

	// Inverse distance squared falloff for repulsion
	forceMag := repulsionK / (dist * dist)
	dir := vec.Normalize()

	return dir.Mul(forceMag)
}

// ----------------------------------------------------
// ## TDA Module (Tracks two centers)
// ----------------------------------------------------

// TopologicalAnalysisModule calculates actual COM and connectivity for two groups.
func TopologicalAnalysisModule(data []Particle, ids []int, coms []mgl32.Vec3, connected []bool) {
	if len(data) == 0 {
		coms[0], coms[1] = mgl32.Vec3{0, 0, 0}, mgl32.Vec3{0, 0, 0}
		connected[0], connected[1] = true, true
		return
	}

	// 1. Calculate Center of Mass for each sphere
	sumCom := [2]mgl32.Vec3{}
	counts := [2]int{}

	for i, p := range data {
		id := ids[i]
		sumCom[id] = sumCom[id].Add(p.Pos)
		counts[id]++
	}

	for i := 0; i < 2; i++ {
		if counts[i] > 0 {
			coms[i] = sumCom[i].Mul(1.0 / float32(counts[i])) // Actual COM
		} else {
			coms[i] = mgl32.Vec3{0, 0, 0}
		}
	}

	// 2. Connected Components Check (Proxy for Betti-0)
	// We check for components *within* each sphere's particle group.
	visited := make([]bool, len(data))

	// Reset connectivity status
	connected[0], connected[1] = true, true

	for i := range data {
		if !visited[i] {
			sphereID := ids[i]

			// If a component is found starting from this particle,
			// check if it belongs to a previously visited particle of the SAME ID.

			// Simple BFS within the same sphere ID group
			visited[i] = true

			queue := []int{i}

			for len(queue) > 0 {
				curr := queue[0]
				queue = queue[1:]

				for j := range data {
					if !visited[j] && ids[j] == sphereID {
						if data[curr].Pos.Sub(data[j].Pos).LenSqr() < tdaClusterRadius*tdaClusterRadius {
							visited[j] = true
							queue = append(queue, j)
						}
					}
				}
			}

			// A more accurate check requires external knowledge or a dedicated component tracking structure
			// Since we start from the first unvisited particle, if a second one of the same ID is found,
			// the first one must be disconnected.

			// For this proxy, if the first particle i is found and the sphere is already
			// marked as visited, we mark it as disconnected.
			// (This is a simplified approach to avoid a full multi-source component analysis)

			if counts[sphereID] < len(data)/2 { // Heuristic: if counts is low, skip BFS
				continue
			}

			// For simplicity and performance, we'll rely on the damping and strong
			// TDA restoration to maintain connectivity, and skip the expensive BFS
			// once a component is established.

			if i > 0 && ids[i] == ids[i-1] && !connected[sphereID] {
				// If a new component is found for a sphere already marked broken,
				// we assume the breakup is persisting.
				// This is a gross simplification of TDA.
				connected[sphereID] = false
			} else if i > 0 && ids[i] != ids[i-1] {
				// Reset check logic when changing sphere
			}

		}
	}

	// Revert to a simpler global component check for TDA proxy:
	// A full BFS over all particles. If total components > 2, it's highly unstable.
	totalVisited := make([]bool, len(data))
	totalComponents := 0

	for i := range data {
		if !totalVisited[i] {
			totalComponents++
			// Resetting the connectivity status based on a simple component count
			if totalComponents > 2 {
				connected[ids[i]] = false
			}

			queue := []int{i}
			totalVisited[i] = true

			for len(queue) > 0 {
				curr := queue[0]
				queue = queue[1:]

				for j := range data {
					if !totalVisited[j] {
						if data[curr].Pos.Sub(data[j].Pos).LenSqr() < tdaClusterRadius*tdaClusterRadius {
							totalVisited[j] = true
							queue = append(queue, j)
						}
					}
				}
			}
		}
	}

	// If the system is highly disconnected (many small clusters), TDA restoration will kick in.
	if totalComponents > 3 { // Allow for some fragmentation (2 spheres + 1 fragment)
		connected[0] = false
		connected[1] = false
	} else if totalComponents == 3 {
		// If exactly 3 components, one sphere broke into two.
		// Set both to 'broken' to ensure strong restoring force on both.
		// A proper TDA would identify which sphere is broken.
		connected[0] = false
		connected[1] = false
	} else if totalComponents == 2 {
		connected[0] = true
		connected[1] = true
	} else if totalComponents == 1 {
		// The two spheres have merged!
		connected[0] = false
		connected[1] = false
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
	proj := mgl32.Perspective(mgl32.DegToRad(45), float32(winWidth)/winHeight, 0.1, 5000)
	view := cameraMatrix()
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
