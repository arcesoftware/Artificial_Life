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

type HashBucket []int

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

func (sh *SpatialHash) HashIndex(pos mgl32.Vec3) int {
	ix := int(pos.X() * sh.InvCellSize)
	iy := int(pos.Y() * sh.InvCellSize)
	iz := int(pos.Z() * sh.InvCellSize)
	return ix*73856093 + iy*1934983 + iz*83492791
}

func (sh *SpatialHash) Update(particles []Particle) {
	sh.Grid = make(map[int]HashBucket, len(particles))
	for i, p := range particles {
		key := sh.HashIndex(p.Pos)
		sh.Grid[key] = append(sh.Grid[key], i)
	}
}

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
}

const (
	winWidth   = 1280
	winHeight  = 720
	nParticles = 1618
	pointSize  = 1.0

	particleVboSize = 6 * 4
	numWorkers      = 8

	kNeighbors = 4
	ddgRadius  = 30.0
	ddgSpringK = 150.0

	tdaClusterRadius = 99.0
	tdaRestoreForce  = 555.0
	tdaDamping       = 0.618033989
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

// --------------- New shape parameters (Torus) ---------------
var (
	torusMajorR             = float32(255.0) // major radius (distance from center to tube center)
	torusMinorR             = float32(87.0)  // minor radius (tube radius)
	surfaceSpringStrength   = float32(35.0)  // spring strength toward torus surface
	surfaceAttractionJitter = float32(16.0)  // random jitter around surface on init
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

	window, err := glfw.CreateWindow(winWidth, winHeight, "DDG/TDA Particles - Torus Shape", nil, nil)
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

	// --- Particle Initialization (Torus) ---
	particles := make([]Particle, nParticles)
	for i := 0; i < nParticles; i++ {
		particles[i].Pos = sampleTorusPoint(torusMajorR, torusMinorR, surfaceAttractionJitter)
		particles[i].Vel = mgl32.Vec3{0, 0, 0}
		// Color mapping: based on position to create gradients
		p := particles[i].Pos
		h := (0.5 + 0.5*float32(math.Sin(float64(p.X()/torusMajorR*2.0)))) // basic hue-ish scalar
		r := float32(0.4 + 0.6*h)
		g := float32(0.1 + 0.6*(1.0-h))
		b := float32(0.1 + 0.2*(h*h))
		particles[i].Col = mgl32.Vec3{r, g, b}
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
// ## Torus sampling + surface attraction
// ----------------------------------------------------

// sampleTorusPoint returns a point near the torus surface with optional jitter
func sampleTorusPoint(R, r, jitter float32) mgl32.Vec3 {
	u := float32(rand.Float64()) * 2.0 * float32(math.Pi) // around major circle
	v := float32(rand.Float64()) * 2.0 * float32(math.Pi) // around tube

	// Perfect torus point
	x := (R + r*float32(math.Cos(float64(v)))) * float32(math.Cos(float64(u)))
	y := (R + r*float32(math.Cos(float64(v)))) * float32(math.Sin(float64(u)))
	z := r * float32(math.Sin(float64(v)))

	// Add small radial jitter so particles aren't exactly on the surface
	jx := (float32(rand.Float32())*2.0 - 1.0) * jitter
	jy := (float32(rand.Float32())*2.0 - 1.0) * jitter
	jz := (float32(rand.Float32())*2.0 - 1.0) * jitter

	return mgl32.Vec3{x + jx, y + jy, z + jz}
}

// nearestPointOnTorus computes an approximate nearest surface point on torus to pos.
// Uses analytic projection by computing angular coordinates from the point.
func nearestPointOnTorus(pos mgl32.Vec3, R, r float32) mgl32.Vec3 {
	x := pos.X()
	y := pos.Y()
	z := pos.Z()

	// angle around major circle
	u := float32(math.Atan2(float64(y), float64(x)))
	// distance from point to major circle in plane
	dxy := float32(math.Sqrt(float64(x*x + y*y)))
	// angle around minor circle
	// if dxy==0 handle gracefully
	var v float32
	if dxy != 0 {
		v = float32(math.Atan2(float64(z), float64(dxy-R)))
	} else {
		// if directly above origin, pick v from z sign
		if z >= 0 {
			v = float32(math.Pi / 2.0)
		} else {
			v = float32(-math.Pi / 2.0)
		}
	}

	nx := (R + r*float32(math.Cos(float64(v)))) * float32(math.Cos(float64(u)))
	ny := (R + r*float32(math.Cos(float64(v)))) * float32(math.Sin(float64(u)))
	nz := r * float32(math.Sin(float64(v)))
	return mgl32.Vec3{nx, ny, nz}
}

// getSurfaceSpringForce attracts a particle toward the torus surface (nearest point) with spring-like behavior.
func getSurfaceSpringForce(pos mgl32.Vec3) mgl32.Vec3 {
	nearest := nearestPointOnTorus(pos, torusMajorR, torusMinorR)
	dir := nearest.Sub(pos)
	// Hooke's law toward surface
	force := dir.Mul(surfaceSpringStrength)
	return force
}

// ----------------------------------------------------
// ## High-Performance Dynamics Loop
// ----------------------------------------------------

func updateParticlesPerformant(dt float64) {
	dt32 := float32(dt)
	dampingMultiplier := float32(1.0 - tdaDamping)

	centerOfMass, isConnected := TopologicalAnalysisModule(simData.Particles)

	sh := NewSpatialHash(simData.Particles, ddgRadius)

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

				ddgForce := DDGModuleSH(i, pos, sh, simData.Particles)
				// Replace center spring with torus surface spring
				centerForce := getSurfaceSpringForce(pos)

				tdaForce := mgl32.Vec3{0, 0, 0}
				if !isConnected {
					dir := centerOfMass.Sub(pos).Normalize()
					tdaForce = dir.Mul(tdaRestoreForce)
				}

				totalAcc := ddgForce.Add(centerForce).Add(tdaForce)

				p.Vel = p.Vel.Add(totalAcc.Mul(dt32))
				p.Vel = p.Vel.Mul(dampingMultiplier)
				p.Pos = p.Pos.Add(p.Vel.Mul(dt32))
			}
		}(start, end)
	}

	wg.Wait()
}

// ----------------------------------------------------
// ## DDG Module (Discrete Curvature Flow Proxy)
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

// ----------------------------------------------------
// ## TDA Module (Persistent Homology Proxy for Betti-0)
// ----------------------------------------------------

func TopologicalAnalysisModule(data []Particle) (centerOfMass mgl32.Vec3, isConnected bool) {
	if len(data) == 0 {
		return mgl32.Vec3{0, 0, 0}, true
	}

	com := mgl32.Vec3{0, 0, 0}
	for _, p := range data {
		com = com.Add(p.Pos)
	}
	centerOfMass = com.Mul(1.0 / float32(len(data)))

	visited := make([]bool, len(data))
	numComponents := 0

	for i := range data {
		if !visited[i] {
			numComponents++
			queue := []int{i}
			visited[i] = true

			for len(queue) > 0 {
				curr := queue[0]
				queue = queue[1:]

				for j := range data {
					if !visited[j] && data[curr].Pos.Sub(data[j].Pos).LenSqr() < tdaClusterRadius*tdaClusterRadius {
						visited[j] = true
						queue = append(queue, j)
					}
				}
			}
		}
	}

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
	if distance < 300 {
		distance = 300
	}
	if distance > 4000 {
		distance = 4000
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
