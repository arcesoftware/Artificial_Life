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
	// large primes to mix coordinates
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
// ## Simulation Data and Constants (Spherical-form state)
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

	tdaClusterRadius = 150.0
	tdaRestoreForce  = 500.0
	tdaDamping       = 0.618033989
)

type Particle struct {
	// Spherical coordinates
	R     float32 // radial distance
	Theta float32 // azimuth around Y axis (XZ plane), θ
	Phi   float32 // elevation (from XZ plane), φ

	// Velocities in spherical generalized coordinates
	Vr     float32
	Vtheta float32
	Vphi   float32

	// Cached Cartesian position (for hashing & rendering)
	Pos mgl32.Vec3

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

// Torus shape parameters (we'll attract particles to torus surface)
var (
	torusMajorR             = float32(200.0)
	torusMinorR             = float32(60.0)
	surfaceSpringStrength   = float32(30.0)
	surfaceAttractionJitter = float32(8.0)
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

	window, err := glfw.CreateWindow(winWidth, winHeight, "Spherical DDG/TDA - Torus (Spherical state)", nil, nil)
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

	// Initialize particles: sample torus in Cartesian, convert to spherical state
	particles := make([]Particle, nParticles)
	for i := 0; i < nParticles; i++ {
		cart := sampleTorusPoint(torusMajorR, torusMinorR, surfaceAttractionJitter)
		r, theta, phi := cartToSpherical(cart)
		particles[i].R = r
		particles[i].Theta = theta
		particles[i].Phi = phi
		particles[i].Vr = 0
		particles[i].Vtheta = 0
		particles[i].Vphi = 0
		particles[i].Pos = cart
		// color
		h := (0.5 + 0.5*float32(math.Sin(float64(cart.X()/torusMajorR*2.0))))
		rcol := float32(0.4 + 0.6*h)
		gcol := float32(0.1 + 0.6*(1.0-h))
		bcol := float32(0.1 + 0.2*(h*h))
		particles[i].Col = mgl32.Vec3{rcol, gcol, bcol}
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
// ## Spherical <-> Cartesian helpers & basis vectors
// ----------------------------------------------------

// sphToCart: uses convention x = r*cosφ*cosθ, y = r*sinφ, z = r*cosφ*sinθ
func sphToCart(r, theta, phi float32) mgl32.Vec3 {
	cp := float32(math.Cos(float64(phi)))
	sp := float32(math.Sin(float64(phi)))
	ct := float32(math.Cos(float64(theta)))
	st := float32(math.Sin(float64(theta)))
	x := r * cp * ct
	y := r * sp
	z := r * cp * st
	return mgl32.Vec3{x, y, z}
}

// cartToSpherical inverse (handles safe domain)
func cartToSpherical(pos mgl32.Vec3) (r, theta, phi float32) {
	x := pos.X()
	y := pos.Y()
	z := pos.Z()
	rf := float32(math.Sqrt(float64(x*x + y*y + z*z)))
	theta = float32(math.Atan2(float64(z), float64(x))) // azimuth in XZ plane
	rho := float32(math.Sqrt(float64(x*x + z*z)))       // projection onto XZ plane
	// phi is elevation angle from XZ plane
	phi = float32(math.Atan2(float64(y), float64(rho)))
	return rf, theta, phi
}

// spherical basis unit vectors in Cartesian coords for given theta, phi
// e_r, e_theta (azimuthal), e_phi (elevation)
func sphericalBasis(theta, phi float32) (e_r, e_theta, e_phi mgl32.Vec3) {
	ct := float32(math.Cos(float64(theta)))
	st := float32(math.Sin(float64(theta)))
	cp := float32(math.Cos(float64(phi)))
	sp := float32(math.Sin(float64(phi)))

	// e_r
	e_r = mgl32.Vec3{cp * ct, sp, cp * st}
	// e_theta (derivative w.r.t theta, normalized): (-sinθ, 0, cosθ)
	e_theta = mgl32.Vec3{-st, 0, ct}
	// e_phi (derivative w.r.t phi, normalized): (-sinφ*cosθ, cosφ, -sinφ*sinθ)
	e_phi = mgl32.Vec3{-sp * ct, cp, -sp * st}

	// Note: e_theta and e_phi are already unit vectors under chosen parametrization.
	return
}

// project Cartesian force into spherical generalized coordinate accelerations
// returns (a_r, a_theta, a_phi) where
// a_r = F·e_r
// a_theta = (F·e_theta) / (r * cosφ)  (guard cosφ small)
// a_phi = (F·e_phi) / r
func cartesianForceToSphericalAcc(F mgl32.Vec3, r, theta, phi float32) (a_r, a_theta, a_phi float32) {
	e_r, e_theta, e_phi := sphericalBasis(theta, phi)
	Fr := dot(F, e_r)
	Ftheta := dot(F, e_theta)
	Fphi := dot(F, e_phi)

	// guard
	const eps = 1e-6
	cp := float32(math.Cos(float64(phi)))
	if cp < eps && cp > -eps {
		// avoid division by near-zero; scale cosφ slightly
		if cp >= 0 {
			cp = eps
		} else {
			cp = -eps
		}
	}
	if r < eps {
		r = eps
	}

	a_r = Fr
	a_theta = Ftheta / (r * cp)
	a_phi = Fphi / r
	return
}

// dot product helper
func dot(a, b mgl32.Vec3) float32 {
	return a.X()*b.X() + a.Y()*b.Y() + a.Z()*b.Z()
}

// ----------------------------------------------------
// ## Torus sampling + nearest point (Cartesian helpers)
// ----------------------------------------------------

// sampleTorusPoint returns a Cartesian point near torus surface (same as earlier)
func sampleTorusPoint(R, r, jitter float32) mgl32.Vec3 {
	u := float32(rand.Float64()) * 2.0 * float32(math.Pi) // around major circle
	v := float32(rand.Float64()) * 2.0 * float32(math.Pi) // around tube

	x := (R + r*float32(math.Cos(float64(v)))) * float32(math.Cos(float64(u)))
	y := (R + r*float32(math.Cos(float64(v)))) * float32(math.Sin(float64(u)))
	z := r * float32(math.Sin(float64(v)))

	jx := (float32(rand.Float32())*2.0 - 1.0) * jitter
	jy := (float32(rand.Float32())*2.0 - 1.0) * jitter
	jz := (float32(rand.Float32())*2.0 - 1.0) * jitter

	return mgl32.Vec3{x + jx, y + jy, z + jz}
}

// nearestPointOnTorus (Cartesian) — same approach as before
func nearestPointOnTorus(pos mgl32.Vec3, R, r float32) mgl32.Vec3 {
	x := pos.X()
	y := pos.Y()
	z := pos.Z()

	u := float32(math.Atan2(float64(y), float64(x)))
	dxy := float32(math.Sqrt(float64(x*x + y*y)))
	var v float32
	if dxy != 0 {
		v = float32(math.Atan2(float64(z), float64(dxy-R)))
	} else {
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

// ----------------------------------------------------
// ## High-Performance Dynamics Loop (Spherical state update)
// ----------------------------------------------------

func updateParticlesPerformant(dt float64) {
	dt32 := float32(dt)
	dampingMultiplier := float32(1.0 - tdaDamping)

	// Build Cartesian positions (cached) for neighbor finding and COM
	for i := range simData.Particles {
		p := &simData.Particles[i]
		p.Pos = sphToCart(p.R, p.Theta, p.Phi)
	}

	// Compute center of mass in Cartesian (for TDA)
	centerOfMass, isConnected := TopologicalAnalysisModule(simData.Particles)

	// Build spatial hash using cached Cartesian positions
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

				// Recompute current Cartesian pos (safe)
				posCart := sphToCart(p.R, p.Theta, p.Phi)

				// 1) DDG force in Cartesian (sum of neighbor springs)
				Fddg := mgl32.Vec3{0, 0, 0}
				neighbors := sh.FindNeighborsSH(i, posCart, simData.Particles, ddgRadius)
				if len(neighbors) > 0 {
					targetDist := ddgRadius / float32(math.Sqrt(float64(kNeighbors))) * 0.7
					for _, ni := range neighbors {
						np := simData.Particles[ni]
						vec := np.Pos.Sub(posCart)
						dist := vec.Len()
						forceMag := ddgSpringK * (dist - targetDist)
						if dist > 0.001 {
							Fddg = Fddg.Add(vec.Normalize().Mul(forceMag * 0.5))
						}
					}
					Fddg = Fddg.Mul(1.0 / float32(len(neighbors)))
				}

				// 2) Surface spring to torus (Cartesian)
				nearest := nearestPointOnTorus(posCart, torusMajorR, torusMinorR)
				Fsurf := nearest.Sub(posCart).Mul(surfaceSpringStrength)

				// 3) TDA restoring force toward COM if disconnected (Cartesian)
				Ftda := mgl32.Vec3{0, 0, 0}
				if !isConnected {
					dir := centerOfMass.Sub(posCart)
					if dir.Len() > 0.001 {
						Ftda = dir.Normalize().Mul(tdaRestoreForce)
					}
				}

				// Total force in Cartesian
				Ftotal := Fddg.Add(Fsurf).Add(Ftda)

				// Project into spherical generalized accelerations
				a_r, a_theta, a_phi := cartesianForceToSphericalAcc(Ftotal, p.R, p.Theta, p.Phi)

				// Integrate in spherical coordinates (explicit Euler)
				p.Vr += a_r * dt32
				p.Vtheta += a_theta * dt32
				p.Vphi += a_phi * dt32

				// damping applied to velocities
				p.Vr = p.Vr * dampingMultiplier
				p.Vtheta = p.Vtheta * dampingMultiplier
				p.Vphi = p.Vphi * dampingMultiplier

				// Update generalized coordinates
				p.R += p.Vr * dt32
				p.Theta += p.Vtheta * dt32
				p.Phi += p.Vphi * dt32

				// keep theta in sane range
				if p.Theta > math.Pi*2 {
					p.Theta = float32(math.Mod(float64(p.Theta), 2.0*math.Pi))
				}
				if p.Theta < 0 {
					p.Theta = float32(math.Mod(float64(p.Theta), 2.0*math.Pi))
				}
				// clamp phi to slightly below ±pi/2 to avoid singularity
				maxPhi := float32(1.4999) // ~85.9 degrees
				if p.Phi > maxPhi {
					p.Phi = maxPhi
					p.Vphi = 0
				}
				if p.Phi < -maxPhi {
					p.Phi = -maxPhi
					p.Vphi = 0
				}
				// ensure r stays positive
				if p.R < 0.1 {
					p.R = 0.1
					p.Vr = 0
				}

				// Update cached Cartesian pos for rendering/hashing next frame
				p.Pos = sphToCart(p.R, p.Theta, p.Phi)
			}
		}(start, end)
	}
	wg.Wait()
}

// ----------------------------------------------------
// ## DDG Module removed (we compute DDG in loop) — kept for compatibility
// ----------------------------------------------------

func DDGModuleSH(index int, pos mgl32.Vec3, sh *SpatialHash, data []Particle) mgl32.Vec3 {
	// Not used for spherical run; kept as fallback (computes Cartesian ddg)
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
// ## TDA Module (uses cached Cartesian positions)
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
// ## Rendering, Camera, and Shaders (unchanged, uses cached Cartesian pos)
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
		// ensure Pos cached
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

// Camera, input
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

// Shader helpers (unchanged)
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

// ----------------------------------------------------
// ## End of File
// ----------------------------------------------------
