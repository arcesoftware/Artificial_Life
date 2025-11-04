package main

import (
	"log"
	"math"
	"math/rand"
	"runtime"
	"time"

	"github.com/go-gl/gl/v4.1-core/gl"
	"github.com/go-gl/glfw/v3.3/glfw"
	"github.com/go-gl/mathgl/mgl32"
)

// --- Constants and Global Settings ---
const (
	width  = 1280
	height = 800
	title  = "3D Orbiting Glowing Particle Octahedron (Spring Physics)"

	BOUNDARY = 1.8 // Large enough to contain the octahedron (side length sqrt(2))

	NODE_SIZE         = 0.04 // Slightly larger nodes for fewer particles
	MAX_CONN_DISTANCE = 1.5  // Max distance to draw lines (covers octahedron edge)

	// Octahedron Specifics
	PARTICLE_COUNT              = 6          // Octahedron has 6 vertices
	OCTAHEDRON_REST_LEN float32 = 1.41421356 // Edge length of a unit octahedron (sqrt(2))

	// Interactivity / Simulation
	MOUSE_FORCE_RADIUS       = 1.0 // Increased interaction radius
	MOUSE_REPULSION_STRENGTH = 2.5 // Increased repulsion strength
)

// --- Global mouse / camera state ---
var (
	mouseNDC          mgl32.Vec2 // cursor in NDC (-1..1)
	leftMouseDown     = false
	lastMouseX        float64
	lastMouseY        float64
	yaw               float32 = 0.8  // horizontal angle
	pitch             float32 = 0.15 // vertical angle
	cameraDistance    float32 = 6.0
	scrollSensitivity         = 0.4
	orbitSensitivity          = 0.005
)

// Octahedron edges defined by the indices of the 6 vertices (0-5)
// This array defines the 12 spring connections for the octahedron's structure.
var octahedronEdges = [][2]int{
	{0, 2}, {0, 3}, {0, 4}, {0, 5}, // Connects +X to N, S, +Z, -Z
	{1, 2}, {1, 3}, {1, 4}, {1, 5}, // Connects -X to N, S, +Z, -Z
	{2, 4}, {2, 5}, // Connects +Y to +Z, -Z
	{3, 4}, {3, 5}, // Connects -Y to +Z, -Z
}

// --- Shader Sources ---
var vertexShaderSource = `#version 410
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 color;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

out vec3 vertColor;
out vec3 vertPosition;

void main() {
	vec4 worldPos = model * vec4(position, 1.0);
	gl_Position = projection * view * worldPos;
	vertColor = color;
	vertPosition = worldPos.xyz;
}
` + "\x00"

var lineFragmentShaderSource = `#version 410
in vec3 vertColor;
out vec4 fragColor;
void main() {
	fragColor = vec4(vertColor, 1.0);
}
` + "\x00"

var particleFragmentShaderSource = `#version 410
in vec3 vertColor;
in vec3 vertPosition;
out vec4 fragColor;
void main() {
	// simple boosted color for center brightness; additive blending will create glow
	fragColor = vec4(vertColor * 2.3, 0.85);
}
` + "\x00"

// --- Data Structures ---
type Particle struct {
	Pos mgl32.Vec3
	Col mgl32.Vec3
	Vel mgl32.Vec3
}

type SimulationData struct {
	Particles []Particle
	// NextIdx is ignored for the octahedron; all connections are handled via octahedronEdges
	NextIdx []int
}

// --- GLFW callbacks ---
func keyCallback(window *glfw.Window, key glfw.Key, scancode int, action glfw.Action, mods glfw.ModifierKey) {
	if key == glfw.KeyEscape && action == glfw.Press {
		window.SetShouldClose(true)
	}
}

func cursorPosCallback(window *glfw.Window, xpos float64, ypos float64) {
	// NDC coords
	w, h := window.GetSize()
	nx := float32(xpos/float64(w)*2.0 - 1.0)
	ny := float32(1.0 - ypos/float64(h)*2.0)
	mouseNDC = mgl32.Vec2{nx, ny}

	// If left button down, use motion to orbit camera
	if leftMouseDown {
		dx := xpos - lastMouseX
		dy := ypos - lastMouseY
		lastMouseX = xpos
		lastMouseY = ypos

		// update yaw/pitch based on movement (invert dy to match intuitive up/down)
		yaw -= float32(dx) * float32(orbitSensitivity)
		pitch += float32(dy) * float32(orbitSensitivity)

		// clamp pitch
		maxPitch := float32(math.Pi/2 - 0.05)
		if pitch > maxPitch {
			pitch = maxPitch
		}
		if pitch < -maxPitch {
			pitch = -maxPitch
		}
	}
}

func mouseButtonCallback(window *glfw.Window, button glfw.MouseButton, action glfw.Action, mods glfw.ModifierKey) {
	if button == glfw.MouseButtonLeft {
		switch action {
		case glfw.Press:
			leftMouseDown = true
			lastMouseX, lastMouseY = window.GetCursorPos()
		case glfw.Release:
			leftMouseDown = false
		}
	}
}

func scrollCallback(window *glfw.Window, xoff float64, yoff float64) {
	cameraDistance -= float32(yoff) * float32(scrollSensitivity)
	if cameraDistance < 1.5 {
		cameraDistance = 1.5
	}
	if cameraDistance > 40.0 {
		cameraDistance = 40.0
	}
}

// --- OpenGL helpers ---
func compileShader(source string, shaderType uint32) uint32 {
	shader := gl.CreateShader(shaderType)
	csources, free := gl.Strs(source)
	gl.ShaderSource(shader, 1, csources, nil)
	free()
	gl.CompileShader(shader)

	var status int32
	gl.GetShaderiv(shader, gl.COMPILE_STATUS, &status)
	if status == gl.FALSE {
		var logLength int32
		gl.GetShaderiv(shader, gl.INFO_LOG_LENGTH, &logLength)
		logStr := make([]byte, logLength+1)
		gl.GetShaderInfoLog(shader, logLength, nil, &logStr[0])
		log.Fatalf("failed to compile shader %v: %v", shaderType, gl.GoStr(&logStr[0]))
	}
	return shader
}

func createProgram(vertexShader, fragmentShader uint32) uint32 {
	program := gl.CreateProgram()
	gl.AttachShader(program, vertexShader)
	gl.AttachShader(program, fragmentShader)
	gl.LinkProgram(program)

	var status int32
	gl.GetProgramiv(program, gl.LINK_STATUS, &status)
	if status == gl.FALSE {
		var logLength int32
		gl.GetProgramiv(program, gl.INFO_LOG_LENGTH, &logLength)
		logStr := make([]byte, logLength+1)
		gl.GetProgramInfoLog(program, logLength, nil, &logStr[0])
		log.Fatalf("failed to link program: %v", gl.GoStr(&logStr[0]))
	}

	gl.DeleteShader(vertexShader)
	gl.DeleteShader(fragmentShader)

	return program
}

func initVao() (uint32, uint32) {
	var vao uint32
	gl.GenVertexArrays(1, &vao)
	gl.BindVertexArray(vao)

	var vbo uint32
	gl.GenBuffers(1, &vbo)
	gl.BindBuffer(gl.ARRAY_BUFFER, vbo)

	stride := int32(6 * 4) // 6 floats * 4 bytes

	gl.EnableVertexAttribArray(0)
	gl.VertexAttribPointerWithOffset(0, 3, gl.FLOAT, false, stride, 0)

	gl.EnableVertexAttribArray(1)
	gl.VertexAttribPointerWithOffset(1, 3, gl.FLOAT, false, stride, uintptr(3*4))

	gl.BindVertexArray(0)
	gl.BindBuffer(gl.ARRAY_BUFFER, 0)

	return vao, vbo
}

// --- Simulation Update (with springs) ---
func updateSimulation(simData *SimulationData, deltaTime float64, projection, view mgl32.Mat4, cameraPos mgl32.Vec3) {
	if deltaTime <= 0 {
		return
	}
	dt := float32(deltaTime)
	const globalDamping = 0.985

	// 1. Mouse position in world (intersect camera ray with z=0 plane)
	mouseWorld := unprojectMouseToWorldPlane(mouseNDC, projection, view, cameraPos, 0.0)

	// Spring settings
	const springK = 8.8 // stiffness
	const springDamping = 0.95
	// Use the specific rest length for the octahedron structure
	restLen := OCTAHEDRON_REST_LEN

	for i := range simData.Particles {
		p := &simData.Particles[i]

		// 1) Apply Octahedron Spring forces (12 edges, 2 forces per particle)
		// We iterate over the fixed set of edges
		if len(simData.Particles) == PARTICLE_COUNT { // Only run for the 6-particle octahedron
			for _, edge := range octahedronEdges {
				idxA, idxB := edge[0], edge[1]

				// Only calculate force once per edge (e.g., when i is idxA)
				if i == idxA {
					other := &simData.Particles[idxB]
					delta := other.Pos.Sub(p.Pos)
					dist := delta.Len()

					if dist > 0.0001 {
						dir := delta.Normalize()
						// Hooke's law: F = -k(x - x0)
						x := dist - restLen
						force := dir.Mul(springK * x)
						// apply equal & opposite
						p.Vel = p.Vel.Add(force.Mul(dt))
						other.Vel = other.Vel.Sub(force.Mul(dt))
					}
				}
			}
		}

		// 2) Mouse repulsion
		mv := mouseWorld.Sub(p.Pos)
		distMouse := mv.Len()
		if distMouse < MOUSE_FORCE_RADIUS && distMouse > 0.0001 {
			forceMag := float32(MOUSE_REPULSION_STRENGTH) * (1.0 - distMouse/float32(MOUSE_FORCE_RADIUS))
			rep := mv.Normalize().Mul(-forceMag)
			// push in XY, small effect on Z
			p.Vel = p.Vel.Add(mgl32.Vec3{rep.X(), rep.Y(), rep.Z() * 0.2}.Mul(dt))
		}

		// 3) Gentle random jitter
		jx := (rand.Float32()*2.0 - 1.0) * 0.02
		jy := (rand.Float32()*2.0 - 1.0) * 0.02
		jz := (rand.Float32()*2.0 - 1.0) * 0.02
		p.Vel = p.Vel.Add(mgl32.Vec3{jx, jy, jz}.Mul(0.3 * dt))

		// 4) Integrate velocity -> position
		p.Pos = p.Pos.Add(p.Vel.Mul(dt))

		// 5) Boundaries bounce (soft)
		if p.Pos.X() > BOUNDARY {
			p.Pos[0] = BOUNDARY
			p.Vel[0] = -p.Vel[0] * 0.6
		}
		if p.Pos.X() < -BOUNDARY {
			p.Pos[0] = -BOUNDARY
			p.Vel[0] = -p.Vel[0] * 0.6
		}
		if p.Pos.Y() > BOUNDARY {
			p.Pos[1] = BOUNDARY
			p.Vel[1] = -p.Vel[1] * 0.6
		}
		if p.Pos.Y() < -BOUNDARY {
			p.Pos[1] = -BOUNDARY
			p.Vel[1] = -p.Vel[1] * 0.6
		}
		if p.Pos.Z() > BOUNDARY {
			p.Pos[2] = BOUNDARY
			p.Vel[2] = -p.Vel[2] * 0.6
		}
		if p.Pos.Z() < -BOUNDARY {
			p.Pos[2] = -BOUNDARY
			p.Vel[2] = -p.Vel[2] * 0.6
		}

		// 6) Damping
		p.Vel = p.Vel.Mul(globalDamping).Mul(springDamping)
	}
}

// unproject mouse NDC to world-space point on plane z = planeZ
func unprojectMouseToWorldPlane(mouseNDC mgl32.Vec2, projection, view mgl32.Mat4, _ mgl32.Vec3, planeZ float32) mgl32.Vec3 {
	// Build inverse of projection*view
	pv := projection.Mul4(view)
	inv := pv.Inv()

	// NDC point at near plane (z = -1) and far plane (z = 1) in clip space
	near := mgl32.Vec4{mouseNDC.X(), mouseNDC.Y(), -1.0, 1.0}
	far := mgl32.Vec4{mouseNDC.X(), mouseNDC.Y(), 1.0, 1.0}

	worldNear4 := inv.Mul4x1(near)
	worldFar4 := inv.Mul4x1(far)
	if worldNear4.W() != 0 {
		worldNear4 = worldNear4.Mul(1.0 / worldNear4.W())
	}
	if worldFar4.W() != 0 {
		worldFar4 = worldFar4.Mul(1.0 / worldFar4.W())
	}
	worldNear := mgl32.Vec3{worldNear4.X(), worldNear4.Y(), worldNear4.Z()}
	worldFar := mgl32.Vec3{worldFar4.X(), worldFar4.Y(), worldFar4.Z()}

	dir := worldFar.Sub(worldNear)
	if math.Abs(float64(dir.Z())) < 1e-6 {
		// nearly parallel: return projection onto plane using camera XY
		return mgl32.Vec3{worldNear.X(), worldNear.Y(), planeZ}
	}
	t := (planeZ - worldNear.Z()) / dir.Z()
	point := worldNear.Add(dir.Mul(t))
	// Slight safeguard: if t outside [0,1], fallback to camera XY plane projection
	return point
}

// --- Drawing Functions ---
func drawLines(program uint32, vbo uint32, vao uint32, simData *SimulationData) {
	n := len(simData.Particles)
	lineVertices := make([]float32, 0, n*2*6)
	var vertexCount int32 = 0

	base := mgl32.Vec3{0.33, 0.33, 0.4}

	// Octahedron Drawing Logic (12 fixed edges)
	if n == PARTICLE_COUNT {
		for _, edge := range octahedronEdges {
			p1 := simData.Particles[edge[0]]
			p2 := simData.Particles[edge[1]]

			dist := p1.Pos.Sub(p2.Pos).Len()
			fade := float32(1.0 - math.Min(float64(dist), float64(MAX_CONN_DISTANCE))/MAX_CONN_DISTANCE)
			if fade <= 0.02 {
				continue
			}

			// Color lines based on the particles' colors
			col1 := p1.Col.Add(base).Mul(0.5).Mul(fade)
			col2 := p2.Col.Add(base).Mul(0.5).Mul(fade)

			lineVertices = append(lineVertices,
				p1.Pos.X(), p1.Pos.Y(), p1.Pos.Z(),
				col1.X(), col1.Y(), col1.Z(),
			)
			lineVertices = append(lineVertices,
				p2.Pos.X(), p2.Pos.Y(), p2.Pos.Z(),
				col2.X(), col2.Y(), col2.Z(),
			)
			vertexCount += 2
		}
	}

	if vertexCount == 0 {
		return
	}

	gl.BindBuffer(gl.ARRAY_BUFFER, vbo)
	dataSize := int(vertexCount) * 6 * 4
	gl.BufferData(gl.ARRAY_BUFFER, dataSize, gl.Ptr(lineVertices), gl.DYNAMIC_DRAW)

	gl.UseProgram(program)
	gl.BindVertexArray(vao)

	gl.Enable(gl.BLEND)
	gl.BlendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA)
	gl.LineWidth(3.0) // Thicker lines for the shape
	gl.DrawArrays(gl.LINES, 0, vertexCount)
	gl.Disable(gl.BLEND)

	gl.BindVertexArray(0)
	gl.BindBuffer(gl.ARRAY_BUFFER, 0)
}

func drawParticles(program uint32, vbo uint32, vao uint32, simData *SimulationData) {
	n := len(simData.Particles)
	// 6 vertices per particle, 6 floats per vertex
	particleVertices := make([]float32, 0, n*6*6)

	for _, p := range simData.Particles {
		x := p.Pos.X()
		y := p.Pos.Y()
		z := p.Pos.Z()

		// brightness based on speed
		speed := p.Vel.Len()
		brightness := 1.0 + speed*6.0
		col := p.Col.Mul(brightness)

		// Triangle 1 (Billboard Quad)
		particleVertices = append(particleVertices, x-NODE_SIZE, y-NODE_SIZE, z)
		particleVertices = append(particleVertices, col.X(), col.Y(), col.Z())
		particleVertices = append(particleVertices, x-NODE_SIZE, y+NODE_SIZE, z)
		particleVertices = append(particleVertices, col.X(), col.Y(), col.Z())
		particleVertices = append(particleVertices, x+NODE_SIZE, y+NODE_SIZE, z)
		particleVertices = append(particleVertices, col.X(), col.Y(), col.Z())

		// Triangle 2
		particleVertices = append(particleVertices, x-NODE_SIZE, y-NODE_SIZE, z)
		particleVertices = append(particleVertices, col.X(), col.Y(), col.Z())
		particleVertices = append(particleVertices, x+NODE_SIZE, y+NODE_SIZE, z)
		particleVertices = append(particleVertices, col.X(), col.Y(), col.Z())
		particleVertices = append(particleVertices, x+NODE_SIZE, y-NODE_SIZE, z)
		particleVertices = append(particleVertices, col.X(), col.Y(), col.Z())
	}

	vertexCount := int32(len(particleVertices) / 6)
	if vertexCount == 0 {
		return
	}

	gl.BindBuffer(gl.ARRAY_BUFFER, vbo)
	dataSize := len(particleVertices) * 4
	gl.BufferData(gl.ARRAY_BUFFER, dataSize, gl.Ptr(particleVertices), gl.DYNAMIC_DRAW)

	gl.UseProgram(program)
	gl.BindVertexArray(vao)

	gl.Enable(gl.BLEND)
	gl.BlendFunc(gl.SRC_ALPHA, gl.ONE) // additive glow
	gl.DrawArrays(gl.TRIANGLES, 0, vertexCount)
	gl.Disable(gl.BLEND)

	gl.BindVertexArray(0)
	gl.BindBuffer(gl.ARRAY_BUFFER, 0)
}

// --- Octahedron Data creation ---
func createMockData() *SimulationData {
	particles := make([]Particle, PARTICLE_COUNT)
	// Vertices of a unit octahedron along the axes
	vertices := []mgl32.Vec3{
		{1.0, 0.0, 0.0},  // 0: +X
		{-1.0, 0.0, 0.0}, // 1: -X
		{0.0, 1.0, 0.0},  // 2: +Y
		{0.0, -1.0, 0.0}, // 3: -Y
		{0.0, 0.0, 1.0},  // 4: +Z
		{0.0, 0.0, -1.0}, // 5: -Z
	}

	// Define distinct colors for each face/pole
	colors := []mgl32.Vec3{
		{1.0, 0.3, 0.3}, // Red/Pink
		{0.3, 1.0, 0.3}, // Green
		{0.3, 0.3, 1.0}, // Blue
		{1.0, 1.0, 0.3}, // Yellow
		{1.0, 0.3, 1.0}, // Magenta
		{0.3, 1.0, 1.0}, // Cyan
	}

	for i := 0; i < PARTICLE_COUNT; i++ {
		// Initial gentle spin velocity
		x, y, z := vertices[i].X(), vertices[i].Y(), vertices[i].Z()

		// Velocity for rotation (proportional to distance from center)
		vel := mgl32.Vec3{
			(-y * 0.08) + (z * 0.08),
			(x * 0.08) + (-z * 0.08),
			(-x * 0.08) + (y * 0.08),
		}.Mul(0.1)

		particles[i] = Particle{
			Pos: vertices[i],
			Col: colors[i],
			Vel: vel,
		}
	}

	// NextIdx is required by the struct definition but is unused/dummy for the Octahedron
	nextIdx := make([]int, PARTICLE_COUNT)
	for i := range nextIdx {
		nextIdx[i] = -1
	}

	return &SimulationData{
		Particles: particles,
		NextIdx:   nextIdx,
	}
}

// --- Main ---
func main() {
	runtime.LockOSThread()
	rand.Seed(time.Now().UnixNano())

	if err := glfw.Init(); err != nil {
		log.Fatalf("failed to initialize glfw: %v", err)
	}
	defer glfw.Terminate()

	glfw.WindowHint(glfw.ContextVersionMajor, 4)
	glfw.WindowHint(glfw.ContextVersionMinor, 1)
	glfw.WindowHint(glfw.OpenGLProfile, glfw.OpenGLCoreProfile)
	glfw.WindowHint(glfw.OpenGLForwardCompatible, glfw.True)

	window, err := glfw.CreateWindow(width, height, title, nil, nil)
	if err != nil {
		log.Fatalf("failed to create window: %v", err)
	}
	window.MakeContextCurrent()
	window.SetKeyCallback(keyCallback)
	window.SetCursorPosCallback(cursorPosCallback)
	window.SetMouseButtonCallback(mouseButtonCallback)
	window.SetScrollCallback(scrollCallback)

	if err := gl.Init(); err != nil {
		log.Fatalf("failed to init gl: %v", err)
	}
	log.Printf("OpenGL Version: %s", gl.GoStr(gl.GetString(gl.VERSION)))

	// Compile shaders
	vert := compileShader(vertexShaderSource, gl.VERTEX_SHADER)

	fragLine := compileShader(lineFragmentShaderSource, gl.FRAGMENT_SHADER)
	lineProgram := createProgram(vert, fragLine)
	projLineLoc := gl.GetUniformLocation(lineProgram, gl.Str("projection\x00"))
	viewLineLoc := gl.GetUniformLocation(lineProgram, gl.Str("view\x00"))
	modelLineLoc := gl.GetUniformLocation(lineProgram, gl.Str("model\x00"))

	fragPart := compileShader(particleFragmentShaderSource, gl.FRAGMENT_SHADER)
	partProgram := createProgram(vert, fragPart)
	projPartLoc := gl.GetUniformLocation(partProgram, gl.Str("projection\x00"))
	viewPartLoc := gl.GetUniformLocation(partProgram, gl.Str("view\x00"))
	modelPartLoc := gl.GetUniformLocation(partProgram, gl.Str("model\x00"))

	// Setup VAOs/VBOs
	vaoLine, vboLine := initVao()
	vaoPart, vboPart := initVao()

	gl.Viewport(0, 0, width, height)
	gl.Enable(gl.DEPTH_TEST)
	gl.DepthFunc(gl.LEQUAL)
	gl.ClearColor(0.02, 0.02, 0.05, 1.0)

	// initial projection (will be updated each frame)
	ratio := float32(width) / float32(height)
	projection := mgl32.Perspective(mgl32.DegToRad(45.0), ratio, 0.1, 100.0)

	model := mgl32.Ident4()

	// simulation
	sim := createMockData()

	var lastTime float64 = glfw.GetTime()

	// Main loop
	for !window.ShouldClose() {
		cur := glfw.GetTime()
		dt := cur - lastTime
		lastTime = cur

		// Camera: spherical -> cartesian
		cosPitch := float32(math.Cos(float64(pitch)))
		camX := cameraDistance * cosPitch * float32(math.Sin(float64(yaw)))
		camY := cameraDistance * float32(math.Sin(float64(pitch)))
		camZ := cameraDistance * cosPitch * float32(math.Cos(float64(yaw)))
		cameraPos := mgl32.Vec3{camX, camY, camZ}
		cameraTarget := mgl32.Vec3{0, 0, 0}
		cameraUp := mgl32.Vec3{0, 1, 0}
		view := mgl32.LookAtV(cameraPos, cameraTarget, cameraUp)

		// small Z oscillation based on time for depth motion
		t := float32(cur)
		for i := range sim.Particles {
			sim.Particles[i].Pos[2] += float32(math.Sin(float64(t*0.35+float32(i)*0.17))) * 0.0005
		}

		// update sim (with springs & mouse) using current projection/view/camera
		updateSimulation(sim, dt, projection, view, cameraPos)

		// Clear
		gl.Clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)

		// Send uniforms for line program
		gl.UseProgram(lineProgram)
		gl.UniformMatrix4fv(projLineLoc, 1, false, &projection[0])
		gl.UniformMatrix4fv(viewLineLoc, 1, false, &view[0])
		gl.UniformMatrix4fv(modelLineLoc, 1, false, &model[0])
		gl.UseProgram(0)

		// Send uniforms for particle program
		gl.UseProgram(partProgram)
		gl.UniformMatrix4fv(projPartLoc, 1, false, &projection[0])
		gl.UniformMatrix4fv(viewPartLoc, 1, false, &view[0])
		gl.UniformMatrix4fv(modelPartLoc, 1, false, &model[0])
		gl.UseProgram(0)

		// Draw
		drawLines(lineProgram, vboLine, vaoLine, sim)
		drawParticles(partProgram, vboPart, vaoPart, sim)

		window.SwapBuffers()
		glfw.PollEvents()

		// small sleep to avoid burning 100%
		time.Sleep(time.Millisecond * 3)
	}
}
