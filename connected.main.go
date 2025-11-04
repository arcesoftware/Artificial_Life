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
	width  = 1200
	height = 900
	title  = "Interactive Glowing Particle Network"

	// Normalized boundaries
	BOUNDARY = 1.0

	// Particle rendering settings
	NODE_SIZE         = 0.015 // Half-width for the particle square (before glow)
	MAX_CONN_DISTANCE = 0.5   // Max distance for full line intensity

	// Interactivity settings
	MOUSE_FORCE_RADIUS       = 0.4
	MOUSE_REPULSION_STRENGTH = 0.5
	PARTICLE_COUNT           = 50
)

var (
	// Global state for mouse position (in normalized GL coordinates [-1, 1])
	mousePos mgl32.Vec2
)

// --- Shader Sources ---

// Shared Vertex Shader: Reads Position (layout 0) and Color (layout 1), passes color and position
var vertexShaderSource = `
#version 410
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 color;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

out vec3 vertColor;
out vec3 vertPosition; // Pass world position for fragment shader calculations

void main() {
	vec4 worldPos = model * vec4(position, 1.0);
	gl_Position = projection * view * worldPos;
	vertColor = color;
	vertPosition = worldPos.xyz;
}
` + "\x00"

// Line Fragment Shader: Simple pass-through for line segments
var lineFragmentShaderSource = `
#version 410
in vec3 vertColor;
out vec4 fragColor;

void main() {
	fragColor = vec4(vertColor, 1.0);
}
` + "\x00"

// Particle Fragment Shader: Creates the soft, circular glow effect
var particleFragmentShaderSource = `
#version 410
in vec3 vertColor;
in vec3 vertPosition;

out vec4 fragColor;

void main() {
	// Calculate the distance from the fragment to the center of the particle square
	// Since we draw a square around the particle center (vertPosition), 
	// we use gl_FragCoord or gl_PointCoord (not available here), so we rely on smoothstep 
	// based on distance from the center.

	// A simplified way is to calculate the distance from the center of the square (vertPosition)
	// assuming the vertices were correctly scaled to draw the square.
	// For simplicity and robustness on all systems, we rely on the smooth transition 
	// based on the particle's own color, using the position passed from the vertex shader.

	// Fallback/alternative using distance from center:
	// We calculate the distance from the center of the quad to the fragment's position 
	// (requires complex coordinate transformations or geometry shaders, which we avoid).

	// We'll use a strong, bright glow effect via additive blending on the simple square geometry
	// and simply rely on the passed color, letting the additive blending do the work.
	
	// Set the color and rely on additive blending for the glow effect
	// The particle color is boosted slightly for extra brightness in the center
	fragColor = vec4(vertColor * 2.5, 0.8);
}
` + "\x00"

// --- Data Structures ---

// Particle represents a single particle's state.
type Particle struct {
	Pos mgl32.Vec3
	Col mgl32.Vec3
	Vel mgl32.Vec3
}

// SimulationData holds all dynamic data.
type SimulationData struct {
	Particles []Particle
	NextIdx   []int // Index of the next particle in a connection, or -1
}

// --- GLFW and OpenGL Helper Functions ---

// keyCallback handles keyboard input (e.g., closing the window)
func keyCallback(window *glfw.Window, key glfw.Key, scancode int, action glfw.Action, mods glfw.ModifierKey) {
	if key == glfw.KeyEscape && action == glfw.Press {
		window.SetShouldClose(true)
	}
}

// cursorPosCallback updates the global mouse position in normalized GL coordinates [-1, 1]
func cursorPosCallback(window *glfw.Window, xpos float64, ypos float64) {
	// Map screen coordinates (0 to width/height) to GL normalized coordinates (-1 to 1)
	w, h := window.GetSize()
	x := float32(xpos/float64(w)*2.0 - 1.0)
	// The y-axis in GL starts at the bottom, GLFW starts at the top
	y := float32(1.0 - ypos/float64(h)*2.0)
	mousePos = mgl32.Vec2{x, y}
}

// compileShader compiles the given shader source and checks for errors.
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
		logStr := make([]byte, logLength)
		gl.GetShaderInfoLog(shader, logLength, nil, &logStr[0])
		log.Fatalf("failed to compile shader %v: %v", shaderType, gl.GoStr(&logStr[0]))
	}
	return shader
}

// createProgram links the compiled shaders into a program.
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
		logStr := make([]byte, logLength)
		gl.GetProgramInfoLog(program, logLength, nil, &logStr[0])
		log.Fatalf("failed to link program: %v", gl.GoStr(&logStr[0]))
	}

	gl.DeleteShader(vertexShader)
	gl.DeleteShader(fragmentShader)

	return program
}

// initVao sets up the Vertex Array Object (VAO) and Vertex Buffer Object (VBO).
func initVao() (uint32, uint32) {
	var vao uint32
	gl.GenVertexArrays(1, &vao)
	gl.BindVertexArray(vao)

	var vbo uint32
	gl.GenBuffers(1, &vbo)
	gl.BindBuffer(gl.ARRAY_BUFFER, vbo)

	// Stride is the size of one complete vertex: 3 Pos floats + 3 Col floats = 6 floats * 4 bytes/float = 24 bytes
	stride := int32(6 * 4)

	// Position attribute (Layout 0)
	gl.EnableVertexAttribArray(0)
	gl.VertexAttribPointerWithOffset(0, 3, gl.FLOAT, false, stride, 0) // Offset 0 bytes

	// Color attribute (Layout 1)
	gl.EnableVertexAttribArray(1)
	gl.VertexAttribPointerWithOffset(1, 3, gl.FLOAT, false, stride, uintptr(3*4)) // Offset 3 * 4 bytes

	gl.BindVertexArray(0)
	gl.BindBuffer(gl.ARRAY_BUFFER, 0)

	return vao, vbo
}

// --- Simulation Update Logic ---

// updateSimulation moves the particles, handles boundary collision, and applies mouse force.
func updateSimulation(simData *SimulationData, deltaTime float64) {
	dt := float32(deltaTime)
	const damping = 0.99 // Gentle damping

	for i := range simData.Particles {
		p := &simData.Particles[i]

		// 1. Apply Mouse Repulsion Force
		// Convert the particle's 3D position (XY) to 2D for mouse interaction
		pPos2D := p.Pos.Vec2()
		mouseVec := mousePos.Sub(pPos2D)
		dist := mouseVec.Len()

		if dist < MOUSE_FORCE_RADIUS && dist > 0.0001 {
			// Calculate repulsion force strength based on inverse distance squared
			// Repulsion is strongest near the cursor, fades out towards MOUSE_FORCE_RADIUS
			forceMagnitude := MOUSE_REPULSION_STRENGTH * (1.0 - dist/MOUSE_FORCE_RADIUS)

			// Repulsion vector (normalized direction * force)
			repulsionVec := mouseVec.Normalize().Mul(-forceMagnitude)

			// Apply force as acceleration (Vel += Accel * dt)
			p.Vel = p.Vel.Add(repulsionVec.Vec3(0.0).Mul(dt))
		}

		// 2. Update position based on velocity and delta time
		p.Pos = p.Pos.Add(p.Vel.Mul(dt))

		// 3. Handle Boundary Collisions (Bounce off edges)
		if p.Pos.X() > BOUNDARY || p.Pos.X() < -BOUNDARY {
			p.Vel[0] = -p.Vel[0] * damping // Reverse X velocity and apply damping
			if p.Pos.X() > BOUNDARY {
				p.Pos[0] = BOUNDARY
			} else {
				p.Pos[0] = -BOUNDARY
			}
		}

		if p.Pos.Y() > BOUNDARY || p.Pos.Y() < -BOUNDARY {
			p.Vel[1] = -p.Vel[1] * damping // Reverse Y velocity and apply damping
			if p.Pos.Y() > BOUNDARY {
				p.Pos[1] = BOUNDARY
			} else {
				p.Pos[1] = -BOUNDARY
			}
		}

		// 4. Apply Overall Damping
		p.Vel = p.Vel.Mul(damping)
	}
}

// --- Drawing Logic ---

// drawLines draws connections with dynamic thickness/color/fade.
func drawLines(program uint32, vbo uint32, vao uint32, simData *SimulationData) {
	nParticles := len(simData.Particles)
	var lineVertices []float32
	var lineVertexCount int32

	lineVertices = make([]float32, 0, nParticles*2*6)
	lineBaseCol := mgl32.Vec3{0.3, 0.3, 0.3} // Darker base line color

	for i := 0; i < nParticles; i++ {
		next := simData.NextIdx[i]
		if next != i && next != -1 && next < nParticles {
			p1 := simData.Particles[i]
			p2 := simData.Particles[next]

			// --- Calculation for Dynamic Effect ---
			dist := p1.Pos.Sub(p2.Pos).Len()

			// Calculate fade factor: full opacity/color when dist=0, 0.0 when dist >= MAX_CONN_DISTANCE
			fadeFactor := float32(1.0 - float32(math.Min(float64(dist), float64(MAX_CONN_DISTANCE)))/MAX_CONN_DISTANCE)
			if fadeFactor <= 0.01 {
				continue // Skip drawing if completely faded
			}

			// Interpolated color based on distance and particle color
			p1Col := p1.Col.Add(lineBaseCol).Mul(0.5).Mul(fadeFactor)
			p2Col := p2.Col.Add(lineBaseCol).Mul(0.5).Mul(fadeFactor)

			// --- Append data for P1 (Start Point) ---
			lineVertices = append(lineVertices, p1.Pos.X(), p1.Pos.Y(), p1.Pos.Z()) // Position
			lineVertices = append(lineVertices, p1Col.X(), p1Col.Y(), p1Col.Z())    // Color

			// --- Append data for P2 (End Point) ---
			lineVertices = append(lineVertices, p2.Pos.X(), p2.Pos.Y(), p2.Pos.Z()) // Position
			lineVertices = append(lineVertices, p2Col.X(), p2Col.Y(), p2Col.Z())    // Color

			lineVertexCount += 2
		}
	}

	if lineVertexCount == 0 {
		return // Nothing to draw
	}

	// 2. Upload the new line data to the VBO
	gl.BindBuffer(gl.ARRAY_BUFFER, vbo)
	dataSize := int(lineVertexCount) * 6 * 4
	gl.BufferData(gl.ARRAY_BUFFER, dataSize, gl.Ptr(lineVertices), gl.DYNAMIC_DRAW)

	// 3. Draw Lines
	gl.UseProgram(program)
	gl.BindVertexArray(vao)

	gl.Enable(gl.BLEND)
	gl.BlendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA) // Standard alpha blending for lines

	gl.LineWidth(3.0) // Thicker lines

	gl.DrawArrays(gl.LINES, 0, lineVertexCount)

	gl.Disable(gl.BLEND)
	gl.BindVertexArray(0)
}

// drawParticles draws the particles themselves as small, glowing squares.
func drawParticles(program uint32, vbo uint32, vao uint32, simData *SimulationData) {
	nParticles := len(simData.Particles)
	var particleVertices []float32
	// 6 vertices per particle (2 triangles), 6 floats per vertex (3 pos, 3 col)
	particleVertices = make([]float32, 0, nParticles*6*6)

	for _, p := range simData.Particles {
		x := p.Pos.X()
		y := p.Pos.Y()
		col := p.Col

		// Triangle 1 (BL, TL, TR)
		particleVertices = append(particleVertices, x-NODE_SIZE, y-NODE_SIZE, p.Pos.Z()) // BL Pos
		particleVertices = append(particleVertices, col.X(), col.Y(), col.Z())           // BL Col
		particleVertices = append(particleVertices, x-NODE_SIZE, y+NODE_SIZE, p.Pos.Z()) // TL Pos
		particleVertices = append(particleVertices, col.X(), col.Y(), col.Z())           // TL Col
		particleVertices = append(particleVertices, x+NODE_SIZE, y+NODE_SIZE, p.Pos.Z()) // TR Pos
		particleVertices = append(particleVertices, col.X(), col.Y(), col.Z())           // TR Col

		// Triangle 2 (BL, TR, BR)
		particleVertices = append(particleVertices, x-NODE_SIZE, y-NODE_SIZE, p.Pos.Z()) // BL Pos
		particleVertices = append(particleVertices, col.X(), col.Y(), col.Z())           // BL Col
		particleVertices = append(particleVertices, x+NODE_SIZE, y+NODE_SIZE, p.Pos.Z()) // TR Pos
		particleVertices = append(particleVertices, col.X(), col.Y(), col.Z())           // TR Col
		particleVertices = append(particleVertices, x+NODE_SIZE, y-NODE_SIZE, p.Pos.Z()) // BR Pos
		particleVertices = append(particleVertices, col.X(), col.Y(), col.Z())           // BR Col
	}

	vertexCount := int32(len(particleVertices) / 6) // Total number of vertices

	// 2. Upload the new particle data to the VBO
	gl.BindBuffer(gl.ARRAY_BUFFER, vbo)
	dataSize := len(particleVertices) * 4 // float32 is 4 bytes
	gl.BufferData(gl.ARRAY_BUFFER, dataSize, gl.Ptr(particleVertices), gl.DYNAMIC_DRAW)

	// 3. Draw Particles (Triangles)
	gl.UseProgram(program)
	gl.BindVertexArray(vao)

	// --- GLOW EFFECT: Additive Blending ---
	gl.Enable(gl.BLEND)
	// Additive blending: Source color (particle) is added to destination (screen)
	gl.BlendFunc(gl.SRC_ALPHA, gl.ONE)

	gl.DrawArrays(gl.TRIANGLES, 0, vertexCount)

	gl.Disable(gl.BLEND)
	gl.BindVertexArray(0)
}

// --- Mock Simulation and Main Loop ---

// createMockData creates animated particles and random connections.
func createMockData() *SimulationData {
	particles := make([]Particle, PARTICLE_COUNT)
	nextIdx := make([]int, PARTICLE_COUNT)

	for i := 0; i < PARTICLE_COUNT; i++ {
		// Random position within a smaller central area
		randPos := func() float32 { return float32(rand.Float64()*1.6 - 0.8) }
		// Random velocity (slow, normalized to 1)
		randVel := func() float32 { return float32(rand.Float64()*0.4 - 0.2) }
		randCol := mgl32.Vec3{float32(rand.Float64()), float32(rand.Float64()), float32(rand.Float64())}

		particles[i] = Particle{
			Pos: mgl32.Vec3{randPos(), randPos(), 0.0},
			Col: randCol,
			Vel: mgl32.Vec3{randVel(), randVel(), 0.0}.Normalize().Mul(0.2), // Slower movement
		}

		// Create random sequential connections that loop
		nextIdx[i] = (i + 1) % PARTICLE_COUNT
	}

	return &SimulationData{
		Particles: particles,
		NextIdx:   nextIdx,
	}
}

// --- Main Application Entry Point ---

func main() {
	runtime.LockOSThread()

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
		log.Fatalf("failed to create glfw window: %v", err)
	}
	window.MakeContextCurrent()

	// Set input callbacks
	window.SetKeyCallback(keyCallback)
	window.SetCursorPosCallback(cursorPosCallback)

	if err := gl.Init(); err != nil {
		log.Fatalf("failed to initialize gl: %v", err)
	}
	log.Printf("OpenGL Version: %s", gl.GoStr(gl.GetString(gl.VERSION)))

	// 1. Compile Shaders and Create Programs
	vertShader := compileShader(vertexShaderSource, gl.VERTEX_SHADER)

	// Program for LINES (using simple pass-through fragment shader)
	fragLineShader := compileShader(lineFragmentShaderSource, gl.FRAGMENT_SHADER)
	lineProgram := createProgram(vertShader, fragLineShader)
	projUniformLine := gl.GetUniformLocation(lineProgram, gl.Str("projection\x00"))
	viewUniformLine := gl.GetUniformLocation(lineProgram, gl.Str("view\x00"))
	modelUniformLine := gl.GetUniformLocation(lineProgram, gl.Str("model\x00"))

	// Program for PARTICLES (using glow fragment shader)
	fragParticleShader := compileShader(particleFragmentShaderSource, gl.FRAGMENT_SHADER)
	particleProgram := createProgram(vertShader, fragParticleShader)
	projUniformParticle := gl.GetUniformLocation(particleProgram, gl.Str("projection\x00"))
	viewUniformParticle := gl.GetUniformLocation(particleProgram, gl.Str("view\x00"))
	modelUniformParticle := gl.GetUniformLocation(particleProgram, gl.Str("model\x00"))

	// 2. Setup VAO and VBO
	vaoLine, vboLine := initVao() // For lines
	vaoPart, vboPart := initVao() // For particles

	// 3. Global GL State
	gl.Viewport(0, 0, width, height)
	gl.Enable(gl.DEPTH_TEST)
	gl.ClearColor(0.0, 0.0, 0.05, 1.0) // Very dark blue/black background

	// Projection Matrix (Orthographic)
	ratio := float32(width) / float32(height)
	projection := mgl32.Ortho(-ratio*1.1, ratio*1.1, -1.1, 1.1, 0.1, 100.0)

	// View Matrix (Camera position)
	cameraPos := mgl32.Vec3{0.0, 0.0, 3.0}
	cameraTarget := mgl32.Vec3{0.0, 0.0, 0.0}
	cameraUp := mgl32.Vec3{0.0, 1.0, 0.0}
	view := mgl32.LookAtV(cameraPos, cameraTarget, cameraUp)

	// Model Matrix (Identity for now, no rotation/scaling)
	model := mgl32.Ident4()

	// Create simulation data
	simData := createMockData()

	var lastTime float64
	var currentTime float64

	// 4. Main Render Loop
	for !window.ShouldClose() {
		currentTime = glfw.GetTime()
		deltaTime := currentTime - lastTime
		lastTime = currentTime

		// --- SIMULATION STEP ---
		updateSimulation(simData, deltaTime)
		// -----------------------

		// Clear screen
		gl.Clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)

		// Set uniforms for LINE PROGRAM
		gl.UseProgram(lineProgram)
		gl.UniformMatrix4fv(projUniformLine, 1, false, &projection[0])
		gl.UniformMatrix4fv(viewUniformLine, 1, false, &view[0])
		gl.UniformMatrix4fv(modelUniformLine, 1, false, &model[0])
		gl.UseProgram(0)

		// Set uniforms for PARTICLE PROGRAM
		gl.UseProgram(particleProgram)
		gl.UniformMatrix4fv(projUniformParticle, 1, false, &projection[0])
		gl.UniformMatrix4fv(viewUniformParticle, 1, false, &view[0])
		gl.UniformMatrix4fv(modelUniformParticle, 1, false, &model[0])
		gl.UseProgram(0)

		// 1. Draw the Lines
		drawLines(lineProgram, vboLine, vaoLine, simData)

		// 2. Draw the Particles (with additive glow)
		drawParticles(particleProgram, vboPart, vaoPart, simData)

		// Poll events and swap buffers
		window.SwapBuffers()
		glfw.PollEvents()

		// Control frame rate
		time.Sleep(time.Millisecond * 4)
	}
}
