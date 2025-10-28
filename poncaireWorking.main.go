//go:build darwin || linux || windows

package main

import (
	"log"
	"math/rand"
	"runtime"
	"time"
	"unsafe"

	"github.com/go-gl/gl/v4.1-core/gl"
	"github.com/go-gl/glfw/v3.3/glfw"
	"github.com/go-gl/mathgl/mgl32"
)

// ----------------------------------------------------
// GLOBALS
// ----------------------------------------------------
const (
	windowWidth  = 1200
	windowHeight = 1200
	nParticles   = 1200
)

var (
	vao, vbo           uint32
	prog               uint32
	particleVboSize    = 6 // pos(3) + color(3)
	repulsionK         = float32(0.01)
	repulsionDistCheck = float32(0.2)
)

// ----------------------------------------------------
// DATA STRUCTURES
// ----------------------------------------------------
type Particle struct {
	Pos mgl32.Vec3
	Vel mgl32.Vec3
	Col mgl32.Vec3
}

// ----------------------------------------------------
// MAIN ENTRY
// ----------------------------------------------------
func main() {
	// Lock thread for OpenGL
	runtime.LockOSThread()

	// Initialize GLFW
	if err := glfw.Init(); err != nil {
		log.Fatalln("failed to initialize glfw:", err)
	}
	defer glfw.Terminate()

	glfw.WindowHint(glfw.ContextVersionMajor, 4)
	glfw.WindowHint(glfw.ContextVersionMinor, 1)
	glfw.WindowHint(glfw.OpenGLProfile, glfw.OpenGLCoreProfile)
	glfw.WindowHint(glfw.Resizable, glfw.True)

	window, err := glfw.CreateWindow(windowWidth, windowHeight, "Persistent Homology Visualizer", nil, nil)
	if err != nil {
		panic(err)
	}
	window.MakeContextCurrent()

	if err := gl.Init(); err != nil {
		panic(err)
	}

	log.Println("OpenGL version", gl.GoStr(gl.GetString(gl.VERSION)))

	prog = createShaderProgram(vertexShaderSource, fragmentShaderSource)
	setupBuffers()

	// Initialize particle data
	particles := make([]Particle, nParticles)
	for i := range particles {
		particles[i].Pos = mgl32.Vec3{
			rand.Float32()*2 - 1,
			rand.Float32()*2 - 1,
			rand.Float32()*2 - 1,
		}
		particles[i].Vel = mgl32.Vec3{0, 0, 0}
		particles[i].Col = mgl32.Vec3{rand.Float32(), rand.Float32(), rand.Float32()}
	}

	// Main loop
	last := time.Now()
	for !window.ShouldClose() {
		now := time.Now()
		dt := float32(now.Sub(last).Seconds())
		last = now

		updateParticles(particles, dt)
		updateBuffer(particles)
		drawScene()

		window.SwapBuffers()
		glfw.PollEvents()
	}
}

// ----------------------------------------------------
// PHYSICS / PARTICLE LOGIC
// ----------------------------------------------------
func updateParticles(particles []Particle, dt float32) {
	for i := range particles {
		for j := range particles {
			if i == j {
				continue
			}
			force := getInterSphereRepulsion(particles[i].Pos, particles[j].Pos)
			particles[i].Vel = particles[i].Vel.Add(force.Mul(dt))
		}

		// Simple Euler integration
		particles[i].Pos = particles[i].Pos.Add(particles[i].Vel.Mul(dt))

		// Boundaries
		for k := 0; k < 3; k++ {
			if particles[i].Pos[k] > 1 || particles[i].Pos[k] < -1 {
				particles[i].Vel[k] *= -0.5
			}
		}
	}
}

// ----------------------------------------------------
// TOPOLOGICAL ANALYSIS PLACEHOLDER (BETA-0, ETC.)
// ----------------------------------------------------
func TopologicalAnalysisModule(particles []Particle) {
	// This is where you'd compute Betti numbers or persistent homology.
	// For now, it's a stub to connect your real-time TDA pipeline.
}

// ----------------------------------------------------
// REPULSION FUNCTION
// ----------------------------------------------------
func getInterSphereRepulsion(pos, otherCenter mgl32.Vec3) mgl32.Vec3 {
	vec := pos.Sub(otherCenter)
	dist := vec.Len()

	if dist > repulsionDistCheck || dist == 0 {
		return mgl32.Vec3{0, 0, 0}
	}

	// Inverse distance squared falloff for repulsion: F = k / d^2
	forceMag := repulsionK / (dist * dist)
	dir := vec.Normalize()
	return dir.Mul(forceMag)
}

// ----------------------------------------------------
// OPENGL BUFFERS
// ----------------------------------------------------
func setupBuffers() {
	gl.GenVertexArrays(1, &vao)
	gl.BindVertexArray(vao)

	gl.GenBuffers(1, &vbo)
	gl.BindBuffer(gl.ARRAY_BUFFER, vbo)

	bufferSize := nParticles * particleVboSize * int(unsafe.Sizeof(float32(0)))
	gl.BufferData(gl.ARRAY_BUFFER, bufferSize, nil, gl.DYNAMIC_DRAW)

	stride := int32(particleVboSize * int(unsafe.Sizeof(float32(0))))

	// Position attribute (layout = 0)
	gl.EnableVertexAttribArray(0)
	gl.VertexAttribPointer(0, 3, gl.FLOAT, false, stride, gl.PtrOffset(0))

	// Color attribute (layout = 1)
	gl.EnableVertexAttribArray(1)
	gl.VertexAttribPointer(1, 3, gl.FLOAT, false, stride, gl.PtrOffset(3*4))

	gl.BindVertexArray(0)
}

// ----------------------------------------------------
// UPDATE BUFFER
// ----------------------------------------------------
func updateBuffer(particles []Particle) {
	gl.BindBuffer(gl.ARRAY_BUFFER, vbo)

	data := make([]float32, 0, len(particles)*particleVboSize)
	for _, p := range particles {
		data = append(data, p.Pos[:]...)
		data = append(data, p.Col[:]...)
	}

	gl.BufferSubData(gl.ARRAY_BUFFER, 0, len(data)*4, gl.Ptr(data))
}

// ----------------------------------------------------
// DRAW SCENE
// ----------------------------------------------------
func drawScene() {
	gl.Clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)
	gl.UseProgram(prog)
	gl.BindVertexArray(vao)
	gl.DrawArrays(gl.POINTS, 0, int32(nParticles))
	gl.BindVertexArray(0)
}

// ----------------------------------------------------
// SHADERS
// ----------------------------------------------------
var vertexShaderSource = `
#version 410 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
out vec3 vColor;
void main() {
    gl_Position = vec4(aPos, 1.0);
    vColor = aColor;
    gl_PointSize = 4.0;
}
` + "\x00"

var fragmentShaderSource = `
#version 410 core
in vec3 vColor;
out vec4 FragColor;
void main() {
    FragColor = vec4(vColor, 1.0);
}
` + "\x00"

// ----------------------------------------------------
// SHADER COMPILATION
// ----------------------------------------------------
func createShaderProgram(vsSrc, fsSrc string) uint32 {
	vertexShader := gl.CreateShader(gl.VERTEX_SHADER)
	csources, free := gl.Strs(vsSrc)
	gl.ShaderSource(vertexShader, 1, csources, nil)
	free()
	gl.CompileShader(vertexShader)

	var status int32
	gl.GetShaderiv(vertexShader, gl.COMPILE_STATUS, &status)
	if status == gl.FALSE {
		logShaderError(vertexShader, "VERTEX")
	}

	fragmentShader := gl.CreateShader(gl.FRAGMENT_SHADER)
	csources, free = gl.Strs(fsSrc)
	gl.ShaderSource(fragmentShader, 1, csources, nil)
	free()
	gl.CompileShader(fragmentShader)
	gl.GetShaderiv(fragmentShader, gl.COMPILE_STATUS, &status)
	if status == gl.FALSE {
		logShaderError(fragmentShader, "FRAGMENT")
	}

	program := gl.CreateProgram()
	gl.AttachShader(program, vertexShader)
	gl.AttachShader(program, fragmentShader)
	gl.LinkProgram(program)

	gl.GetProgramiv(program, gl.LINK_STATUS, &status)
	if status == gl.FALSE {
		logProgramError(program)
	}

	gl.DeleteShader(vertexShader)
	gl.DeleteShader(fragmentShader)

	return program
}

func logShaderError(shader uint32, shaderType string) {
	var logLength int32
	gl.GetShaderiv(shader, gl.INFO_LOG_LENGTH, &logLength)
	logStr := make([]byte, logLength+1)
	gl.GetShaderInfoLog(shader, logLength, nil, &logStr[0])
	log.Fatalf("%s shader compile error: %s", shaderType, logStr)
}

func logProgramError(program uint32) {
	var logLength int32
	gl.GetProgramiv(program, gl.INFO_LOG_LENGTH, &logLength)
	logStr := make([]byte, logLength+1)
	gl.GetProgramInfoLog(program, logLength, nil, &logStr[0])
	log.Fatalf("program link error: %s", logStr)
}
