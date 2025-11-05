// main.go
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

const (
	numParticles    = 2000
	rTarget         = 2.0
	rSpringK        = 1.0
	angularSpringK  = 0.05
	radialDamping   = 0.92
	angularDamping  = 0.96
	timeStep        = 0.016
	autoRotateSpeed = 0.2 // degrees per frame
	windowWidth     = 1280
	windowHeight    = 720
)

type Particle struct {
	r, theta, phi    float32
	vr, vtheta, vphi float32
}

func init() { runtime.LockOSThread() }

func main() {
	if err := glfw.Init(); err != nil {
		log.Fatalln("failed to initialize glfw:", err)
	}
	defer glfw.Terminate()

	glfw.WindowHint(glfw.Resizable, glfw.True)
	glfw.WindowHint(glfw.ContextVersionMajor, 4)
	glfw.WindowHint(glfw.ContextVersionMinor, 1)
	glfw.WindowHint(glfw.OpenGLProfile, glfw.OpenGLCoreProfile)
	glfw.WindowHint(glfw.OpenGLForwardCompatible, glfw.True)

	window, err := glfw.CreateWindow(windowWidth, windowHeight, "Polar Particle System", nil, nil)
	if err != nil {
		log.Fatalln("failed to create window:", err)
	}
	window.MakeContextCurrent()
	if err := gl.Init(); err != nil {
		log.Fatalln("failed to initialize OpenGL:", err)
	}
	gl.Enable(gl.DEPTH_TEST)

	program := newProgram(vertexShaderSource, fragmentShaderSource)
	gl.UseProgram(program)

	proj := mgl32.Perspective(mgl32.DegToRad(45.0), float32(windowWidth)/windowHeight, 0.1, 100.0)
	projUniform := gl.GetUniformLocation(program, gl.Str("projection\x00"))
	viewUniform := gl.GetUniformLocation(program, gl.Str("view\x00"))
	modelUniform := gl.GetUniformLocation(program, gl.Str("model\x00"))
	gl.UniformMatrix4fv(projUniform, 1, false, &proj[0])

	particles := make([]Particle, numParticles)
	for i := range particles {
		particles[i].r = rTarget + rand.Float32()*0.5 - 0.25
		particles[i].theta = rand.Float32() * math.Pi
		particles[i].phi = rand.Float32() * 2 * math.Pi
	}

	points := make([]float32, numParticles*3)
	var vao, vbo uint32
	gl.GenVertexArrays(1, &vao)
	gl.GenBuffers(1, &vbo)
	gl.BindVertexArray(vao)
	gl.BindBuffer(gl.ARRAY_BUFFER, vbo)
	gl.BufferData(gl.ARRAY_BUFFER, len(points)*4, gl.Ptr(points), gl.DYNAMIC_DRAW)

	posAttrib := uint32(gl.GetAttribLocation(program, gl.Str("position\x00")))
	gl.EnableVertexAttribArray(posAttrib)
	gl.VertexAttribPointer(posAttrib, 3, gl.FLOAT, false, 0, nil)

	// Camera rotation variables
	var yaw, pitch float32
	lastX, lastY := float64(windowWidth/2), float64(windowHeight/2)
	mousePressed := false
	window.SetCursorPosCallback(func(w *glfw.Window, xpos, ypos float64) {
		if mousePressed {
			xoffset := float32(xpos - lastX)
			yoffset := float32(ypos - lastY)
			yaw += xoffset * 0.3
			pitch += yoffset * 0.3
			if pitch > 89 {
				pitch = 89
			}
			if pitch < -89 {
				pitch = -89
			}
		}
		lastX, lastY = xpos, ypos
	})
	window.SetMouseButtonCallback(func(w *glfw.Window, button glfw.MouseButton, action glfw.Action, mods glfw.ModifierKey) {
		if button == glfw.MouseButtonLeft {
			mousePressed = (action == glfw.Press)
		}
	})

	lastTime := time.Now()
	autoAngle := float32(0)
	for !window.ShouldClose() {
		now := time.Now()
		dt := float32(now.Sub(lastTime).Seconds())
		lastTime = now

		updateParticles(particles, dt)
		for i, p := range particles {
			sinT, cosT := float32(math.Sin(float64(p.theta))), float32(math.Cos(float64(p.theta)))
			sinP, cosP := float32(math.Sin(float64(p.phi))), float32(math.Cos(float64(p.phi)))
			points[i*3+0] = p.r * sinT * cosP
			points[i*3+1] = p.r * cosT
			points[i*3+2] = p.r * sinT * sinP
		}

		gl.BindBuffer(gl.ARRAY_BUFFER, vbo)
		gl.BufferSubData(gl.ARRAY_BUFFER, 0, len(points)*4, gl.Ptr(points))

		gl.Clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)
		gl.ClearColor(0.02, 0.02, 0.05, 1.0)

		// Camera
		autoAngle += autoRotateSpeed * dt
		yawTotal := yaw + autoAngle
		camX := float32(math.Sin(float64(mgl32.DegToRad(yawTotal)))) * 6
		camZ := float32(math.Cos(float64(mgl32.DegToRad(yawTotal)))) * 6
		camY := float32(math.Sin(float64(mgl32.DegToRad(pitch)))) * 3

		view := mgl32.LookAt(camX, camY, camZ, 0, 0, 0, 0, 1, 0)
		model := mgl32.Ident4()
		gl.UniformMatrix4fv(viewUniform, 1, false, &view[0])
		gl.UniformMatrix4fv(modelUniform, 1, false, &model[0])

		gl.BindVertexArray(vao)
		gl.DrawArrays(gl.POINTS, 0, int32(numParticles))

		window.SwapBuffers()
		glfw.PollEvents()
	}
}

func updateParticles(particles []Particle, dt float32) {
	for i := range particles {
		p := &particles[i]

		// Springs toward equilibrium
		ar := -rSpringK * (p.r - rTarget)
		atheta := -angularSpringK * p.theta
		aphi := -angularSpringK * p.phi

		// Semi-implicit integration
		p.vr += ar * dt
		p.vtheta += atheta * dt
		p.vphi += aphi * dt

		p.vr *= radialDamping
		p.vtheta *= angularDamping
		p.vphi *= angularDamping

		p.r += p.vr * dt
		p.theta += p.vtheta * dt
		p.phi += p.vphi * dt
	}
}

var vertexShaderSource = `
#version 410 core
layout (location = 0) in vec3 position;
uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;
void main() {
    gl_Position = projection * view * model * vec4(position, 1.0);
    gl_PointSize = 2.0;
}` + "\x00"

var fragmentShaderSource = `
#version 410 core
out vec4 FragColor;
void main() {
    FragColor = vec4(0.4, 0.8, 1.0, 1.0);
}` + "\x00"

func newProgram(vsSource, fsSource string) uint32 {
	vs := compileShader(vsSource, gl.VERTEX_SHADER)
	fs := compileShader(fsSource, gl.FRAGMENT_SHADER)
	program := gl.CreateProgram()
	gl.AttachShader(program, vs)
	gl.AttachShader(program, fs)
	gl.LinkProgram(program)
	var status int32
	gl.GetProgramiv(program, gl.LINK_STATUS, &status)
	if status == gl.FALSE {
		log.Fatal("shader link failed")
	}
	gl.DeleteShader(vs)
	gl.DeleteShader(fs)
	return program
}

func compileShader(source string, shaderType uint32) uint32 {
	shader := gl.CreateShader(shaderType)
	csources, free := gl.Strs(source)
	gl.ShaderSource(shader, 1, csources, nil)
	free()
	gl.CompileShader(shader)
	var status int32
	gl.GetShaderiv(shader, gl.COMPILE_STATUS, &status)
	if status == gl.FALSE {
		log.Fatal("shader compile failed")
	}
	return shader
}
