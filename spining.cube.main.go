// main.go
package main

import (
	"log"
	"math"
	"runtime"
	"time"

	"github.com/go-gl/gl/v4.1-core/gl"
	"github.com/go-gl/glfw/v3.3/glfw"
	"github.com/go-gl/mathgl/mgl32"
)

const (
	width  = 1280
	height = 720
)

func init() { runtime.LockOSThread() }

type Mesh struct {
	vao, vbo uint32
	count    int32
}

func makeShader(src string, stype uint32) uint32 {
	s := gl.CreateShader(stype)
	csrc, free := gl.Strs(src + "\x00")
	gl.ShaderSource(s, 1, csrc, nil)
	free()
	gl.CompileShader(s)
	var ok int32
	gl.GetShaderiv(s, gl.COMPILE_STATUS, &ok)
	if ok == gl.FALSE {
		var logLen int32
		gl.GetShaderiv(s, gl.INFO_LOG_LENGTH, &logLen)
		buf := make([]byte, logLen)
		gl.GetShaderInfoLog(s, logLen, nil, &buf[0])
		log.Fatalf("shader compile error: %s", buf)
	}
	return s
}

func newProgram(vs, fs string) uint32 {
	v := makeShader(vs, gl.VERTEX_SHADER)
	f := makeShader(fs, gl.FRAGMENT_SHADER)
	p := gl.CreateProgram()
	gl.AttachShader(p, v)
	gl.AttachShader(p, f)
	gl.LinkProgram(p)
	gl.DeleteShader(v)
	gl.DeleteShader(f)
	return p
}

// Create a simple colored XYZ frame
func makeFrame() *Mesh {
	verts := []float32{
		0, 0, 0, 1, 0, 0, // X axis (red)
		1, 0, 0, 1, 0, 0,
		0, 0, 0, 0, 1, 0, // Y axis (green)
		0, 1, 0, 0, 1, 0,
		0, 0, 0, 0, 0, 1, // Z axis (blue)
		0, 0, 1, 0, 0, 1,
	}
	var vao, vbo uint32
	gl.GenVertexArrays(1, &vao)
	gl.GenBuffers(1, &vbo)
	gl.BindVertexArray(vao)
	gl.BindBuffer(gl.ARRAY_BUFFER, vbo)
	gl.BufferData(gl.ARRAY_BUFFER, len(verts)*4, gl.Ptr(verts), gl.STATIC_DRAW)
	gl.VertexAttribPointer(0, 3, gl.FLOAT, false, 6*4, gl.PtrOffset(0))
	gl.EnableVertexAttribArray(0)
	gl.VertexAttribPointer(1, 3, gl.FLOAT, false, 6*4, gl.PtrOffset(3*4))
	gl.EnableVertexAttribArray(1)
	return &Mesh{vao, vbo, int32(len(verts) / 6)}
}

// Simple cube mesh
func makeCube() *Mesh {
	verts := []float32{
		// positions        // colors
		-0.5, -0.5, -0.5, 0.7, 0.7, 1.0,
		0.5, -0.5, -0.5, 0.7, 0.7, 1.0,
		0.5, 0.5, -0.5, 0.7, 0.7, 1.0,
		-0.5, 0.5, -0.5, 0.7, 0.7, 1.0,
		-0.5, -0.5, 0.5, 0.7, 0.7, 1.0,
		0.5, -0.5, 0.5, 0.7, 0.7, 1.0,
		0.5, 0.5, 0.5, 0.7, 0.7, 1.0,
		-0.5, 0.5, 0.5, 0.7, 0.7, 1.0,
	}
	indices := []uint32{
		0, 1, 2, 2, 3, 0,
		4, 5, 6, 6, 7, 4,
		0, 4, 7, 7, 3, 0,
		1, 5, 6, 6, 2, 1,
		3, 2, 6, 6, 7, 3,
		0, 1, 5, 5, 4, 0,
	}
	var vao, vbo, ebo uint32
	gl.GenVertexArrays(1, &vao)
	gl.GenBuffers(1, &vbo)
	gl.GenBuffers(1, &ebo)
	gl.BindVertexArray(vao)
	gl.BindBuffer(gl.ARRAY_BUFFER, vbo)
	gl.BufferData(gl.ARRAY_BUFFER, len(verts)*4, gl.Ptr(verts), gl.STATIC_DRAW)
	gl.BindBuffer(gl.ELEMENT_ARRAY_BUFFER, ebo)
	gl.BufferData(gl.ELEMENT_ARRAY_BUFFER, len(indices)*4, gl.Ptr(indices), gl.STATIC_DRAW)
	gl.VertexAttribPointer(0, 3, gl.FLOAT, false, 6*4, gl.PtrOffset(0))
	gl.EnableVertexAttribArray(0)
	gl.VertexAttribPointer(1, 3, gl.FLOAT, false, 6*4, gl.PtrOffset(3*4))
	gl.EnableVertexAttribArray(1)
	return &Mesh{vao, vbo, int32(len(indices))}
}

func makeSphere(radius float32, rings, sectors int) *Mesh {
	var verts []float32
	for r := 0; r <= rings; r++ {
		for s := 0; s <= sectors; s++ {
			y := float32(math.Sin(math.Pi*float64(r)/float64(rings) - math.Pi/2))
			x := float32(math.Cos(2 * math.Pi * float64(s) / float64(sectors)))
			z := float32(math.Sin(2 * math.Pi * float64(s) / float64(sectors)))
			verts = append(verts, radius*x, radius*y, radius*z, 1, 0.8, 0.2)
		}
	}
	var vao, vbo uint32
	gl.GenVertexArrays(1, &vao)
	gl.GenBuffers(1, &vbo)
	gl.BindVertexArray(vao)
	gl.BindBuffer(gl.ARRAY_BUFFER, vbo)
	gl.BufferData(gl.ARRAY_BUFFER, len(verts)*4, gl.Ptr(verts), gl.STATIC_DRAW)
	gl.VertexAttribPointer(0, 3, gl.FLOAT, false, 6*4, gl.PtrOffset(0))
	gl.EnableVertexAttribArray(0)
	gl.VertexAttribPointer(1, 3, gl.FLOAT, false, 6*4, gl.PtrOffset(3*4))
	gl.EnableVertexAttribArray(1)
	return &Mesh{vao, vbo, int32(len(verts) / 6)}
}

func main() {
	if err := glfw.Init(); err != nil {
		panic(err)
	}
	defer glfw.Terminate()
	glfw.WindowHint(glfw.ContextVersionMajor, 4)
	glfw.WindowHint(glfw.ContextVersionMinor, 1)
	glfw.WindowHint(glfw.OpenGLProfile, glfw.OpenGLCoreProfile)
	window, _ := glfw.CreateWindow(width, height, "3D Spinor - Belt Trick", nil, nil)
	window.MakeContextCurrent()
	if err := gl.Init(); err != nil {
		panic(err)
	}

	gl.Enable(gl.DEPTH_TEST)

	vertexShader := `
	#version 410 core
	layout(location=0) in vec3 aPos;
	layout(location=1) in vec3 aColor;
	out vec3 Color;
	uniform mat4 MVP;
	void main(){
		Color = aColor;
		gl_Position = MVP * vec4(aPos,1.0);
	}`
	fragmentShader := `
	#version 410 core
	in vec3 Color;
	out vec4 FragColor;
	void main(){
		FragColor = vec4(Color,1.0);
	}`

	program := newProgram(vertexShader, fragmentShader)
	gl.UseProgram(program)
	mvpLoc := gl.GetUniformLocation(program, gl.Str("MVP\x00"))

	frame := makeFrame()
	cube := makeCube()
	sphere := makeSphere(0.07, 10, 10)

	proj := mgl32.Perspective(mgl32.DegToRad(45), float32(width)/height, 0.1, 100)
	view := mgl32.LookAtV(mgl32.Vec3{3, 2.5, 3}, mgl32.Vec3{0, 0, 0}, mgl32.Vec3{0, 1, 0})

	// 4 belts from cube corners to world anchors
	anchors := []mgl32.Vec3{
		{1, 0, 0}, {-1, 0, 0}, {0, 0, 1}, {0, 0, -1},
	}

	last := time.Now()
	angle := float32(0)
	for !window.ShouldClose() {
		gl.ClearColor(0.03, 0.03, 0.05, 1)
		gl.Clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)

		dt := float32(time.Since(last).Seconds())
		last = time.Now()
		angle += dt * 0.8 // rotation speed

		cubeModel := mgl32.HomogRotate3DY(angle).Mul4(mgl32.HomogRotate3DX(angle * 0.5))
		mvp := proj.Mul4(view).Mul4(cubeModel)
		gl.UniformMatrix4fv(mvpLoc, 1, false, &mvp[0])
		gl.BindVertexArray(cube.vao)
		gl.DrawElements(gl.TRIANGLES, cube.count, gl.UNSIGNED_INT, nil)

		// draw frame
		mvpFrame := proj.Mul4(view).Mul4(mgl32.Scale3D(1.5, 1.5, 1.5))
		gl.UniformMatrix4fv(mvpLoc, 1, false, &mvpFrame[0])
		gl.BindVertexArray(frame.vao)
		gl.DrawArrays(gl.LINES, 0, frame.count)

		// draw anchors (spheres)
		for _, a := range anchors {
			model := mgl32.Translate3D(a.X(), a.Y(), a.Z())
			mvpA := proj.Mul4(view).Mul4(model)
			gl.UniformMatrix4fv(mvpLoc, 1, false, &mvpA[0])
			gl.BindVertexArray(sphere.vao)
			gl.DrawArrays(gl.POINTS, 0, sphere.count)
		}

		window.SwapBuffers()
		glfw.PollEvents()
	}
}
