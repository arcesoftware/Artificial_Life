package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"runtime"
	"strings"
	"time"
	"unsafe"

	"github.com/go-gl/gl/v4.1-core/gl"
	"github.com/go-gl/glfw/v3.3/glfw"
)

const (
	width     = 256
	height    = 256
	winWidth  = 800
	winHeight = 800
)

var (
	field   [height][width]float32
	texture uint32
	vao     uint32
	program uint32
)

func init() {
	runtime.LockOSThread()
}

// --- kernel and field update ---
func kernel(dx, dy int) float32 {
	r := math.Sqrt(float64(dx*dx + dy*dy))
	sigma := 1.5
	return float32(math.Exp(-r * r / (2 * sigma * sigma)))
}

func updateField() {
	var next [height][width]float32
	for j := 0; j < height; j++ {
		for i := 0; i < width; i++ {
			var sum float32
			for dy := -2; dy <= 2; dy++ {
				for dx := -2; dx <= 2; dx++ {
					ni := (i + dx + width) % width
					nj := (j + dy + height) % height
					sum += field[nj][ni] * kernel(dx, dy)
				}
			}
			g := 0.2 * (float32(math.Sin(float64(sum)*3.0)) - field[j][i])
			val := field[j][i] + g
			if val < 0 {
				val = 0
			}
			if val > 1 {
				val = 1
			}
			next[j][i] = val
		}
	}
	field = next
}

// --- OpenGL setup ---
func compileShader(source string, shaderType uint32) (uint32, error) {
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
		log := strings.Repeat("\x00", int(logLength+1))
		gl.GetShaderInfoLog(shader, logLength, nil, gl.Str(log))
		return 0, fmt.Errorf("shader compile error: %v", log)
	}
	return shader, nil
}

func newProgram() uint32 {
	vertexShaderSource := `
	#version 410
	layout(location = 0) in vec2 vert;
	out vec2 texCoord;
	void main() {
		texCoord = (vert + 1.0) / 2.0;
		gl_Position = vec4(vert, 0.0, 1.0);
	}` + "\x00"

	fragmentShaderSource := `
	#version 410
	in vec2 texCoord;
	out vec4 fragColor;
	uniform sampler2D tex;
	void main() {
		vec3 color = texture(tex, texCoord).rgb;
		fragColor = vec4(color, 1.0);
	}` + "\x00"

	vertexShader, err := compileShader(vertexShaderSource, gl.VERTEX_SHADER)
	if err != nil {
		log.Fatalln(err)
	}
	fragmentShader, err := compileShader(fragmentShaderSource, gl.FRAGMENT_SHADER)
	if err != nil {
		log.Fatalln(err)
	}

	prog := gl.CreateProgram()
	gl.AttachShader(prog, vertexShader)
	gl.AttachShader(prog, fragmentShader)
	gl.LinkProgram(prog)

	var status int32
	gl.GetProgramiv(prog, gl.LINK_STATUS, &status)
	if status == gl.FALSE {
		var logLength int32
		gl.GetProgramiv(prog, gl.INFO_LOG_LENGTH, &logLength)
		log := strings.Repeat("\x00", int(logLength+1))
		gl.GetProgramInfoLog(prog, logLength, nil, gl.Str(log))

	}
	gl.DeleteShader(vertexShader)
	gl.DeleteShader(fragmentShader)

	return prog
}

func initGL() {
	gl.GenTextures(1, &texture)
	gl.BindTexture(gl.TEXTURE_2D, texture)
	gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
	gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
	gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT)
	gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT)

	program = newProgram()
	gl.UseProgram(program)

	// fullscreen quad
	vertices := []float32{
		-1, -1,
		1, -1,
		1, 1,
		-1, 1,
	}
	indices := []uint32{0, 1, 2, 2, 3, 0}

	var vbo, ebo uint32
	gl.GenVertexArrays(1, &vao)
	gl.GenBuffers(1, &vbo)
	gl.GenBuffers(1, &ebo)

	gl.BindVertexArray(vao)
	gl.BindBuffer(gl.ARRAY_BUFFER, vbo)
	gl.BufferData(gl.ARRAY_BUFFER, len(vertices)*4, gl.Ptr(vertices), gl.STATIC_DRAW)

	gl.BindBuffer(gl.ELEMENT_ARRAY_BUFFER, ebo)
	gl.BufferData(gl.ELEMENT_ARRAY_BUFFER, len(indices)*4, gl.Ptr(indices), gl.STATIC_DRAW)

	gl.EnableVertexAttribArray(0)
	gl.VertexAttribPointer(0, 2, gl.FLOAT, false, 0, unsafe.Pointer(nil))
	gl.BindVertexArray(0)
}

// --- draw ---
func drawField() {
	pixels := make([]uint8, width*height*3)
	for j := 0; j < height; j++ {
		for i := 0; i < width; i++ {
			v := field[j][i]
			// purple → yellow gradient
			r := uint8(255 * v)
			g := uint8(255 * math.Sqrt(float64(v)))
			b := uint8(255 * (1 - v*v))
			idx := (j*width + i) * 3
			pixels[idx+0] = r
			pixels[idx+1] = g
			pixels[idx+2] = b
		}
	}
	gl.BindTexture(gl.TEXTURE_2D, texture)
	gl.TexImage2D(gl.TEXTURE_2D, 0, gl.RGB, int32(width), int32(height), 0, gl.RGB, gl.UNSIGNED_BYTE, gl.Ptr(pixels))
	gl.Clear(gl.COLOR_BUFFER_BIT)
	gl.UseProgram(program)
	gl.BindVertexArray(vao)
	gl.DrawElements(gl.TRIANGLES, 6, gl.UNSIGNED_INT, unsafe.Pointer(nil))
}

func main() {
	if err := glfw.Init(); err != nil {
		log.Fatalln("failed to init glfw:", err)
	}
	defer glfw.Terminate()

	window, err := glfw.CreateWindow(winWidth, winHeight, "Go Lenia — Modern OpenGL", nil, nil)
	if err != nil {
		panic(err)
	}
	window.MakeContextCurrent()

	if err := gl.Init(); err != nil {
		panic(fmt.Errorf("gl init failed: %v", err))
	}

	gl.ClearColor(0, 0, 0, 1)
	initGL()

	// initialize field
	for j := 0; j < height; j++ {
		for i := 0; i < width; i++ {
			field[j][i] = rand.Float32() * 0.8
		}
	}

	last := time.Now()
	for !window.ShouldClose() {
		now := time.Now()
		if now.Sub(last) > time.Millisecond*16 { // 60 FPS
			updateField()
			last = now
		}
		drawField()
		window.SwapBuffers()
		glfw.PollEvents()
	}
}
