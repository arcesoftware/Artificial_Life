package main

import (
	"fmt"
	"log"
	"math"
	"runtime"
	"strings"
	"time"
	"unsafe"

	"github.com/go-gl/gl/v4.1-core/gl"
	"github.com/go-gl/glfw/v3.3/glfw"
)

const (
	// Simulation Grid Size
	width  = 256
	height = 256
	// Window Size for Visualization
	winWidth  = 800
	winHeight = 800

	// --- Lenia Parameters for a stable "Glider" (Unicellular Mover) ---
	// These parameters are optimized for stable, self-propelling movement.
	R      = 7.0        // Radius of the kernel (Interaction range)
	sigmaK = 3.14159    // Kernel Gaussian sigma (Controls kernel spread/sharpness)
	muG    = 3.14159    // Growth function center (mu - optimal density for growth)
	sigmaG = 3.14159    // Growth function width (sigma - sharpness of the growth curve)
	dt     = 0.00618033 // Time step (smaller dt increases stability and frame rate)

	// Pre-calculate the integer radius for the convolution loop
	RConv = int(R)
)

var (
	field   [height][width]float32
	texture uint32
	vao     uint32
	program uint32
)

func init() {
	// GLFW requires the main thread to be locked
	runtime.LockOSThread()
}

// --- kernel and field update ---

// Gaussian Kernel for convolution (K(r))
func kernel(dx, dy int) float32 {
	r := math.Sqrt(float64(dx*dx + dy*dy))

	// Optimization: If r > R, return 0 (though the loop handles this, this ensures boundary check)
	if r > R {
		return 0.0
	}

	// Simple Gaussian kernel based on radius
	return float32(math.Exp(-r * r / (2 * sigmaK * sigmaK)))
}

// Standard Lenia Growth Function (G(S))
// Calculates the rate of change based on the summed activity S.
func growth(S float64) float32 {
	// This is a Gaussian bell curve centered at muG with width sigmaG.
	// The 2.0 factor is the maximum growth rate.
	exponent := -math.Pow(S-muG, 2) / (2 * math.Pow(sigmaG, 2))
	return float32(2.0*math.Exp(exponent) - 1.0)
}

func updateField() {
	var next [height][width]float32

	// Loop through every cell in the grid
	for j := 0; j < height; j++ {
		for i := 0; i < width; i++ {
			var sum float64

			// Convolution loop (neighborhood calculation)
			// This now correctly iterates up to RConv (integer radius of 5)
			for dy := -RConv; dy <= RConv; dy++ {
				for dx := -RConv; dx <= RConv; dx++ {

					// Apply toroidal boundary conditions (wraps around the edge)
					ni := (i + dx + width) % width
					nj := (j + dy + height) % height

					// Calculate the kernel weight and accumulate the activity sum (S)
					k := kernel(dx, dy)
					sum += float64(field[nj][ni]) * float64(k)
				}
			}

			// Lenia update rule: A_new = A_old + dt * G(S)
			val := field[j][i] + float32(dt)*growth(sum)

			// Clamp the field value (activity) to the range [0, 1]
			if val < 0 {
				val = 0
			}
			if val > 1 {
				val = 1
			}
			next[j][i] = val
		}
	}
	// Swap buffers: The new field becomes the current field
	field = next
}

// --- OpenGL setup (unchanged as it is standard) ---
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
		// Sample the texture for color data
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
	// Setup texture for the simulation grid
	gl.GenTextures(1, &texture)
	gl.BindTexture(gl.TEXTURE_2D, texture)
	gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
	gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
	gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT)
	gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT)

	program = newProgram()
	gl.UseProgram(program)

	// Define a fullscreen quad for drawing the texture
	vertices := []float32{
		-1, -1, // bottom left
		1, -1, // bottom right
		1, 1, // top right
		-1, 1, // top left
	}
	indices := []uint32{0, 1, 2, 2, 3, 0} // Two triangles forming the quad

	var vbo, ebo uint32
	gl.GenVertexArrays(1, &vao)
	gl.GenBuffers(1, &vbo)
	gl.GenBuffers(1, &ebo)

	gl.BindVertexArray(vao)
	gl.BindBuffer(gl.ARRAY_BUFFER, vbo)
	gl.BufferData(gl.ARRAY_BUFFER, len(vertices)*4, gl.Ptr(vertices), gl.STATIC_DRAW)

	gl.BindBuffer(gl.ELEMENT_ARRAY_BUFFER, ebo)
	gl.BufferData(gl.ELEMENT_ARRAY_BUFFER, len(indices)*4, gl.Ptr(indices), gl.STATIC_DRAW)

	// Configure vertex attribute 0 (position/texture coordinate)
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

			// --- Modern Lenia Color Ramp (Dark Blue/Black background to Yellow/White center) ---
			// Use a gamma curve to increase contrast
			intensity := math.Pow(float64(v), 0.5)

			// Dark Blue/Black to Bright Yellow/White transition
			r := uint8(255 * math.Min(1.0, 1.5*intensity)) // Red ramps up quickly
			g := uint8(255 * math.Min(1.0, 1.0*intensity)) // Green ramps up moderately
			b := uint8(255 * math.Max(0.0, 1.0-intensity)) // Blue decreases, creating yellow/red peak

			idx := (j*width + i) * 3
			pixels[idx+0] = r
			pixels[idx+1] = g
			pixels[idx+2] = b
		}
	}

	// Update the OpenGL texture with the new pixel data
	gl.BindTexture(gl.TEXTURE_2D, texture)
	gl.TexImage2D(gl.TEXTURE_2D, 0, gl.RGB, int32(width), int32(height), 0, gl.RGB, gl.UNSIGNED_BYTE, gl.Ptr(pixels))

	// Draw the quad
	gl.Clear(gl.COLOR_BUFFER_BIT)
	gl.UseProgram(program)
	gl.BindVertexArray(vao)
	gl.DrawElements(gl.TRIANGLES, 8, gl.UNSIGNED_INT, unsafe.Pointer(nil))
}

// initializeBlob centers a small, high-density Gaussian distribution on the field
func initializeBlob(cx, cy int, radius float64, initialVal float32) {
	sigma := radius / 3.0 // Width of the initial Gaussian
	for j := 0; j < height; j++ {
		for i := 0; i < width; i++ {
			dx := float64(i - cx)
			dy := float64(j - cy)
			r := math.Sqrt(dx*dx + dy*dy)

			// Gaussian blob initialization
			val := initialVal * float32(math.Exp(-r*r/(2*sigma*sigma)))

			// Clamp to [0, 1]
			if val > 1.0 {
				val = 1.0
			}
			field[j][i] = val
		}
	}
}

func main() {
	if err := glfw.Init(); err != nil {
		log.Fatalln("failed to init glfw:", err)
	}
	defer glfw.Terminate()

	glfw.WindowHint(glfw.ContextVersionMajor, 4)
	glfw.WindowHint(glfw.ContextVersionMinor, 1)
	glfw.WindowHint(glfw.OpenGLProfile, glfw.OpenGLCoreProfile)
	glfw.WindowHint(glfw.OpenGLForwardCompatible, glfw.True)

	window, err := glfw.CreateWindow(winWidth, winHeight, "Go Lenia — Unicellular Mover", nil, nil)
	if err != nil {
		panic(err)
	}
	window.MakeContextCurrent()

	if err := gl.Init(); err != nil {
		panic(fmt.Errorf("gl init failed: %v", err))
	}

	gl.ClearColor(0.05, 0.05, 0.1, 1) // Slightly dark blue background
	initGL()

	// Initialize field with a central, small Gaussian blob to seed the lifeform
	initializeBlob(width/2, height/2, 12.0, 1.0)

	last := time.Now()
	// Target approximately 60 FPS (16ms per frame) for smooth visuals
	frameTime := time.Millisecond * 16

	for !window.ShouldClose() {
		now := time.Now()

		// Run simulation step only if enough time has passed
		if now.Sub(last) >= frameTime {
			updateField()
			last = now
		}

		drawField()
		window.SwapBuffers()
		glfw.PollEvents()
	}
}
