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

func init() {
	runtime.LockOSThread()
}

type Ribbon struct {
	points []mgl32.Vec3
	vao    uint32
	vbo    uint32
	count  int32
}

func newRibbon(points []mgl32.Vec3) *Ribbon {
	var vao, vbo uint32
	gl.GenVertexArrays(1, &vao)
	gl.GenBuffers(1, &vbo)
	gl.BindVertexArray(vao)
	gl.BindBuffer(gl.ARRAY_BUFFER, vbo)

	// Build triangle strip (ribbon thickness)
	thickness := float32(0.05)
	var vertices []float32
	for i := 0; i < len(points)-1; i++ {
		p1 := points[i]
		p2 := points[i+1]
		dir := p2.Sub(p1).Normalize()
		side := dir.Cross(mgl32.Vec3{0, 1, 0}).Normalize().Mul(thickness)
		if side.Len() < 0.001 {
			side = mgl32.Vec3{thickness, 0, 0}
		}
		v1 := p1.Add(side)
		v2 := p1.Sub(side)
		v3 := p2.Add(side)
		v4 := p2.Sub(side)

		norm := dir.Cross(side).Normalize()

		for _, v := range []mgl32.Vec3{v1, v2, v3, v4} {
			vertices = append(vertices,
				v.X(), v.Y(), v.Z(),
				norm.X(), norm.Y(), norm.Z(),
			)
		}
	}

	gl.BufferData(gl.ARRAY_BUFFER, len(vertices)*4, gl.Ptr(vertices), gl.STATIC_DRAW)
	gl.EnableVertexAttribArray(0)
	gl.VertexAttribPointer(0, 3, gl.FLOAT, false, 6*4, gl.PtrOffset(0))
	gl.EnableVertexAttribArray(1)
	gl.VertexAttribPointer(1, 3, gl.FLOAT, false, 6*4, gl.PtrOffset(3*4))

	gl.BindVertexArray(0)

	return &Ribbon{points: points, vao: vao, vbo: vbo, count: int32(len(vertices) / 6)}
}

func makeShader(src string, stype uint32) uint32 {
	shader := gl.CreateShader(stype)
	csrc, free := gl.Strs(src + "\x00")
	gl.ShaderSource(shader, 1, csrc, nil)
	free()
	gl.CompileShader(shader)
	var status int32
	gl.GetShaderiv(shader, gl.COMPILE_STATUS, &status)
	if status == gl.FALSE {
		var logLen int32
		gl.GetShaderiv(shader, gl.INFO_LOG_LENGTH, &logLen)
		logBuf := make([]byte, logLen)
		gl.GetShaderInfoLog(shader, logLen, nil, &logBuf[0])
		log.Fatalf("shader compile: %s", logBuf)
	}
	return shader
}

func newProgram(vsrc, fsrc string) uint32 {
	vshader := makeShader(vsrc, gl.VERTEX_SHADER)
	fshader := makeShader(fsrc, gl.FRAGMENT_SHADER)
	prog := gl.CreateProgram()
	gl.AttachShader(prog, vshader)
	gl.AttachShader(prog, fshader)
	gl.LinkProgram(prog)
	gl.DeleteShader(vshader)
	gl.DeleteShader(fshader)
	return prog
}

func main() {
	if err := glfw.Init(); err != nil {
		panic(err)
	}
	defer glfw.Terminate()
	glfw.WindowHint(glfw.ContextVersionMajor, 4)
	glfw.WindowHint(glfw.ContextVersionMinor, 1)
	glfw.WindowHint(glfw.OpenGLProfile, glfw.OpenGLCoreProfile)
	glfw.WindowHint(glfw.Resizable, glfw.True)
	window, _ := glfw.CreateWindow(width, height, "Ribbon Demo", nil, nil)
	window.MakeContextCurrent()

	if err := gl.Init(); err != nil {
		panic(err)
	}

	gl.Enable(gl.DEPTH_TEST)
	gl.Enable(gl.CULL_FACE)
	gl.CullFace(gl.BACK)
	gl.Enable(gl.BLEND)
	gl.BlendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA)

	vertexShader := `
	#version 410 core
	layout(location=0) in vec3 aPos;
	layout(location=1) in vec3 aNormal;

	uniform mat4 MVP;
	uniform mat4 Model;
	uniform mat3 NormalMatrix;
	out vec3 Normal;
	out vec3 FragPos;

	void main(){
		FragPos = vec3(Model * vec4(aPos, 1.0));
		Normal = normalize(NormalMatrix * aNormal);
		gl_Position = MVP * vec4(aPos, 1.0);
	}
	`
	fragmentShader := `
	#version 410 core
	in vec3 Normal;
	in vec3 FragPos;
	out vec4 FragColor;

	uniform vec3 LightPos;
	uniform vec3 ViewPos;

	void main(){
		vec3 lightColor = vec3(1.0, 0.9, 0.7);
		vec3 objectColor = vec3(0.3, 0.6, 1.0);
		vec3 ambient = 0.2 * lightColor;
		vec3 norm = normalize(Normal);
		vec3 lightDir = normalize(LightPos - FragPos);
		float diff = max(dot(norm, lightDir), 0.0);
		vec3 diffuse = diff * lightColor;
		vec3 viewDir = normalize(ViewPos - FragPos);
		vec3 reflectDir = reflect(-lightDir, norm);
		float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
		vec3 specular = 0.3 * spec * lightColor;
		vec3 result = (ambient + diffuse + specular) * objectColor;
		FragColor = vec4(result, 0.7);
	}
	`

	program := newProgram(vertexShader, fragmentShader)
	gl.UseProgram(program)

	// Cube anchors
	anchors := []mgl32.Vec3{
		{0.5, 0.5, 0.5}, {-0.5, 0.5, 0.5}, {-0.5, -0.5, 0.5}, {0.5, -0.5, 0.5},
		{0.5, 0.5, -0.5}, {-0.5, 0.5, -0.5}, {-0.5, -0.5, -0.5}, {0.5, -0.5, -0.5},
	}

	// Generate ribbons connecting opposite faces
	var ribbons []*Ribbon
	for i := 0; i < 4; i++ {
		var pts []mgl32.Vec3
		for t := 0.0; t <= 1.0; t += 0.02 {
			a := anchors[i]
			b := anchors[i+4]
			mid := a.Mul(float32(1 - t)).Add(b.Mul(float32(t)))
			mid[0] += float32(math.Sin(float64(t*2*math.Pi))) * 0.2
			mid[1] += float32(math.Cos(float64(t*2*math.Pi))) * 0.2
			pts = append(pts, mid)
		}
		ribbons = append(ribbons, newRibbon(pts))
	}

	proj := mgl32.Perspective(mgl32.DegToRad(45), float32(width)/height, 0.1, 100)
	view := mgl32.LookAtV(mgl32.Vec3{3, 3, 3}, mgl32.Vec3{0, 0, 0}, mgl32.Vec3{0, 1, 0})
	model := mgl32.Ident4()
	mvpLoc := gl.GetUniformLocation(program, gl.Str("MVP\x00"))
	modelLoc := gl.GetUniformLocation(program, gl.Str("Model\x00"))
	normalLoc := gl.GetUniformLocation(program, gl.Str("NormalMatrix\x00"))
	lightLoc := gl.GetUniformLocation(program, gl.Str("LightPos\x00"))
	viewLoc := gl.GetUniformLocation(program, gl.Str("ViewPos\x00"))

	angle := float32(0)
	last := time.Now()
	for !window.ShouldClose() {
		gl.ClearColor(0.02, 0.02, 0.04, 1)
		gl.Clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)

		now := time.Now()
		dt := float32(now.Sub(last).Seconds())
		last = now
		angle += dt * 0.7

		model = mgl32.HomogRotate3DY(angle).Mul4(mgl32.HomogRotate3DX(angle * 0.7))
		mvp := proj.Mul4(view).Mul4(model)
		normalMat := model.Mat3()

		gl.UniformMatrix4fv(mvpLoc, 1, false, &mvp[0])
		gl.UniformMatrix4fv(modelLoc, 1, false, &model[0])
		gl.UniformMatrix3fv(normalLoc, 1, false, &normalMat[0])
		gl.Uniform3f(lightLoc, 2, 3, 3)
		gl.Uniform3f(viewLoc, 3, 3, 3)

		for _, r := range ribbons {
			gl.BindVertexArray(r.vao)
			gl.DrawArrays(gl.TRIANGLE_STRIP, 0, r.count)
		}

		window.SwapBuffers()
		glfw.PollEvents()
	}
}
