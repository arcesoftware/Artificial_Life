// main.go
package main

import (
	"log"
	"runtime"
	"time"

	"github.com/go-gl/gl/v4.1-core/gl"
	"github.com/go-gl/glfw/v3.3/glfw"
	"github.com/go-gl/mathgl/mgl32"
)

const (
	winW = 1280
	winH = 720
)

func init() { runtime.LockOSThread() }

type Ribbon struct {
	vao   uint32
	vbo   uint32
	count int32
}

// buildRibbon constructs a small triangle strip between two moving points
func buildRibbon(a, b mgl32.Vec3, segments int, thickness float32) *Ribbon {
	var vao, vbo uint32
	gl.GenVertexArrays(1, &vao)
	gl.GenBuffers(1, &vbo)

	var vertices []float32
	for i := 0; i < segments; i++ {
		t0 := float32(i) / float32(segments)
		t1 := float32(i+1) / float32(segments)

		p1 := a.Mul(1 - t0).Add(b.Mul(t0))
		p2 := a.Mul(1 - t1).Add(b.Mul(t1))

		dir := p2.Sub(p1).Normalize()
		side := dir.Cross(mgl32.Vec3{0, 1, 0})
		if side.Len() < 0.001 {
			side = dir.Cross(mgl32.Vec3{1, 0, 0})
		}
		side = side.Normalize().Mul(thickness)
		norm := dir.Cross(side).Normalize()

		for _, v := range []mgl32.Vec3{
			p1.Add(side), p1.Sub(side),
			p2.Add(side), p2.Sub(side),
		} {
			vertices = append(vertices, v.X(), v.Y(), v.Z(),
				norm.X(), norm.Y(), norm.Z())
		}
	}

	gl.BindVertexArray(vao)
	gl.BindBuffer(gl.ARRAY_BUFFER, vbo)
	gl.BufferData(gl.ARRAY_BUFFER, len(vertices)*4, gl.Ptr(vertices), gl.DYNAMIC_DRAW)
	gl.EnableVertexAttribArray(0)
	gl.VertexAttribPointer(0, 3, gl.FLOAT, false, 6*4, gl.PtrOffset(0))
	gl.EnableVertexAttribArray(1)
	gl.VertexAttribPointer(1, 3, gl.FLOAT, false, 6*4, gl.PtrOffset(3*4))
	gl.BindVertexArray(0)

	return &Ribbon{vao: vao, vbo: vbo, count: int32(len(vertices) / 6)}
}

func makeShader(src string, stype uint32) uint32 {
	sh := gl.CreateShader(stype)
	csrc, free := gl.Strs(src + "\x00")
	gl.ShaderSource(sh, 1, csrc, nil)
	free()
	gl.CompileShader(sh)
	var status int32
	gl.GetShaderiv(sh, gl.COMPILE_STATUS, &status)
	if status == gl.FALSE {
		var logLen int32
		gl.GetShaderiv(sh, gl.INFO_LOG_LENGTH, &logLen)
		buf := make([]byte, logLen)
		gl.GetShaderInfoLog(sh, logLen, nil, &buf[0])
		log.Fatalf("Shader error: %s", buf)
	}
	return sh
}

func makeProgram(vs, fs string) uint32 {
	v := makeShader(vs, gl.VERTEX_SHADER)
	f := makeShader(fs, gl.FRAGMENT_SHADER)
	prog := gl.CreateProgram()
	gl.AttachShader(prog, v)
	gl.AttachShader(prog, f)
	gl.LinkProgram(prog)
	gl.DeleteShader(v)
	gl.DeleteShader(f)
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
	win, _ := glfw.CreateWindow(winW, winH, "Spinor Belt Trick Cube", nil, nil)
	win.MakeContextCurrent()

	if err := gl.Init(); err != nil {
		panic(err)
	}

	gl.Enable(gl.DEPTH_TEST)
	gl.Enable(gl.CULL_FACE)
	gl.CullFace(gl.BACK)
	gl.ClearColor(0.05, 0.05, 0.08, 1)

	vertex := `
	#version 410 core
	layout(location=0) in vec3 aPos;
	layout(location=1) in vec3 aNormal;
	uniform mat4 MVP;
	uniform mat4 Model;
	uniform mat3 NormalMatrix;
	out vec3 Normal;
	out vec3 FragPos;
	void main() {
		FragPos = vec3(Model * vec4(aPos, 1.0));
		Normal = normalize(NormalMatrix * aNormal);
		gl_Position = MVP * vec4(aPos, 1.0);
	}`

	fragment := `
	#version 410 core
	in vec3 Normal;
	in vec3 FragPos;
	out vec4 FragColor;
	uniform vec3 LightPos;
	uniform vec3 ViewPos;
	uniform vec3 BaseColor;
	void main() {
		vec3 lightColor = vec3(1.0, 0.9, 0.7);
		vec3 ambient = 0.2 * lightColor;
		vec3 norm = normalize(Normal);
		vec3 lightDir = normalize(LightPos - FragPos);
		float diff = max(dot(norm, lightDir), 0.0);
		vec3 diffuse = diff * lightColor;
		vec3 viewDir = normalize(ViewPos - FragPos);
		vec3 reflectDir = reflect(-lightDir, norm);
		float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
		vec3 specular = 0.3 * spec * lightColor;
		vec3 result = (ambient + diffuse + specular) * BaseColor;
		FragColor = vec4(result, 1.0);
	}`

	prog := makeProgram(vertex, fragment)
	gl.UseProgram(prog)

	mvpLoc := gl.GetUniformLocation(prog, gl.Str("MVP\x00"))
	modelLoc := gl.GetUniformLocation(prog, gl.Str("Model\x00"))
	normLoc := gl.GetUniformLocation(prog, gl.Str("NormalMatrix\x00"))
	lightLoc := gl.GetUniformLocation(prog, gl.Str("LightPos\x00"))
	viewLoc := gl.GetUniformLocation(prog, gl.Str("ViewPos\x00"))
	colorLoc := gl.GetUniformLocation(prog, gl.Str("BaseColor\x00"))

	// Cube vertices
	cubeVerts := []float32{
		// positions        // normals
		-0.5, -0.5, -0.5, 0, 0, -1,
		0.5, -0.5, -0.5, 0, 0, -1,
		0.5, 0.5, -0.5, 0, 0, -1,
		0.5, 0.5, -0.5, 0, 0, -1,
		-0.5, 0.5, -0.5, 0, 0, -1,
		-0.5, -0.5, -0.5, 0, 0, -1,
	}
	var cubeVAO, cubeVBO uint32
	gl.GenVertexArrays(1, &cubeVAO)
	gl.GenBuffers(1, &cubeVBO)
	gl.BindVertexArray(cubeVAO)
	gl.BindBuffer(gl.ARRAY_BUFFER, cubeVBO)
	gl.BufferData(gl.ARRAY_BUFFER, len(cubeVerts)*4, gl.Ptr(cubeVerts), gl.STATIC_DRAW)
	gl.EnableVertexAttribArray(0)
	gl.VertexAttribPointer(0, 3, gl.FLOAT, false, 6*4, gl.PtrOffset(0))
	gl.EnableVertexAttribArray(1)
	gl.VertexAttribPointer(1, 3, gl.FLOAT, false, 6*4, gl.PtrOffset(3*4))
	gl.BindVertexArray(0)

	// Ribbon endpoints
	anchors := []mgl32.Vec3{
		{1.5, 0, 0}, {-1.5, 0, 0},
		{0, 1.5, 0}, {0, -1.5, 0},
	}

	ribbons := []*Ribbon{
		buildRibbon(mgl32.Vec3{0.5, 0.5, 0.5}, anchors[0], 30, 0.03),
		buildRibbon(mgl32.Vec3{-0.5, 0.5, 0.5}, anchors[1], 30, 0.03),
		buildRibbon(mgl32.Vec3{0.5, 0.5, -0.5}, anchors[2], 30, 0.03),
		buildRibbon(mgl32.Vec3{-0.5, -0.5, -0.5}, anchors[3], 30, 0.03),
	}

	proj := mgl32.Perspective(mgl32.DegToRad(45), float32(winW)/winH, 0.1, 100)
	view := mgl32.LookAtV(mgl32.Vec3{3, 3, 3}, mgl32.Vec3{0, 0, 0}, mgl32.Vec3{0, 1, 0})

	last := time.Now()
	angle := float32(0)

	for !win.ShouldClose() {
		gl.Clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)
		now := time.Now()
		dt := float32(now.Sub(last).Seconds())
		last = now
		angle += dt * 0.6 // radians/sec

		// full 720° rotation
		model := mgl32.HomogRotate3DY(angle).Mul4(mgl32.HomogRotate3DX(angle * 0.7))
		mvp := proj.Mul4(view).Mul4(model)
		normMat := model.Mat3()

		gl.UniformMatrix4fv(mvpLoc, 1, false, &mvp[0])
		gl.UniformMatrix4fv(modelLoc, 1, false, &model[0])
		gl.UniformMatrix3fv(normLoc, 1, false, &normMat[0])
		gl.Uniform3f(lightLoc, 2, 3, 3)
		gl.Uniform3f(viewLoc, 3, 3, 3)

		// Draw cube
		gl.Uniform3f(colorLoc, 0.3, 0.6, 1.0)
		gl.BindVertexArray(cubeVAO)
		gl.DrawArrays(gl.TRIANGLES, 0, int32(len(cubeVerts)/6))

		// Draw ribbons
		gl.Uniform3f(colorLoc, 0.9, 0.6, 0.3)
		for _, rb := range ribbons {
			gl.BindVertexArray(rb.vao)
			gl.DrawArrays(gl.TRIANGLE_STRIP, 0, rb.count)
		}

		win.SwapBuffers()
		glfw.PollEvents()
	}
}
