// main.go
package main

import (
	"fmt"
	"log"
	"math"
	"runtime"
	"time"

	"github.com/go-gl/gl/v4.1-core/gl"
	"github.com/go-gl/glfw/v3.3/glfw"
	"github.com/go-gl/mathgl/mgl32"
)

const (
	winW = 1280
	winH = 720

	// visualization params
	numBelts         = 12   // number of ribbons
	segmentsPerBelt  = 160  // points along each belt centerline
	anchorRadius     = 420  // radius of fixed anchor circle
	cubeAttachRadius = 120  // radius for attach points placed on cube faces
	helixAmp         = 18.0 // helix amplitude (centerline offset)
	helixTurns       = 4.0  // helix turns along belt length
	rotSpeedDeg      = 36.0 // degrees per second (rotation speed of cube)
	ribbonWidth      = 30.0 // width of ribbon
	pointSize        = 4.0
	lineWidth        = 2.0
)

// GL globals
var (
	prog uint32
	vao  uint32
	vbo  uint32
)

// camera / mouse
var (
	camYaw, camPitch float32 = -10, -12
	camDist          float32 = 1400
	lastX, lastY     float64
	dragging         bool
	autoCamSpin      float32 = 6.0 // deg/sec
)

func init() { runtime.LockOSThread() }

func main() {
	if err := glfw.Init(); err != nil {
		log.Fatalln("glfw init:", err)
	}
	defer glfw.Terminate()

	glfw.WindowHint(glfw.ContextVersionMajor, 4)
	glfw.WindowHint(glfw.ContextVersionMinor, 1)
	glfw.WindowHint(glfw.OpenGLProfile, glfw.OpenGLCoreProfile)
	glfw.WindowHint(glfw.OpenGLForwardCompatible, glfw.True)

	window, err := glfw.CreateWindow(winW, winH, "Dirac Belt Trick — Ribbons & Cube Faces", nil, nil)
	if err != nil {
		log.Fatalln("create window:", err)
	}
	window.MakeContextCurrent()
	if err := gl.Init(); err != nil {
		log.Fatalln("gl init:", err)
	}

	fmt.Println("OpenGL:", gl.GoStr(gl.GetString(gl.VERSION)))
	gl.Enable(gl.DEPTH_TEST)
	gl.Disable(gl.CULL_FACE) // show both sides of ribbon
	gl.PointSize(pointSize)
	gl.LineWidth(lineWidth)

	prog = newProgram(vertexShader, fragmentShader)
	gl.UseProgram(prog)

	// allocate VAO/VBO large enough for all ribbons (2 vertices per segment)
	maxVerts := numBelts * segmentsPerBelt * 2
	gl.GenVertexArrays(1, &vao)
	gl.BindVertexArray(vao)
	gl.GenBuffers(1, &vbo)
	gl.BindBuffer(gl.ARRAY_BUFFER, vbo)
	gl.BufferData(gl.ARRAY_BUFFER, maxVerts*3*4, nil, gl.DYNAMIC_DRAW) // float32 -> 4 bytes
	posLoc := uint32(gl.GetAttribLocation(prog, gl.Str("inPos\x00")))
	gl.EnableVertexAttribArray(posLoc)
	gl.VertexAttribPointer(posLoc, 3, gl.FLOAT, false, 0, nil)
	gl.BindVertexArray(0)

	// input
	window.SetCursorPosCallback(mouseMove)
	window.SetMouseButtonCallback(mouseBtn)
	window.SetScrollCallback(scroll)

	// uniforms
	proj := mgl32.Perspective(mgl32.DegToRad(45.0), float32(winW)/winH, 0.1, 5000.0)
	projLoc := gl.GetUniformLocation(prog, gl.Str("uProj\x00"))
	viewLoc := gl.GetUniformLocation(prog, gl.Str("uView\x00"))
	modelLoc := gl.GetUniformLocation(prog, gl.Str("uModel\x00"))
	colorLoc := gl.GetUniformLocation(prog, gl.Str("uColor\x00"))
	gl.UniformMatrix4fv(projLoc, 1, false, &proj[0])

	// anchors around a circle
	anchors := make([]mgl32.Vec3, numBelts)
	for i := 0; i < numBelts; i++ {
		a := float32(i) / float32(numBelts) * float32(2*math.Pi)
		x := float32(math.Cos(float64(a))) * anchorRadius
		y := float32(-220) // place anchors below cube for better visual
		z := float32(math.Sin(float64(a))) * anchorRadius
		anchors[i] = mgl32.Vec3{x, y, z}
	}

	// attach offsets now placed on different cube faces (round-robin across 6 faces)
	attachOffsets := make([]mgl32.Vec3, numBelts)
	faceCenters := []mgl32.Vec3{
		{1, 0, 0},  // +X
		{-1, 0, 0}, // -X
		{0, 1, 0},  // +Y
		{0, -1, 0}, // -Y
		{0, 0, 1},  // +Z
		{0, 0, -1}, // -Z
	}
	// spread belts with slight jitter across faces
	for i := 0; i < numBelts; i++ {
		f := i % len(faceCenters)
		center := faceCenters[f]
		// choose local polar coords on face to position attach point near face center with small offset

		angle := float32(i) * 2.0 * float32(math.Pi) / float32(numBelts)
		// local offset tangent to face:
		// pick two tangents on face by cross product with world axes
		var tangent1 mgl32.Vec3
		if math.Abs(float64(center.X())) > 0.5 {
			tangent1 = mgl32.Vec3{0, 1, 0}
		} else {
			tangent1 = mgl32.Vec3{1, 0, 0}
		}
		tangent2 := center.Cross(tangent1).Normalize()
		tangent1 = tangent2.Cross(center).Normalize()
		// position on face: center * cubeAttachRadius + small jitter on tangents
		jx := float32(math.Cos(float64(angle))) * (cubeAttachRadius*0.5 + float32(i%3)*8.0)
		jy := float32(math.Sin(float64(angle))) * (cubeAttachRadius*0.5 + float32((i+1)%3)*6.0)
		pos := center.Mul(cubeAttachRadius).Add(tangent1.Mul(jx)).Add(tangent2.Mul(jy))
		attachOffsets[i] = pos
	}

	// main loop
	prev := time.Now()
	rotAngle := float32(0)
	for !window.ShouldClose() {
		now := time.Now()
		dt := float32(now.Sub(prev).Seconds())
		prev = now

		rotAngle += rotSpeedDeg * dt
		rotRad := rotAngle * float32(math.Pi/180.0)

		// camera auto spin and view
		camYaw += autoCamSpin * dt
		camX := camDist * float32(math.Sin(float64(mgl32.DegToRad(camYaw))))
		camZ := camDist * float32(math.Cos(float64(mgl32.DegToRad(camYaw))))
		camY := camDist * float32(math.Sin(float64(mgl32.DegToRad(camPitch))))
		view := mgl32.LookAt(camX, camY, camZ, 0, 0, 0, 0, 1, 0)
		gl.UniformMatrix4fv(viewLoc, 1, false, &view[0])

		// cube model (apply rotation around Y)
		cubeModel := mgl32.HomogRotate3DY(rotRad)
		gl.UniformMatrix4fv(modelLoc, 1, false, &cubeModel[0])

		// Build ribbons vertices into a single CPU slice, then upload to VBO
		// Each belt produces 2*segmentsPerBelt vertices (left,right, left,right, ...)
		totalVerts := numBelts * segmentsPerBelt * 2
		verts := make([]float32, 0, totalVerts*3)

		for bi := 0; bi < numBelts; bi++ {
			anchor := anchors[bi]
			// compute attach point by rotating the attach offset with cubeModel
			att := cubeModel.Mul4x1(mgl32.Vec4{attachOffsets[bi].X(), attachOffsets[bi].Y(), attachOffsets[bi].Z(), 1.0})
			attachPt := mgl32.Vec3{att.X(), att.Y(), att.Z()}

			segDir := attachPt.Sub(anchor)
			segLen := segDir.Len()
			if segLen < 1e-6 {
				segLen = 1
			}
			dirNorm := segDir.Mul(1.0 / segLen)

			// choose stable perpendicular basis
			up := mgl32.Vec3{0, 1, 0}
			if math.Abs(float64(dotf(dirNorm, up))) > 0.999 {
				up = mgl32.Vec3{1, 0, 0}
			}
			perp1 := dirNorm.Cross(up).Normalize()    // first perp (width direction base)
			perp2 := dirNorm.Cross(perp1).Normalize() // second perp

			phaseBase := rotRad * 0.5 // rot/2 so 360deg reverses helix, 720deg returns

			for si := 0; si < segmentsPerBelt; si++ {
				t := float32(si) / float32(segmentsPerBelt-1)
				center := anchor.Add(segDir.Mul(t))
				phase := phaseBase + float32(2.0*math.Pi)*float32(helixTurns)*t
				taper := 1.0 - (0.5 * (float32(math.Sin(float64((t-0.5)*math.Pi))) + 1.0))
				amp := float32(helixAmp) * taper

				// helix offset around center using perp1/perp2 -> gives circular displacement
				helixOffset := perp1.Mul(float32(math.Cos(float64(phase)))).Add(perp2.Mul(float32(math.Sin(float64(phase)))))
				helixOffset = helixOffset.Mul(amp)
				ribbonCenter := center.Add(helixOffset)

				// choose width direction and rotate it by phase to get twisting ribbon
				widthDir := perp1.Mul(float32(math.Cos(float64(phase)))).Add(perp2.Mul(float32(math.Sin(float64(phase)))))
				widthDir = widthDir.Normalize()

				halfWidth := float32(ribbonWidth * 0.5)
				left := ribbonCenter.Add(widthDir.Mul(-halfWidth))
				right := ribbonCenter.Add(widthDir.Mul(halfWidth))

				// triangle strip order: left0, right0, left1, right1, ...
				verts = append(verts, left.X(), left.Y(), left.Z())
				verts = append(verts, right.X(), right.Y(), right.Z())
			}
		}

		// Upload verts to GPU
		gl.BindBuffer(gl.ARRAY_BUFFER, vbo)
		gl.BufferSubData(gl.ARRAY_BUFFER, 0, len(verts)*4, gl.Ptr(verts))

		// Render
		gl.ClearColor(0.02, 0.02, 0.05, 1.0)
		gl.Clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)
		gl.UseProgram(prog)

		// draw ribbons (color)
		gl.Uniform4f(colorLoc, 0.95, 0.75, 0.25, 1.0)
		gl.BindVertexArray(vao)
		// draw each belt's triangle strip sequentially
		vertsPerBelt := segmentsPerBelt * 2
		for bi := 0; bi < numBelts; bi++ {
			first := int32(bi * vertsPerBelt)
			count := int32(vertsPerBelt)
			gl.DrawArrays(gl.TRIANGLE_STRIP, first, count)
		}

		// draw attach knots (small points) on cube (recompute attach pts)
		attachVerts := make([]float32, 0, numBelts*3)
		for bi := 0; bi < numBelts; bi++ {
			att := cubeModel.Mul4x1(mgl32.Vec4{attachOffsets[bi].X(), attachOffsets[bi].Y(), attachOffsets[bi].Z(), 1.0})
			attachPt := mgl32.Vec3{att.X(), att.Y(), att.Z()}
			attachVerts = append(attachVerts, attachPt.X(), attachPt.Y(), attachPt.Z())
		}
		// upload attach verts at buffer offset after ribbons (safe because buffer big enough)
		attachOffsetBytes := maxVerts*3*4 - len(attachVerts)*4 // put near end (not necessary but safe)
		if attachOffsetBytes < 0 {
			attachOffsetBytes = 0
		}
		gl.BufferSubData(gl.ARRAY_BUFFER, int(attachOffsetBytes), len(attachVerts)*4, gl.Ptr(attachVerts))
		gl.Uniform4f(colorLoc, 0.2, 0.9, 1.0, 1.0)
		// draw points from that offset — easier to rebind a small temporary VBO; instead do a quick draw: create temp VBO
		var tmpVBO uint32
		gl.GenBuffers(1, &tmpVBO)
		gl.BindBuffer(gl.ARRAY_BUFFER, tmpVBO)
		gl.BufferData(gl.ARRAY_BUFFER, len(attachVerts)*4, gl.Ptr(attachVerts), gl.STREAM_DRAW)
		gl.EnableVertexAttribArray(posLoc)
		gl.VertexAttribPointer(posLoc, 3, gl.FLOAT, false, 0, nil)
		gl.DrawArrays(gl.POINTS, 0, int32(len(attachVerts)/3))
		// cleanup tmpVBO
		gl.DeleteBuffers(1, &tmpVBO)

		// draw cube wireframe
		drawWireCube(cubeModel)

		window.SwapBuffers()
		glfw.PollEvents()
	}

	// cleanup
	gl.BindBuffer(gl.ARRAY_BUFFER, 0)
	gl.BindVertexArray(0)
	gl.DeleteBuffers(1, &vbo)
	gl.DeleteVertexArrays(1, &vao)
}

// drawWireCube draws a simple wireframe cube (size chosen)
func drawWireCube(model mgl32.Mat4) {
	s := float32(80.0)
	corners := []mgl32.Vec3{
		{-s, -s, -s}, {s, -s, -s}, {s, s, -s}, {-s, s, -s},
		{-s, -s, s}, {s, -s, s}, {s, s, s}, {-s, s, s},
	}
	edges := [][2]int{
		{0, 1}, {1, 2}, {2, 3}, {3, 0},
		{4, 5}, {5, 6}, {6, 7}, {7, 4},
		{0, 4}, {1, 5}, {2, 6}, {3, 7},
	}
	verts := make([]float32, 0, len(edges)*2*3)
	for _, e := range edges {
		a := model.Mul4x1(mgl32.Vec4{corners[e[0]].X(), corners[e[0]].Y(), corners[e[0]].Z(), 1})
		b := model.Mul4x1(mgl32.Vec4{corners[e[1]].X(), corners[e[1]].Y(), corners[e[1]].Z(), 1})
		verts = append(verts, a.X(), a.Y(), a.Z(), b.X(), b.Y(), b.Z())
	}
	var tmpVBO uint32
	gl.GenBuffers(1, &tmpVBO)
	gl.BindBuffer(gl.ARRAY_BUFFER, tmpVBO)
	gl.BufferData(gl.ARRAY_BUFFER, len(verts)*4, gl.Ptr(verts), gl.STREAM_DRAW)
	posLoc := uint32(gl.GetAttribLocation(prog, gl.Str("inPos\x00")))
	gl.EnableVertexAttribArray(posLoc)
	gl.VertexAttribPointer(posLoc, 3, gl.FLOAT, false, 0, nil)
	colLoc := gl.GetUniformLocation(prog, gl.Str("uColor\x00"))
	gl.Uniform4f(colLoc, 0.9, 0.9, 0.9, 1.0)
	gl.DrawArrays(gl.LINES, 0, int32(len(verts)/3))
	gl.DeleteBuffers(1, &tmpVBO)
}

// input callbacks
func mouseMove(w *glfw.Window, xpos, ypos float64) {
	if dragging {
		camYaw += float32((xpos - lastX) * 0.25)
		camPitch += float32((ypos - lastY) * 0.15)
		if camPitch > 89 {
			camPitch = 89
		}
		if camPitch < -89 {
			camPitch = -89
		}
	}
	lastX = xpos
	lastY = ypos
}

func mouseBtn(w *glfw.Window, button glfw.MouseButton, action glfw.Action, mods glfw.ModifierKey) {
	if button == glfw.MouseButton1 {
		dragging = (action == glfw.Press)
	}
}

func scroll(w *glfw.Window, xoff, yoff float64) {
	camDist -= float32(yoff) * 40.0
	if camDist < 300 {
		camDist = 300
	}
	if camDist > 3500 {
		camDist = 3500
	}
}

// helpers
func dotf(a, b mgl32.Vec3) float32 { return a.X()*b.X() + a.Y()*b.Y() + a.Z()*b.Z() }

// ---------------- shaders ----------------
var vertexShader = `
#version 410 core
layout(location = 0) in vec3 inPos;
uniform mat4 uProj;
uniform mat4 uView;
uniform mat4 uModel;
void main() {
    gl_Position = uProj * uView * uModel * vec4(inPos, 1.0);
}
` + "\x00"

var fragmentShader = `
#version 410 core
uniform vec4 uColor;
out vec4 oColor;
void main() {
    oColor = uColor;
}
` + "\x00"

// ---------------- shader helpers ----------------
func compileShader(src string, shaderType uint32) uint32 {
	s := gl.CreateShader(shaderType)
	csources, free := gl.Strs(src)
	gl.ShaderSource(s, 1, csources, nil)
	free()
	gl.CompileShader(s)
	var status int32
	gl.GetShaderiv(s, gl.COMPILE_STATUS, &status)
	if status == gl.FALSE {
		var logLen int32
		gl.GetShaderiv(s, gl.INFO_LOG_LENGTH, &logLen)
		log := make([]byte, logLen+1)
		gl.GetShaderInfoLog(s, logLen, nil, &log[0])
	}
	return s
}

func newProgram(vs, fs string) uint32 {
	vsS := compileShader(vs, gl.VERTEX_SHADER)
	fsS := compileShader(fs, gl.FRAGMENT_SHADER)
	p := gl.CreateProgram()
	gl.AttachShader(p, vsS)
	gl.AttachShader(p, fsS)
	gl.LinkProgram(p)
	var status int32
	gl.GetProgramiv(p, gl.LINK_STATUS, &status)
	if status == gl.FALSE {
		var logLen int32
		gl.GetProgramiv(p, gl.INFO_LOG_LENGTH, &logLen)
		log := make([]byte, logLen+1)
		gl.GetProgramInfoLog(p, logLen, nil, &log[0])
	}
	gl.DeleteShader(vsS)
	gl.DeleteShader(fsS)
	return p
}
