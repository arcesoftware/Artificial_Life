package main

import (
	"fmt"
	"image"
	"image/color"
	"image/gif"
	"math"
	"os"
)

// --- External Dependencies (Conceptual, requires actual graphics library) ---
// NOTE: For a real rotating GIF, a library that handles 3D projection 
// onto a 2D canvas (like a custom renderer or a simpler 3D plotting library)
// would be required. This code focuses on the core mathematical logic.
// --------------------------------------------------------------------------

// Matrix for a rotation around the Y-axis (vertical axis of the sphere)
func rotationMatrixY(angle float64) [3][3]float64 {
	c := math.Cos(angle)
	s := math.Sin(angle)
	return [3][3]float64{
		{c, 0, s},
		{0, 1, 0},
		{-s, 0, c},
	}
}

// applyRotation multiplies the given point [x, y, z] by the rotation matrix.
func applyRotation(point [3]float64, matrix [3][3]float64) [3]float64 {
	var result [3]float64
	result[0] = matrix[0][0]*point[0] + matrix[0][1]*point[1] + matrix[0][2]*point[2]
	result[1] = matrix[1][0]*point[0] + matrix[1][1]*point[1] + matrix[1][2]*point[2]
	result[2] = matrix[2][0]*point[0] + matrix[2][1]*point[1] + matrix[2][2]*point[2]
	return result
}

// GenerateSphereVertices creates points on the surface of the Bloch sphere.
func GenerateSphereVertices(resolution int) [][3]float64 {
	vertices := make([][3]float64, 0)
	
	for i := 0; i <= resolution; i++ {
		theta := float64(i) / float64(resolution) * math.Pi
		for j := 0; j <= resolution*2; j++ {
			phi := float64(j) / float64(resolution*2) * 2 * math.Pi

			x := math.Sin(theta) * math.Cos(phi)
			y := math.Cos(theta)
			z := math.Sin(theta) * math.Sin(phi)
			
			vertices = append(vertices, [3]float64{x, y, z})
		}
	}
	return vertices
}

// Conceptual function to draw the sphere vertices onto an image
func drawFrame(img *image.Paletted, vertices [][3]float64) {
	// --- Simplistic 3D to 2D Projection ---
	// Project x and z coordinates, ignoring depth (z) for simplicity
	// In a real implementation, you'd apply a perspective projection matrix 
	// and handle Z-buffering/hidden surface removal.
	
	bounds := img.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()
	centerX := width / 2
	centerY := height / 2
	scale := float64(width) / 2.5 // Scale factor

	for _, v := range vertices {
		// Project 3D point (x, y, z) to 2D screen coordinate (px, py)
		px := int(scale*v[0]) + centerX
		py := int(scale*v[1]) + centerY
		
		// Set a pixel if within bounds (simplistic drawing)
		if px >= 0 && px < width && py >= 0 && py < height {
			img.SetColorIndex(px, py, 1) // Draw with the main color (index 1)
		}
	}
}

func main() {
	const (
		numFrames    = 60
		frameDelay   = 5 // Delay in 100ths of a second
		resolution   = 30
		imageWidth   = 200
		imageHeight  = 200
	)

	// 1. Initialize GIF structure and palette
	outGIF := &gif.GIF{}
	palette := []color.Color{
		color.White, // Background (index 0)
		color.RGBA{R: 0, G: 0, B: 255, A: 0xff}, // Sphere color (index 1)
	}

	// 2. Generate initial vertices
	initialVertices := GenerateSphereVertices(resolution)

	// 3. Loop through frames, apply rotation, and draw
	for i := 0; i < numFrames; i++ {
		// Calculate rotation angle (0 to 2*Pi over numFrames)
		angle := 2 * math.Pi * float64(i) / float64(numFrames)
		rotationMatrix := rotationMatrixY(angle)
		
		// Create a new image frame
		rect := image.Rect(0, 0, imageWidth, imageHeight)
		img := image.NewPaletted(rect, palette)

		// Transform and draw the vertices
		var rotatedVertices [][3]float64
		for _, v := range initialVertices {
			rotatedVertices = append(rotatedVertices, applyRotation(v, rotationMatrix))
		}
		
		drawFrame(img, rotatedVertices) // Draw the rotated points

		// Add frame to the GIF
		outGIF.Image = append(outGIF.Image, img)
		outGIF.Delay = append(outGIF.Delay, frameDelay)
	}

	// 4. Save the GIF file
	f, err := os.Create("bloch_sphere_rotating.gif")
	if err != nil {
		fmt.Println("Error creating file:", err)
		return
	}
	defer f.Close()

	if err := gif.EncodeAll(f, outGIF); err != nil {
		fmt.Println("Error encoding GIF:", err)
	} else {
		fmt.Printf("Successfully generated %d-frame rotating Bloch Sphere GIF and saved to bloch_sphere_rotating.gif\n", numFrames)
	}
}

/*
To execute this program, you will need to run:
go run main.go
*/
