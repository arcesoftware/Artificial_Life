package main

import (
	"fmt"
	"image/color"
	"math"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

// BlochSpherePoints generates points (x, y, z) on the Bloch sphere.
// A general state is |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩
// The coordinates are: x = sin(θ)cos(φ), y = sin(θ)sin(φ), z = cos(θ)
func BlochSpherePoints(numPoints int) plotter.XYs {
	pts := make(plotter.XYs, 0)
	
	// Iterate over the polar angle (theta) and azimuthal angle (phi)
	for i := 0; i <= numPoints; i++ {
		theta := float64(i) / float64(numPoints) * math.Pi
		
		for j := 0; j <= numPoints*2; j++ { // Phi goes from 0 to 2*Pi
			phi := float64(j) / float64(numPoints*2) * 2 * math.Pi

			// Bloch sphere coordinates
			x := math.Sin(theta) * math.Cos(phi)
			y := math.Sin(theta) * math.Sin(phi)
			// z := math.Cos(theta)
			
			// For a 2D visualization of the sphere's surface, 
			// we plot the (x, y) coordinates.
			pts = append(pts, plotter.XY{X: x, Y: y})
		}
	}
	return pts
}

func main() {
	// --- Step 1: Generate the data for the Bloch Sphere ---
	const resolution = 20 // Number of steps for theta/phi
	data := BlochSpherePoints(resolution)

	// --- Step 2: Create a new plot ---
	p := plot.New()

	p.Title.Text = "Bloch Sphere 2D Projection (Spin-1/2 Topology)"
	p.X.Label.Text = "X (sin(θ)cos(φ))"
	p.Y.Label.Text = "Y (sin(θ)sin(φ))"

	// Ensure the plot is a circle (equal aspect ratio)
	p.X.Min = -1.1
	p.X.Max = 1.1
	p.Y.Min = -1.1
	p.Y.Max = 1.1
	p.HideAxes() // Hide axes for a cleaner sphere view
	
	// Add a Scatter Plot of the generated points
	scatter, err := plotter.NewScatter(data)
	if err != nil {
		fmt.Println("Error creating scatter plot:", err)
		return
	}
	
	// Style the points to look like a sphere boundary
	scatter.Color = color.RGBA{R: 0, G: 0, B: 255, A: 255} // Blue
	scatter.Radius = vg.Length(1) // Small points

	p.Add(scatter)
	
	// Add the center point (the origin)
	origin, _ := plotter.NewScatter(plotter.XYs{{X: 0, Y: 0}})
	origin.Color = color.Black
	origin.Radius = vg.Length(3)
	p.Add(origin)

	// --- Step 3: Save the plot to a file ---
	if err := p.Save(4*vg.Inch, 4*vg.Inch, "bloch_sphere.png"); err != nil {
		fmt.Println("Error saving plot:", err)
	} else {
		fmt.Println("Successfully generated Bloch Sphere 2D projection and saved to bloch_sphere.png")
	}
}

// NOTE: To run this code, you must first install the dependency:
// go get gonum.org/v1/plot
