package main

import (
	"log"

	"github.com/tfriedel6/canvas/sdlcanvas"
)

func main() {
	win, cv, err := sdlcanvas.CreateWindow(800, 600, "Canvas OpenGL Test")
	if err != nil {
		log.Fatal(err)
	}
	defer win.Destroy()

	for !win.Closed() {
		cv.SetFillStyle("#222")
		cv.FillRect(0, 0, 800, 600)

		cv.SetFillStyle("lime")
		cv.FillCircle(400, 300, 100)

		win.Update()
	}
}
