# ✈️ C++ Flight Simulator

A lightweight 3D flight simulator written in **C++17** with **OpenGL 3.3**, **GLFW**, **GLAD** and **GLM**.
Take off from a lit runway, fly over procedurally generated mountains and forests, and try to land without crashing.

## Features

**Graphics**
- Procedural terrain (fractal noise) with grass, forest, dry patches, rock and snow coloured by height and slope
- Sky gradient with sun disc and glow, distance haze, gamma-correct lighting and 4x MSAA
- About 1,500 trees in natural clumps and drifting low-poly clouds
- Runway with edge lines, dashed centreline, threshold bars, touchdown marks and glowing edge lights
- Hangar, control tower with a flashing beacon
- Detailed propeller aircraft with canopy, landing gear and a spinning propeller
- Soft ground shadow that fades with altitude

**Flight model**
- Throttle, drag, and speed that is lost climbing and gained diving
- Banking turns the aircraft
- Stall when airborne below stall speed (the nose drops)
- Takeoff speed required to leave the ground
- Crashes from hard landings, wing strikes, landing too fast off the runway, hills, the hangar and the tower

**HUD**
- Speed bar with stall marker, altitude bar, centre marker, flashing stall warning and crash overlay
- Live telemetry in the window title

## Controls

| Key | Action |
|-----|--------|
| **Mouse** | Pitch and heading |
| **W** | Throttle up |
| **S** | Brake |
| **Q** / **E** | Bank left / right |
| **C** | Toggle chase / cockpit view |
| **R** | Reset to the runway |
| **Esc** | Quit |

### Quick flight guide
1. Hold **W** to accelerate down the runway.
2. Once speed passes the takeoff speed (the speed bar passes the white stall marker), pull the mouse **up** gently to climb.
3. Use **Q / E** to bank and turn.
4. To land, line up with the runway, reduce speed with **S**, ease the nose down and touch down gently with the wings level.

## Building

### Requirements
- A C++17 compiler (MSVC, GCC or Clang)
- CMake 3.10 or newer
- OpenGL 3.3 capable GPU and drivers
- [GLFW](https://www.glfw.org/), [GLAD](https://glad.dav1d.de/) (OpenGL 3.3 core) and [GLM](https://github.com/g-truc/glm)

### Visual Studio (CMake project)
1. Open the project folder in Visual Studio (*File → Open → Folder*).
2. Make sure GLFW, GLAD and GLM are available to CMake (for example via vcpkg or by placing them in the project).
3. Select the `FlightSimulator` target and press **F5**.

### Command line
```bash
cmake -S . -B out/build
cmake --build out/build --config Release
```

## Project structure

```
FlightSimulator/
├── CMakeLists.txt
├── CMakePresets.json
├── FlightSimulator.cpp   # simulator source (rendering, physics, input)
├── FlightSimulator.h
└── README.md
```

## Tweaking

Key constants are at the top of `FlightSimulator.cpp`:

- `MAX_SPEED`, `STALL_SPEED`, `TAKEOFF_SPEED` for flight feel
- `FOG_DENSITY` for visibility distance
- `PLANE_SCALE` for aircraft size
- `terrainHeight()` for the shape of the land

## Roadmap ideas

- Text HUD with an artificial horizon and a vertical speed indicator
- Day/night cycle with illuminated runway
- Engine sound and wind audio
- Multiple airports and landing scoring

## License

Add the license of your choice here (for example MIT).