# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Minecraft-style 3D sandbox game built with Three.js, featuring RTX-like graphics with realistic water, procedural terrain, and post-processing effects. The game runs entirely in the browser using WebGL.

## Running the Project

Serve the project with any static HTTP server:
```bash
npx serve .
# or
python -m http.server 3000
```
Then open `http://localhost:3000` in a browser.

## Architecture

### Main Files

- **index.html** - Complete game implementation (~80KB). Contains:
  - Three.js scene setup with ES6 modules via import maps
  - Procedural world generation (terrain, caves, trees, mountains, lake)
  - First-person controls with physics (WASD + mouse)
  - Block placement/destruction system
  - Custom GLSL shaders for water (Gerstner waves, Fresnel, reflections)
  - Post-processing pipeline (EffectComposer, Bloom)
  - Procedural texture generation via Canvas API

- **water.js** - Standalone advanced water module (not used in main game). Features:
  - RealisticWater class with planar reflections via render targets
  - Gerstner wave vertex shader
  - Fresnel, refraction, caustics, foam fragment shader

- **main.js** - Simple Three.js demo with OrbitControls (separate from main game)

### Key Technical Details

**Rendering Pipeline:**
- WebGLRenderer with `powerPreference: "high-performance"` for discrete GPU
- 4096x4096 shadow maps with PCFSoftShadowMap
- ACES Filmic tone mapping
- EffectComposer with RenderPass, UnrealBloomPass, OutputPass

**Water System (in index.html):**
- Render-to-texture for reflections/refractions (reflectionRT, refractionRT)
- Mirror camera for planar reflections
- Custom ShaderMaterial with Gerstner waves computed on GPU
- Fresnel-based blending between reflection and refraction

**World Generation:**
- Procedural terrain with caves (tunnel-based algorithm)
- InstancedMesh for block rendering
- Procedural textures generated via Canvas (grass, dirt, bark, leaves, stone, sand)
- Decorations system (grass blades, flowers, ferns)

**Player Physics:**
- Gravity, jumping, collision detection
- PointerLock controls for FPS-style movement

## Three.js Version

Uses Three.js r160 via CDN import maps:
```javascript
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
```

## Language

Code comments and UI text are in Russian.
