'use client';

import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/examples/jsm/renderers/CSS2DRenderer.js';

export default function FloorPlanCanvas({ layoutData }) {
  const containerRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current) return;

    // --- 1. INITIALIZE THREE.JS SCENE SETUP ---
    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#0f172a'); // Slate 900 background

    // Camera perspective adjustment
    const camera = new THREE.PerspectiveCamera(
      60,
      containerRef.current.clientWidth / containerRef.current.clientHeight,
      0.1,
      1000
    );
    camera.position.set(0, 20, 25);

    // Standard WebGL Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
    containerRef.current.appendChild(renderer.domElement);

    // CSS2DRenderer for Dynamic Spatial Context Labels [cite: 21, 24]
    const labelRenderer = new CSS2DRenderer();
    labelRenderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
    labelRenderer.domElement.style.position = 'absolute';
    labelRenderer.domElement.style.top = '0px';
    labelRenderer.domElement.style.pointerEvents = 'none'; 
    containerRef.current.appendChild(labelRenderer.domElement);

    // Camera Controls (Rotate, Zoom, Pan) [cite: 24]
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    // Environmental Lighting Architecture
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(10, 30, 15);
    scene.add(directionalLight);

    // Add a subtle grid floor plane for spatial orientation
    const gridHelper = new THREE.GridHelper(40, 40, 0x475569, 0x334155);
    scene.add(gridHelper);

    // --- 2. PIPELINE: GEOMETRY REALIZATION & EXTRUSION [cite: 8, 9] ---
    if (layoutData && layoutData.rooms) {
      const WALL_HEIGHT = 3; // Custom height variable [cite: 19]
      const WALL_THICKNESS = 0.3;

      layoutData.rooms.forEach((room) => {
        // Step A: Extrude Wall Polygons from vector coordinates [cite: 19]
        // Expecting layoutData structure: coordinates mapped to room parameters [cite: 7, 8]
        room.walls.forEach((wall) => {
          // Calculate vector limits, length, and midpoints
          const start = new THREE.Vector2(wall.x1, wall.y1);
          const end = new THREE.Vector2(wall.x2, wall.y2);
          const distance = start.distanceTo(end);
          const midPoint = new THREE.Vector2().addVectors(start, end).multiplyScalar(0.5);
          const angle = Math.atan2(end.y - start.y, end.x - start.x);

          // Build volumetric BoxGeometry blocks [cite: 19]
          const wallGeometry = new THREE.BoxGeometry(distance, WALL_HEIGHT, WALL_THICKNESS);
          const wallMaterial = new THREE.MeshStandardMaterial({ 
            color: 0x6366f1, // Indigo 500
            roughness: 0.4 
          });
          
          const wallMesh = new THREE.Mesh(wallGeometry, wallMaterial);
          
          // Map technical structural coordinates directly into the 3D space [cite: 7, 19]
          wallMesh.position.set(midPoint.x, WALL_HEIGHT / 2, midPoint.y);
          wallMesh.rotation.y = -angle;
          scene.add(wallMesh);
        });

        // Step B: Spatial Context Labeling (CSS2DRenderer) [cite: 21]
        if (room.label) {
          const labelDiv = document.createElement('div');
          labelDiv.className = 'bg-slate-800/90 text-white text-xs font-semibold px-2 py-1 rounded shadow-md border border-slate-700 backdrop-blur-sm';
          labelDiv.textContent = `${room.label} (${room.dimensions || ''})`;

          const roomLabelObj = new CSS2DObject(labelDiv);
          // Anchor the hover label element directly above the specific floor plane bounds [cite: 21]
          roomLabelObj.position.set(room.centerX, 0.5, room.centerY);
          scene.add(roomLabelObj);
        }
      });
    }

    // --- 3. ANIMATION & RENDER LOOP ---
    const animate = () => {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
      labelRenderer.render(scene, camera);
    };
    animate();

    // Responsive Window Handler
    const handleResize = () => {
      if (!containerRef.current) return;
      const width = containerRef.current.clientWidth;
      const height = containerRef.current.clientHeight;

      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
      labelRenderer.setSize(width, height);
    };
    window.addEventListener('resize', handleResize);

    // Clean up resource cycles on unmount
    return () => {
      window.removeEventListener('resize', handleResize);
      if (containerRef.current) {
        containerRef.current.innerHTML = '';
      }
    };
  }, [layoutData]);

  return (
    <div className="relative w-full h-[600px] rounded-xl overflow-hidden shadow-2xl border border-slate-800 bg-slate-950">
      <div ref={containerRef} className="w-full h-full" />
    </div>
  );
}