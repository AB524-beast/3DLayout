'use client';

import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/examples/jsm/renderers/CSS2DRenderer.js';

export default function FloorPlanCanvas({ layoutData, activeRoom }) {
  const containerRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const controlsRef = useRef(null);
  const rendererRef = useRef(null);
  const labelRendererRef = useRef(null);
  const roomMeshesRef = useRef([]);
  const frameIdRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current || !layoutData) return;

    const container = containerRef.current;
    const width = container.clientWidth;
    const height = container.clientHeight;

    // --- 1. SCENE SETUP ---
    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#0f172a');
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
    camera.position.set(0, 20, 25);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    rendererRef.current = renderer;
    container.appendChild(renderer.domElement);

    // CSS2D Renderer for labels
    const labelRenderer = new CSS2DRenderer();
    labelRenderer.setSize(width, height);
    labelRenderer.domElement.style.position = 'absolute';
    labelRenderer.domElement.style.top = '0px';
    labelRenderer.domElement.style.left = '0px';
    labelRenderer.domElement.style.width = '100%';
    labelRenderer.domElement.style.height = '100%';
    labelRenderer.domElement.style.pointerEvents = 'none';
    labelRendererRef.current = labelRenderer;
    container.appendChild(labelRenderer.domElement);

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxPolarAngle = Math.PI / 2.1;
    controls.minDistance = 5;
    controls.maxDistance = 60;
    controlsRef.current = controls;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(10, 30, 15);
    directionalLight.castShadow = true;
    directionalLight.shadow.mapSize.width = 2048;
    directionalLight.shadow.mapSize.height = 2048;
    scene.add(directionalLight);

    const fillLight = new THREE.DirectionalLight(0x6366f1, 0.3);
    fillLight.position.set(-10, 10, -10);
    scene.add(fillLight);

    // Grid
    const gridHelper = new THREE.GridHelper(40, 40, 0x475569, 0x334155);
    scene.add(gridHelper);

    // Ground plane
    const groundGeometry = new THREE.PlaneGeometry(100, 100);
    const groundMaterial = new THREE.MeshStandardMaterial({ 
      color: 0x1e293b, 
      roughness: 0.8,
      metalness: 0.2
    });
    const ground = new THREE.Mesh(groundGeometry, groundMaterial);
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -0.01;
    ground.receiveShadow = true;
    scene.add(ground);

    // --- 2. BUILD ROOMS ---
    const WALL_HEIGHT = 3;
    const WALL_THICKNESS = 0.3;
    const roomMeshes = [];
    roomMeshesRef.current = roomMeshes;

    if (layoutData.rooms && Array.isArray(layoutData.rooms)) {
      layoutData.rooms.forEach((room, roomIndex) => {
        if (!room.walls || !Array.isArray(room.walls) || room.walls.length === 0) {
          return;
        }

        const roomGroup = new THREE.Group();
        roomGroup.name = `room-${roomIndex}`;

        // Build walls
        room.walls.forEach((wall, wallIndex) => {
          if (wall.x1 === undefined || wall.y1 === undefined || 
              wall.x2 === undefined || wall.y2 === undefined) {
            return;
          }

          const start = new THREE.Vector2(wall.x1, wall.y1);
          const end = new THREE.Vector2(wall.x2, wall.y2);
          const distance = start.distanceTo(end);
          
          if (distance < 0.1) return;

          const midPoint = new THREE.Vector2().addVectors(start, end).multiplyScalar(0.5);
          const angle = Math.atan2(end.y - start.y, end.x - start.x);

          // Wall mesh
          const wallGeometry = new THREE.BoxGeometry(distance, WALL_HEIGHT, WALL_THICKNESS);
          const wallMaterial = new THREE.MeshStandardMaterial({ 
            color: 0x6366f1,
            roughness: 0.4,
            metalness: 0.1
          });
          
          const wallMesh = new THREE.Mesh(wallGeometry, wallMaterial);
          wallMesh.position.set(midPoint.x, WALL_HEIGHT / 2, midPoint.y);
          wallMesh.rotation.y = -angle;
          wallMesh.castShadow = true;
          wallMesh.receiveShadow = true;
          wallMesh.userData = { roomIndex, type: 'wall' };
          roomGroup.add(wallMesh);

          // Wall cap
          const capGeometry = new THREE.BoxGeometry(distance, 0.1, WALL_THICKNESS + 0.05);
          const capMaterial = new THREE.MeshStandardMaterial({ color: 0x4f46e5 });
          const capMesh = new THREE.Mesh(capGeometry, capMaterial);
          capMesh.position.set(midPoint.x, WALL_HEIGHT, midPoint.y);
          capMesh.rotation.y = -angle;
          roomGroup.add(capMesh);
        });

        // Generate floor from the explicit, pre-ordered outline the backend
        // provides (room.outline). This does NOT depend on the order of
        // room.walls, so it stays correct even for the non-rectangular
        // fallback shape (bay windows, angled walls, disconnected wall
        // segments, etc.) where wall order carries no geometric meaning.
        const outlinePoints = Array.isArray(room.outline)
          ? room.outline
              .filter(p => p && typeof p.x === 'number' && typeof p.y === 'number')
              .map(p => new THREE.Vector2(p.x, p.y))
          : [];

        if (outlinePoints.length >= 3) {
          try {
            const shape = new THREE.Shape(outlinePoints);
            const floorGeometry = new THREE.ShapeGeometry(shape);
            const floorMaterial = new THREE.MeshStandardMaterial({ 
              color: 0x1e293b,
              roughness: 0.9,
              metalness: 0.0,
              side: THREE.DoubleSide,
              transparent: true,
              opacity: 0.8
            });
            const floor = new THREE.Mesh(floorGeometry, floorMaterial);
            floor.rotation.x = -Math.PI / 2;
            floor.position.y = 0.01;
            floor.receiveShadow = true;
            floor.userData = { roomIndex, type: 'floor' };
            roomGroup.add(floor);
          } catch (e) {
            console.warn('Could not generate floor for room:', e);
          }
        }

        // Room label
        if (room.label) {
          const labelDiv = document.createElement('div');
          labelDiv.className = 'bg-slate-800/90 text-white text-xs font-semibold px-2 py-1 rounded shadow-md border border-slate-700 backdrop-blur-sm whitespace-nowrap';
          labelDiv.textContent = `${room.label} (${room.dimensions || ''})`;

          const roomLabelObj = new CSS2DObject(labelDiv);
          roomLabelObj.position.set(room.centerX || 0, WALL_HEIGHT + 0.5, room.centerY || 0);
          roomGroup.add(roomLabelObj);
        }

        scene.add(roomGroup);
        roomMeshes.push({ group: roomGroup, index: roomIndex, originalColor: 0x6366f1 });
      });
    }

    // --- 3. ANIMATION LOOP ---
    const animate = () => {
      frameIdRef.current = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
      labelRenderer.render(scene, camera);
    };
    animate();

    // --- 4. RESIZE HANDLER ---
    const handleResize = () => {
      if (!container) return;
      const newWidth = container.clientWidth;
      const newHeight = container.clientHeight;

      camera.aspect = newWidth / newHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(newWidth, newHeight);
      labelRenderer.setSize(newWidth, newHeight);
    };
    window.addEventListener('resize', handleResize);

    // --- 5. CLEANUP ---
    return () => {
      window.removeEventListener('resize', handleResize);
      if (frameIdRef.current) cancelAnimationFrame(frameIdRef.current);
      
      scene.traverse((object) => {
        if (object.geometry) object.geometry.dispose();
        if (object.material) {
          if (Array.isArray(object.material)) {
            object.material.forEach(m => m.dispose());
          } else {
            object.material.dispose();
          }
        }
      });
      
      renderer.dispose();
      
      if (container) {
        container.innerHTML = '';
      }
    };
  }, [layoutData]);

  // --- HIGHLIGHT ACTIVE ROOM ---
  useEffect(() => {
    if (!sceneRef.current || roomMeshesRef.current.length === 0) return;

    roomMeshesRef.current.forEach(({ group, index }) => {
      group.traverse((child) => {
        if (child.isMesh && child.userData.type === 'wall') {
          const material = child.material.clone();
          
          if (activeRoom === index) {
            // Highlight: brighter, more saturated indigo
            material.color.setHex(0x818cf8);
            material.emissive.setHex(0x312e81);
            material.emissiveIntensity = 0.3;
          } else if (activeRoom === null) {
            // Default state
            material.color.setHex(0x6366f1);
            material.emissive.setHex(0x000000);
            material.emissiveIntensity = 0;
          } else {
            // Dim other rooms
            material.color.setHex(0x4338ca);
            material.emissive.setHex(0x000000);
            material.emissiveIntensity = 0;
          }
          
          child.material = material;
        }
        
        if (child.isMesh && child.userData.type === 'floor') {
          const material = child.material.clone();
          
          if (activeRoom === index) {
            material.color.setHex(0x312e81);
            material.opacity = 0.6;
          } else if (activeRoom === null) {
            material.color.setHex(0x1e293b);
            material.opacity = 0.8;
          } else {
            material.color.setHex(0x0f172a);
            material.opacity = 0.3;
          }
          
          child.material = material;
        }
      });
    });

    // Camera focus on active room
    if (activeRoom !== null && layoutData && layoutData.rooms[activeRoom]) {
      const room = layoutData.rooms[activeRoom];
      const targetX = room.centerX || 0;
      const targetZ = room.centerY || 0;
      
      if (controlsRef.current) {
        controlsRef.current.target.set(targetX, 0, targetZ);
      }
    }
  }, [activeRoom, layoutData]);

  return (
    <div className="relative w-full h-[600px] rounded-xl overflow-hidden shadow-2xl border border-slate-800 bg-slate-950">
      <div ref={containerRef} className="w-full h-full" />
      {!layoutData && (
        <div className="absolute inset-0 flex items-center justify-center text-slate-500">
          <div className="text-center">
            <svg className="mx-auto h-16 w-16 mb-4 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
            </svg>
            <p className="text-sm font-medium">Upload a blueprint to generate 3D view</p>
          </div>
        </div>
      )}
    </div>
  );
}