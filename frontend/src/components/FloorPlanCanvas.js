"use client";

import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

export default function FloorPlanCanvas({ layoutData, activeFloor = 0, imageUrl = "" }) {
  const containerRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current || !layoutData || !layoutData.rooms) return;

    const width = containerRef.current.clientWidth;
    const height = containerRef.current.clientHeight;
    
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x060608);

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.set(0, 16, 16);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    containerRef.current.innerHTML = '';
    containerRef.current.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    scene.add(new THREE.AmbientLight(0xffffff, 0.95));
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.4);
    dirLight.position.set(10, 25, 10);
    scene.add(dirLight);

    const grid = new THREE.GridHelper(30, 30, 0x3b82f6, 0x1e293b);
    grid.position.y = 0.001; 
    scene.add(grid);

    const wallMat = new THREE.MeshStandardMaterial({ color: 0x1d4ed8, roughness: 0.2 }); 
    const openMat = new THREE.MeshStandardMaterial({ color: 0x10b981, transparent: true, opacity: 0.35 });

    const renderWalls = () => {
      layoutData.rooms.forEach((room) => {
        const roomLevel = Math.round(room.elevationZ / 3.0);
        if (layoutData.totalFloors > 1 && roomLevel !== activeFloor) return;

        const wallHeight = 2.4;
        const baseZ = room.elevationZ;

        room.walls.forEach((wall) => {
          const dx = wall.x2 - wall.x1;
          const dy = wall.y2 - wall.y1;
          const length = Math.hypot(dx, dy);
          
          if (length < 0.02) return;

          const geo = new THREE.BoxGeometry(length, wallHeight, 0.15);
          const mesh = new THREE.Mesh(geo, room.isOpenSpace ? openMat : wallMat);

          const cx = (wall.x1 + wall.x2) / 2;
          const cy = (wall.y1 + wall.y2) / 2;

          mesh.position.set(cx, (wallHeight / 2) + baseZ, cy);
          mesh.rotation.y = -Math.atan2(dy, dx);

          scene.add(mesh);
        });
      });
    };

    if (imageUrl) {
      const textureLoader = new THREE.TextureLoader();
      textureLoader.load(imageUrl, (texture) => {
        const img = texture.image;
        const aspect = img.width / img.height;
        
        const planeHeight = 14.0; 
        const planeWidth = planeHeight * aspect;

        const planeGeo = new THREE.PlaneGeometry(planeWidth, planeHeight); 
        const planeMat = new THREE.MeshBasicMaterial({
          map: texture,
          side: THREE.DoubleSide,
          transparent: true,
          opacity: 0.85
        });
        
        const blueprintMesh = new THREE.Mesh(planeGeo, planeMat);
        blueprintMesh.rotation.x = -Math.PI / 2;
        blueprintMesh.position.set(0, 0, 0);
        scene.add(blueprintMesh);

        renderWalls();
      });
    } else {
      renderWalls();
    }

    let frameId;
    const animate = () => {
      frameId = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    const handleResize = () => {
      if (!containerRef.current) return;
      const w = containerRef.current.clientWidth;
      const h = containerRef.current.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      cancelAnimationFrame(frameId);
      window.removeEventListener('resize', handleResize);
      controls.dispose();
      renderer.dispose();
    };
  }, [layoutData, activeFloor, imageUrl]);

  return <div ref={containerRef} className="w-full h-full min-h-[500px]" />;
}