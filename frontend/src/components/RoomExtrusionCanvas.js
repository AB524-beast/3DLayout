"use client";

import React, { useEffect, useMemo, useRef, useCallback } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export default function RoomExtrusionCanvas({
  layoutData,
  activeFloor = 0,
  imageUrl = "",
  onRendererReady,
}) {
  const containerRef = useRef(null);
  const rendererRef = useRef(null);

  const rooms = useMemo(() => layoutData?.rooms || [], [layoutData]);
  const totalFloors = layoutData?.totalFloors || 1;

  const floorFilter = useMemo(() => {
    return (room) => {
      if (totalFloors > 1) {
        const roomLevel = Math.round((room.elevationZ || 0) / 3.0);
        return roomLevel === activeFloor;
      }
      return true;
    };
  }, [activeFloor, totalFloors]);

  useEffect(() => {
    if (!containerRef.current || !layoutData || !layoutData.rooms) return;

    const width = containerRef.current.clientWidth;
    const height = containerRef.current.clientHeight;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x060608);

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;
    containerRef.current.innerHTML = '';
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    if (onRendererReady) onRendererReady(renderer);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.rotateSpeed = 0.6;
    controls.zoomSpeed = 0.8;
    controls.panSpeed = 0.5;
    controls.minDistance = 1;
    controls.maxDistance = 80;
    controls.target.set(0, 0, 0);

    scene.add(new THREE.AmbientLight(0xffffff, 0.95));
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.4);
    dirLight.position.set(10, 25, 10);
    scene.add(dirLight);
    const dirLight2 = new THREE.DirectionalLight(0x8888ff, 0.15);
    dirLight2.position.set(-10, 15, -10);
    scene.add(dirLight2);

    const grid = new THREE.GridHelper(30, 30, 0x3b82f6, 0x1e293b);
    grid.position.y = 0.001;
    scene.add(grid);

    const wallHeight = 2.4;
    const wallThickness = 0.15;
    const MIN_WALL_LENGTH = 0.3;
    const COLLINEAR_ANGLE_EPS = 0.06;

    const wallMat = new THREE.MeshStandardMaterial({
      color: 0x1d4ed8,
      roughness: 0.25,
      metalness: 0.05,
    });
    const wallMatSelected = new THREE.MeshStandardMaterial({
      color: 0x3b82f6,
      roughness: 0.2,
      metalness: 0.1,
    });
    const openMat = new THREE.MeshStandardMaterial({
      color: 0x10b981,
      transparent: true,
      opacity: 0.35,
      roughness: 0.3,
    });

    const mergeCollinearWalls = (walls) => {
      if (!walls || walls.length === 0) return [];

      const merged = [];
      let current = { ...walls[0] };

      for (let i = 1; i < walls.length; i++) {
        const w = walls[i];
        const prevAngle = Math.atan2(current.y2 - current.y1, current.x2 - current.x1);
        const newAngle = Math.atan2(w.y2 - w.y1, w.x2 - w.x1);
        let angleDiff = Math.abs(prevAngle - newAngle);
        if (angleDiff > Math.PI) angleDiff = Math.abs(angleDiff - 2 * Math.PI);

        const touchesEnd =
          Math.hypot(w.x1 - current.x2, w.y1 - current.y2) < 0.05;

        if (touchesEnd && angleDiff < COLLINEAR_ANGLE_EPS) {
          current.x2 = w.x2;
          current.y2 = w.y2;
        } else {
          merged.push(current);
          current = { ...w };
        }
      }
      merged.push(current);
      return merged;
    };

    const allMinX = [], allMaxX = [], allMinZ = [], allMaxZ = [];

    const addWalls = () => {
      const filteredRooms = rooms.filter(floorFilter);

      filteredRooms.forEach((room, roomIdx) => {
        const baseZ = room.elevationZ || 0;
        const cleanedWalls = mergeCollinearWalls(room.walls || []);
        const mat = room.isOpenSpace ? openMat : (roomIdx === 0 ? wallMatSelected : wallMat);

        cleanedWalls.forEach((wall) => {
          const dx = wall.x2 - wall.x1;
          const dy = wall.y2 - wall.y1;
          const length = Math.hypot(dx, dy);
          if (length < MIN_WALL_LENGTH) return;

          allMinX.push(wall.x1, wall.x2);
          allMaxZ.push(wall.y1, wall.y2);

          const geo = new THREE.BoxGeometry(length, wallHeight, wallThickness);
          const mesh = new THREE.Mesh(geo, mat);

          const cx = (wall.x1 + wall.x2) / 2;
          const cy = (wall.y1 + wall.y2) / 2;

          mesh.position.set(cx, wallHeight / 2 + baseZ, cy);
          mesh.rotation.y = -Math.atan2(dy, dx);

          scene.add(mesh);
        });
      });
    };

    const renderBlueprint = () => {
      if (!imageUrl) {
        addWalls();
        fitCamera();
        return;
      }

      const textureLoader = new THREE.TextureLoader();
      textureLoader.load(imageUrl, (texture) => {
        texture.colorSpace = THREE.SRGBColorSpace;
        const img = texture.image;
        const aspect = img.width / img.height;

        const planeHeight = 14.0;
        const planeWidth = planeHeight * aspect;

        const planeGeo = new THREE.PlaneGeometry(planeWidth, planeHeight);
        const planeMat = new THREE.MeshBasicMaterial({
          map: texture,
          side: THREE.DoubleSide,
          transparent: true,
          opacity: 0.85,
        });

        const blueprintMesh = new THREE.Mesh(planeGeo, planeMat);
        blueprintMesh.rotation.x = -Math.PI / 2;
        blueprintMesh.position.set(0, -0.01, 0);
        scene.add(blueprintMesh);

        addWalls();
        fitCamera();
      });
    };

    const fitCamera = () => {
      if (allMinX.length === 0) {
        camera.position.set(0, 16, 16);
        controls.target.set(0, 0, 0);
        controls.update();
        return;
      }

      const minX = Math.min(...allMinX);
      const maxX = Math.max(...allMinX);
      const minZ = Math.min(...allMaxZ);
      const maxZ = Math.max(...allMaxZ);

      const centerX = (minX + maxX) / 2;
      const centerZ = (minZ + maxZ) / 2;
      const spanX = maxX - minX;
      const spanZ = maxZ - minZ;
      const maxSpan = Math.max(spanX, spanZ);

      const fov = camera.fov * (Math.PI / 180);
      const distance = (maxSpan / 2) / Math.tan(fov / 2) * 1.4;

      const finalDist = Math.max(distance, 5);

      camera.position.set(centerX, finalDist * 0.7, centerZ + finalDist * 0.7);
      controls.target.set(centerX, 0, centerZ);
      controls.update();
    };

    renderBlueprint();

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
      rendererRef.current = null;
      if (onRendererReady) onRendererReady(null);
    };
  }, [layoutData, activeFloor, imageUrl, rooms, floorFilter, onRendererReady]);

  return (
    <div
      ref={containerRef}
      className="w-full h-full min-h-[300px]"
      style={{ touchAction: 'none' }}
    />
  );
}
