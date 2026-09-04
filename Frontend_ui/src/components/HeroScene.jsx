import React, { useRef, useEffect } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { Stars, Html, Ring, useTexture, useGLTF } from '@react-three/drei';
import * as THREE from 'three';
import { Suspense, useState } from 'react';
import { gsap } from 'gsap';
import { useAppContext } from '../context/AppContext';

// Simple placeholder satellite, set up to use GLTF when available
const Satellite = ({ position, name, modelUrl }) => {
  const meshRef = useRef();

  useFrame(({ clock }) => {
    // Simple orbit logic
    const t = clock.getElapsedTime();
    const radius = 1.35; // slightly above earth (radius 1)
    // Offset each satellite's phase based on position
    const offset = position[0] * 2 + position[1];
    meshRef.current.position.x = Math.cos(t * 0.15 + offset) * radius;
    meshRef.current.position.z = Math.sin(t * 0.15 + offset) * radius;
    meshRef.current.position.y = Math.sin(t * 0.1 + offset) * 0.5; // slight inclination
    
    // Rotate satellite itself
    meshRef.current.rotation.x += 0.01;
    meshRef.current.rotation.y += 0.01;
  });

  return (
    <group ref={meshRef}>
      <mesh>
        <boxGeometry args={[0.05, 0.02, 0.05]} />
        <meshStandardMaterial color="#001133" metalness={0.8} roughness={0.2} />
      </mesh>
      {/* Solar panels */}
      <mesh position={[0.06, 0, 0]}>
        <planeGeometry args={[0.08, 0.04]} />
        <meshStandardMaterial color="#0000ff" side={THREE.DoubleSide} />
      </mesh>
      <mesh position={[-0.06, 0, 0]}>
        <planeGeometry args={[0.08, 0.04]} />
        <meshStandardMaterial color="#0000ff" side={THREE.DoubleSide} />
      </mesh>
      
      <Html distanceFactor={4} center>
        <div className="flex items-center gap-2 bg-[#06101c]/90 border border-[#00ff73]/50 text-[#00ff73] text-[9px] font-mono px-3 py-1.5 rounded-md backdrop-blur-md whitespace-nowrap shadow-[0_0_10px_rgba(0,255,115,0.15)]">
          <div className="w-1.5 h-1.5 rounded-full bg-[#00ff73] shadow-[0_0_8px_#00ff73] animate-pulse"></div>
          {name}
        </div>
      </Html>
    </group>
  );
};

const Earth = () => {
  const earthRef = useRef();

  // Load high-resolution realistic Earth textures
  const [colorMap, bumpMap, specularMap] = useTexture([
    'https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg',
    'https://unpkg.com/three-globe/example/img/earth-topology.png',
    'https://unpkg.com/three-globe/example/img/earth-water.png'
  ]);

  useFrame(() => {
    if (earthRef.current) {
      earthRef.current.rotation.y += 0.0005; // slow rotation
    }
  });

  return (
    <group ref={earthRef}>
      {/* Realistic full-color globe */}
      <mesh>
        <sphereGeometry args={[1, 64, 64]} />
        <meshPhongMaterial 
          map={colorMap}
          bumpMap={bumpMap}
          bumpScale={0.015}
          specularMap={specularMap}
          specular={new THREE.Color('grey')}
          shininess={30}
        />
      </mesh>

      {/* Very faint tactical Grid overlay to keep the UI theme */}
      <mesh>
        <sphereGeometry args={[1.003, 32, 32]} />
        <meshBasicMaterial 
          color="#00aaff" 
          wireframe={true} 
          transparent={true} 
          opacity={0.05} 
        />
      </mesh>

      {/* Faint Orbital Rings */}
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <ringGeometry args={[1.25, 1.252, 64]} />
        <meshBasicMaterial color="#006699" transparent={true} opacity={0.3} side={THREE.DoubleSide} />
      </mesh>
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <ringGeometry args={[1.45, 1.452, 64]} />
        <meshBasicMaterial color="#006699" transparent={true} opacity={0.15} side={THREE.DoubleSide} />
      </mesh>
    </group>
  );
};

// Converts Lat/Lon to 3D Cartesian coordinates
const latLongToVector3 = (lat, lon, radius) => {
  const phi = (90 - lat) * (Math.PI / 180);
  const theta = (lon + 180) * (Math.PI / 180);

  const x = -(radius * Math.sin(phi) * Math.cos(theta));
  const z = (radius * Math.sin(phi) * Math.sin(theta));
  const y = (radius * Math.cos(phi));

  return new THREE.Vector3(x, y, z);
};

const CameraController = () => {
  const { targetCoordinates, isAnimatingToTarget, setIsAnimatingToTarget, setIs2DView } = useAppContext();
  const { camera } = useThree();
  const targetPosRef = useRef(new THREE.Vector3(0, 0, 3.0));

  useEffect(() => {
    // Initial camera position setup (India)
    const initialPos = latLongToVector3(20.5937, 78.9629, 3.0);
    camera.position.copy(initialPos);
    camera.lookAt(0, 0, 0);
  }, [camera]);

  useEffect(() => {
    if (isAnimatingToTarget) {
      // Calculate target position on a sphere slightly larger than the earth for viewing
      const targetVec = latLongToVector3(targetCoordinates.lat, targetCoordinates.lng, 3.0);
      
      // Slerp to target for 1.5 seconds
      const dummyCamera = camera.clone();
      
      const animObj = { t: 0 };
      gsap.to(animObj, {
        t: 1,
        duration: 1.5,
        ease: "power2.inOut",
        onUpdate: () => {
          camera.position.copy(camera.position).lerp(targetVec, animObj.t);
          camera.lookAt(0, 0, 0);
        },
        onComplete: () => {
          // The Dive (Intense Zoom)
          // Move camera very close to the surface
          const diveTarget = latLongToVector3(targetCoordinates.lat, targetCoordinates.lng, 1.01);
          gsap.to(camera.position, {
            x: diveTarget.x,
            y: diveTarget.y,
            z: diveTarget.z,
            duration: 0.8,
            ease: "power4.in",
            onComplete: () => {
              setIs2DView(true);
              setIsAnimatingToTarget(false);
            }
          });
        }
      });
    }
  }, [isAnimatingToTarget, targetCoordinates, camera, setIs2DView, setIsAnimatingToTarget]);

  return null;
};

const TargetingReticle = () => {
  const { targetCoordinates, isAnimatingToTarget } = useAppContext();
  const [show, setShow] = useState(false);
  const meshRef = useRef();

  useEffect(() => {
    if (isAnimatingToTarget) {
      // Show reticle after slerp (1.5s) right before dive starts
      const timer = setTimeout(() => setShow(true), 1400);
      return () => clearTimeout(timer);
    } else {
      setShow(false);
    }
  }, [isAnimatingToTarget]);

  if (!show) return null;

  const pos = latLongToVector3(targetCoordinates.lat, targetCoordinates.lng, 1.02);

  // Orient reticle to face outwards
  const normal = pos.clone().normalize();
  const up = new THREE.Vector3(0, 1, 0);
  const quaternion = new THREE.Quaternion().setFromUnitVectors(up, normal);

  return (
    <mesh position={pos} quaternion={quaternion} ref={meshRef}>
      <ringGeometry args={[0.02, 0.03, 32]} />
      <meshBasicMaterial color="#ff2a2a" side={THREE.DoubleSide} transparent opacity={0.8} />
    </mesh>
  );
};


const HeroScene = () => {
  return (
    <Canvas camera={{ position: [0, 0, 3.0], fov: 45 }}>
      <Suspense fallback={null}>
      <color attach="background" args={['#050508']} />
      <ambientLight intensity={0.5} />
      <directionalLight position={[5, 3, 5]} intensity={1} color="#00f3ff" />
      <directionalLight position={[-5, -3, -5]} intensity={0.2} />
      
      <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />
      
      <Earth />
      
      <Satellite position={[1, 0, 0]} name="SENTINEL-1 | ACTIVE" modelUrl="/models/sentinel-1.gltf" />
      <Satellite position={[-0.5, 1.5, 0.5]} name="SENTINEL-2 | ACTIVE" modelUrl="/models/sentinel-2.gltf" />
      <Satellite position={[-1, -1, 0]} name="LANDSAT-9 | ACTIVE" modelUrl="/models/landsat-9.gltf" />
      
      {/* We need useState for TargetingReticle inside R3F, so let's import it there */}
      <ReticleWrapper />
      <CameraController />
      </Suspense>
    </Canvas>
  );
};

// Wrapper to use useState inside R3F context for Reticle
const ReticleWrapper = () => {
  return <TargetingReticle />;
}

export default HeroScene;
