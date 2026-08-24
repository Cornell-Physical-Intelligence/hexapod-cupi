import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';
import URDFLoader from 'urdf-loader';

const URDF_PATH = '/hexapod_mkii_mock_assy/urdf/hexapod_mkii_mock_assy.urdf';

export default function App() {
  const mountRef = useRef(null);
  const robotRef = useRef(null);
  const [joints, setJoints] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    const mount = mountRef.current;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x14161a);

    const camera = new THREE.PerspectiveCamera(50, mount.clientWidth / mount.clientHeight, 0.001, 100);
    camera.position.set(0.5, 0.35, 0.5);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    renderer.shadowMap.enabled = true;
    mount.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0.05, 0);

    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const dir = new THREE.DirectionalLight(0xffffff, 1.2);
    dir.position.set(1, 2, 1);
    dir.castShadow = true;
    scene.add(dir);

    const grid = new THREE.GridHelper(2, 40, 0x334, 0x223);
    scene.add(grid);
    scene.add(new THREE.AxesHelper(0.15));

    const loader = new URDFLoader();
    loader.packages = { hexapod_mkii_mock_assy: '/hexapod_mkii_mock_assy' };
    loader.loadMeshCb = (path, manager, done) => {
      new STLLoader(manager).load(
        path,
        (geom) => {
          geom.computeVertexNormals();
          const mesh = new THREE.Mesh(
            geom,
            new THREE.MeshStandardMaterial({ color: 0x999999, metalness: 0.2, roughness: 0.6 })
          );
          mesh.castShadow = true;
          done(mesh);
        },
        undefined,
        (err) => done(null, err)
      );
    };

    loader.load(
      URDF_PATH,
      (robot) => {
        // URDF is Z-up; three.js is Y-up.
        robot.rotation.x = -Math.PI / 2;
        robot.position.y = 0.12;
        scene.add(robot);
        robotRef.current = robot;
        setJoints(
          Object.values(robot.joints)
            .filter((j) => j.jointType === 'revolute')
            .map((j) => ({
              name: j.name,
              lower: j.limit.lower,
              upper: j.limit.upper,
              value: THREE.MathUtils.clamp(0, j.limit.lower, j.limit.upper),
            }))
        );
      },
      undefined,
      (err) => setError(String(err))
    );

    let raf;
    const animate = () => {
      raf = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    const onResize = () => {
      camera.aspect = mount.clientWidth / mount.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(mount.clientWidth, mount.clientHeight);
    };
    window.addEventListener('resize', onResize);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', onResize);
      renderer.dispose();
      mount.removeChild(renderer.domElement);
    };
  }, []);

  const setJoint = (name, value) => {
    robotRef.current?.setJointValue(name, value);
    setJoints((js) => js.map((j) => (j.name === name ? { ...j, value } : j)));
  };

  return (
    <div style={{ display: 'flex', height: '100%', color: '#dde', fontFamily: 'system-ui, sans-serif' }}>
      <div ref={mountRef} style={{ flex: 1, minWidth: 0 }} />
      <div style={{ width: 280, overflowY: 'auto', padding: 16, background: '#1b1e24', fontSize: 12 }}>
        <h2 style={{ margin: '0 0 4px', fontSize: 15 }}>hexapod_mkii_mock_assy</h2>
        <p style={{ margin: '0 0 12px', color: '#89a' }}>
          {error ? `Load error: ${error}` : `${joints.length} revolute joints`}
        </p>
        {joints.map((j) => (
          <label key={j.name} style={{ display: 'block', marginBottom: 10 }}>
            <span style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span>{j.name}</span>
              <span style={{ color: '#89a' }}>{Number(j.value).toFixed(2)}</span>
            </span>
            <input
              type="range"
              min={j.lower}
              max={j.upper}
              step={0.01}
              value={j.value}
              onChange={(e) => setJoint(j.name, parseFloat(e.target.value))}
              style={{ width: '100%' }}
            />
          </label>
        ))}
      </div>
    </div>
  );
}
